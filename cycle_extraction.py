"""
Functions used for cycle decomposition.
Counterpart of CoRAL cycle_decomposition.py
"""

from __future__ import annotations

import logging
import math
 
import pyomo.environ as pyo
 
from cycle_extractor.datatypes import (
    BreakpointGraph,
    CycleExtractionOptions,
    CycleSolution,
    EdgeToCN,
    EdgeType,
    InitialSolution,
    ModelMetadata,
    ModelType,
    OptimizationWalk,
    SolverOptions,
)
from cycle_extractor.models import concrete, cycle_utils
 
logger = logging.getLogger(__name__)
 

# Entry point
def extract_cycles(
    graph: BreakpointGraph,
    options: CycleExtractionOptions | None = None,
    solver_options: SolverOptions | None = None,
    *,
    initial_solution: InitialSolution | None = None,
) -> CycleSolution:
    """Decompose ``graph`` into cycles and paths.
 
    Dispatches on ``options.model_type`` and nothing else. ``initial_solution``
    is accepted as a warm start for the min-cycles model if the caller has one;
    this function never produces one itself.
    """
    options = options or CycleExtractionOptions()
    solver_options = solver_options or SolverOptions()
 
    if graph.num_edges == 0:
        logger.warning("Empty breakpoint graph; nothing to extract.")
        return _empty_solution(options, graph)
 
    logger.info(
        "Extracting cycles for amplicon %d: %d edges, %d nodes, "
        "%d subpath constraints, total length-weighted CN %.2f.",
        graph.amplicon_idx + 1,
        graph.num_edges,
        graph.num_nodes,
        len(graph.longest_path_constraints),
        graph.total_weights,
    )
 
    match options.model_type:
        case ModelType.MIN_CYCLES:
            solution = _extract_min_cycles(
                graph, options, solver_options, initial_solution
            )
        case ModelType.MAX_WEIGHT:
            solution = _extract_max_weight(graph, options, solver_options)
        case _:
            raise ValueError(f"Unsupported model type: {options.model_type}")
 
    logger.info(
        "Extracted %d cycles and %d paths; explained %.2f/%.2f CN; "
        "satisfied %d/%d subpath constraints.",
        solution.num_cycles,
        solution.num_paths,
        solution.total_weights_included,
        graph.total_weights,
        solution.num_pc_satisfied,
        len(graph.longest_path_constraints),
    )
    return solution
 
 
# --------------------------------------------------------------------------- #
# Min-cycles: search over the cycle budget k                                   #
# --------------------------------------------------------------------------- #
 
 
def _initial_k(graph: BreakpointGraph) -> int:
    """Starting cycle budget. A budget, not a policy: the model is infeasible
    if k is too small, which the loop below detects and corrects."""
    return min(max(10, graph.num_disc_edges // 2), graph.num_edges)
 
 
def _extract_min_cycles(
    graph: BreakpointGraph,
    options: CycleExtractionOptions,
    solver_options: SolverOptions,
    initial_solution: InitialSolution | None = None,
) -> CycleSolution:
    """Minimize the number of structures explaining the graph.
 
    Doubles k while the model is infeasible. Infeasibility at k > num_edges is
    returned as-is; it is not silently converted into a greedy run.
    """
    k = _initial_k(graph)
    solution = _empty_solution(options, graph)
 
    while k <= graph.num_edges:
        logger.debug("Solving cycle-minimization model with k = %d.", k)
        model = concrete.build_model(
            graph,
            k=k,
            model_type=ModelType.MIN_CYCLES,
            options=options,
            initial_solution=initial_solution,
        )
        solution = _solve_and_parse(
            model, graph, options, solver_options, k=k, round_idx=None
        )
 
        if solution.termination_condition != pyo.TerminationCondition.infeasible:
            break
 
        if k == graph.num_edges:
            break
        # Clamp rather than plain doubling, so k = num_edges -- the largest
        # budget that can matter -- is always attempted before giving up.
        next_k = min(k * 2, graph.num_edges)
        logger.info("Infeasible at k = %d; raising to %d.", k, next_k)
        k = next_k
 
    if solution.termination_condition == pyo.TerminationCondition.infeasible:
        logger.warning(
            "Cycle minimization remained infeasible past k = %d (num_edges).",
            graph.num_edges,
        )
 
    solution.model_metadata = _metadata(options, graph, k=k)
    return solution
 
 
# --------------------------------------------------------------------------- #
# Max-weight: iterative peel                                                   #
# --------------------------------------------------------------------------- #
 
 
def _extract_max_weight(
    graph: BreakpointGraph,
    options: CycleExtractionOptions,
    solver_options: SolverOptions,
) -> CycleSolution:
    """Extract one heaviest structure at a time, subtracting it and repeating.
 
    Each round solves a k=1 model against the *residual* copy numbers and the
    still-unsatisfied subpath constraints. The graph itself is never modified.
    """
    total_weights = graph.total_weights
    residual_cn = EdgeToCN.from_graph(graph)
    is_pc_unsatisfied = [True] * len(graph.longest_path_constraints)
 
    solution = _empty_solution(options, graph)
    remaining_weights = total_weights
    last_weight = math.inf
    rounds = 0
 
    for round_idx in itertools.count():
        if not _should_continue(
            last_weight=last_weight,
            remaining_weights=remaining_weights,
            total_weights=total_weights,
            num_unsatisfied=sum(is_pc_unsatisfied),
            num_constraints=len(is_pc_unsatisfied),
            options=options,
        ):
            break
 
        logger.debug(
            "Round %d: residual CN %.2f/%.2f, %d/%d constraints unsatisfied.",
            round_idx + 1,
            remaining_weights,
            total_weights,
            sum(is_pc_unsatisfied),
            len(is_pc_unsatisfied),
        )
 
        model = concrete.build_model(
            graph,
            k=1,
            model_type=ModelType.MAX_WEIGHT,
            options=options,
            residual_cn=residual_cn,
            is_pc_unsatisfied=is_pc_unsatisfied,
            remaining_weights=remaining_weights,
        )
        current = _solve_and_parse(
            model, graph, options, solver_options, k=1, round_idx=round_idx
        )
 
        if _is_terminal(current):
            logger.info(
                "Stopping at round %d: solver reported %s.",
                round_idx + 1,
                current.termination_condition,
            )
            solution.solver_status = current.solver_status
            solution.termination_condition = current.termination_condition
            break
 
        extracted = _extract_single_structure(current)
        if extracted is None:
            logger.info(
                "Stopping at round %d: model returned no structure.",
                round_idx + 1,
            )
            break
 
        walk, weight, satisfied, is_cycle = extracted
        _accumulate(solution, walk, weight, satisfied, is_cycle=is_cycle)
        _subtract_walk(
            residual_cn, walk, weight, floor=options.min_structure_weight
        )
        for pc_idx in satisfied:
            is_pc_unsatisfied[pc_idx] = False
 
        remaining_weights -= current.total_weights_included
        solution.total_weights_included += current.total_weights_included
        solution.solver_status = current.solver_status
        solution.termination_condition = current.termination_condition
        last_weight = weight
        rounds = round_idx + 1
 
        if current.total_weights_included < (
            options.min_marginal_weight * total_weights
        ):
            logger.info(
                "Stopping at round %d: marginal explained CN below %.4f "
                "of total.",
                rounds,
                options.min_marginal_weight,
            )
            break
 
    solution.model_metadata = _metadata(options, graph, k=rounds)
    return solution
 
 
def _should_continue(
    *,
    last_weight: float,
    remaining_weights: float,
    total_weights: float,
    num_unsatisfied: int,
    num_constraints: int,
    options: CycleExtractionOptions,
) -> bool:
    """Whether another peel round is warranted.
 
    Pure, so it is testable without a solver. Three clauses: the last structure
    must have cleared the reporting floor, and either enough copy number or
    enough subpath constraints must remain unexplained.
    """
    if last_weight < options.min_structure_weight:
        return False
    cn_remains = remaining_weights > (
        1.0 - options.min_fraction_explained
    ) * total_weights
    constraints_remain = num_unsatisfied > math.floor(
        (1.0 - options.min_fraction_constraints) * num_constraints
    )
    return cn_remains or constraints_remain
 
 
def _extract_single_structure(
    solution: CycleSolution,
) -> tuple[OptimizationWalk, float, Sequence[int], bool] | None:
    """Pull the single structure out of a k=1 solution, cycle preferred."""
    if solution.num_cycles > 0:
        return (
            solution.walks.cycles[0],
            solution.walk_weights.cycles[0],
            solution.satisfied_pc.cycles[0],
            True,
        )
    if solution.num_paths > 0:
        return (
            solution.walks.paths[0],
            solution.walk_weights.paths[0],
            solution.satisfied_pc.paths[0],
            False,
        )
    return None
 
 
def _accumulate(
    solution: CycleSolution,
    walk: OptimizationWalk,
    weight: float,
    satisfied: Sequence[int],
    *,
    is_cycle: bool,
) -> None:
    """Append one extracted structure to the running solution."""
    bucket = "cycles" if is_cycle else "paths"
    getattr(solution.walks, bucket).append(walk)
    getattr(solution.walk_weights, bucket).append(weight)
    getattr(solution.satisfied_pc, bucket).append(list(satisfied))
    solution.satisfied_pc_set |= set(satisfied)
 
 
_RESIDUAL_FIELD_BY_EDGE_TYPE = {
    EdgeType.SEQUENCE: "sequence",
    EdgeType.CONCORDANT: "concordant",
    EdgeType.DISCORDANT: "discordant",
    EdgeType.SOURCE: "source",
    EdgeType.SINK: "source",
}
"""Source and sink edges are two ends of the same source edge, so both debit
``EdgeToCN.source``. Synthetic terminals carry no copy number and are absent."""
 
 
def _subtract_walk(
    residual_cn: EdgeToCN,
    walk: OptimizationWalk,
    weight: float,
    *,
    floor: float,
) -> None:
    """Debit an extracted structure's copy number from the residual.
 
    Values below ``floor`` are clamped to zero, so an edge that is exhausted to
    within reporting resolution stops constraining later rounds.
    """
    for edge_id, multiplicity in walk.items():
        field_name = _RESIDUAL_FIELD_BY_EDGE_TYPE.get(edge_id.type)
        if field_name is None:  # synthetic terminal
            continue
        residual: dict[int, float] = getattr(residual_cn, field_name)
        if edge_id.idx not in residual:
            logger.warning(
                "Walk references %s absent from the residual; skipping.",
                edge_id,
            )
            continue
        residual[edge_id.idx] -= multiplicity * weight
        if residual[edge_id.idx] < floor:
            residual[edge_id.idx] = 0.0
 
 
# --------------------------------------------------------------------------- #
# Shared plumbing                                                              #
# --------------------------------------------------------------------------- #
 
 
def _solve_and_parse(
    model: pyo.Model,
    graph: BreakpointGraph,
    options: CycleExtractionOptions,
    solver_options: SolverOptions,
    *,
    k: int,
    round_idx: int | None,
) -> CycleSolution:
    """The only place in this module that touches a solver."""
    model_name = _model_name(graph, options, k=k, round_idx=round_idx)
    solver = cycle_utils.get_solver(solver_options)
    results = solver.solve(model, model_filepath=_model_path(solver_options, model_name))
    return cycle_utils.parse_solver_output(results, model, graph, k=k)
 
 
def _is_terminal(solution: CycleSolution) -> bool:
    """Whether the solver returned nothing usable and retrying is pointless."""
    if solution.termination_condition == pyo.TerminationCondition.infeasible:
        return True
    return (
        solution.termination_condition == pyo.TerminationCondition.maxTimeLimit
        and solution.solver_status == pyo.SolverStatus.aborted
    )
 
 
def _empty_solution(
    options: CycleExtractionOptions, graph: BreakpointGraph
) -> CycleSolution:
    return CycleSolution(
        solver_status=pyo.SolverStatus.unknown,
        termination_condition=pyo.TerminationCondition.unknown,
        model_metadata=_metadata(options, graph, k=0),
    )
 
 
def _metadata(
    options: CycleExtractionOptions, graph: BreakpointGraph, *, k: int
) -> ModelMetadata:
    return ModelMetadata(
        model_type=options.model_type,
        k=k,
        path_constraint_weight=options.path_constraint_weight,
        total_weights=graph.total_weights,
        min_structure_weight=options.min_structure_weight,
        num_path_constraints=len(graph.longest_path_constraints),
    )
 
 
def _model_name(
    graph: BreakpointGraph,
    options: CycleExtractionOptions,
    *,
    k: int,
    round_idx: int | None,
) -> str:
    suffix = f"_round{round_idx + 1}" if round_idx is not None else f"_k{k}"
    return (
        f"amplicon_{graph.amplicon_idx + 1}_{options.model_type.value}{suffix}"
    )
 
 
def _model_path(solver_options: SolverOptions, model_name: str) -> str:
    prefix = "_".join(
        part
        for part in (solver_options.output_prefix, solver_options.model_prefix)
        if part
    )
    stem = f"{prefix}_{model_name}" if prefix else model_name
    return str(solver_options.output_dir / "models" / stem)