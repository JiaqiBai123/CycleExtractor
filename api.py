"""
CE's interface for CoRAL.
"""
 
from __future__ import annotations
 
import logging
from collections.abc import Sequence
 
from cycle_extractor import extraction
from cycle_extractor.datatypes import (
    GRAPH_INPUT_FIELDS,
    BreakpointGraph,
    CycleExtractionOptions,
    CycleSolution,
    Edge,
    ModelType,
    SolverOptions,
)
 
logger = logging.getLogger(__name__)
 
 
def extract_cycles(
    graph: BreakpointGraph,
    options: CycleExtractionOptions | None = None,
    solver_options: SolverOptions | None = None,
) -> CycleSolution:
    """Decompose a breakpoint graph into cycles and paths.
 
    Args:
        graph: A ``BreakpointGraph``, or any object exposing the attributes in
            ``GRAPH_INPUT_FIELDS`` (CoRAL's graph qualifies). Not modified.
        options: Model choice and stopping rules. Defaults to ``MAX_WEIGHT``
            with hard path constraints.
        solver_options: Solver, thread count, time limit. Defaults to HiGHS.
 
    Returns:
        A ``CycleSolution``. Check ``termination_condition`` before trusting
        it: on infeasibility or timeout the solution may be empty or partial,
        and CE does not silently retry with a different model.
 
    Raises:
        TypeError: ``graph`` lacks required attributes.
        ValueError: ``graph`` or ``options`` violate a precondition.
    """
    options = options or CycleExtractionOptions()
    solver_options = solver_options or SolverOptions()
 
    graph = _coerce_graph(graph)
    _validate_options(options)
    _validate_graph(graph)
 
    if graph.num_edges == 0:
        logger.warning("Empty breakpoint graph; nothing to extract.")
        return extraction.empty_solution(options, graph)
 
    logger.info(
        "Extracting cycles for amplicon %d (%s): %d edges, %d nodes, "
        "%d subpath constraints, total length-weighted CN %.2f.",
        graph.amplicon_idx + 1,
        options.model_type.value,
        graph.num_edges,
        graph.num_nodes,
        len(graph.longest_path_constraints),
        graph.total_weights,
    )
 
    match options.model_type:
        case ModelType.MIN_CYCLES:
            solution = extraction.extract_min_cycles(
                graph, options, solver_options
            )
        case ModelType.MAX_WEIGHT:
            solution = extraction.extract_max_weight(
                graph, options, solver_options
            )
        case _:
            raise ValueError(f"Unsupported model type: {options.model_type}")
 
    logger.info(
        "Extracted %d cycles and %d paths; explained %.2f/%.2f CN; "
        "satisfied %d/%d subpath constraints; terminated as %s.",
        solution.num_cycles,
        solution.num_paths,
        solution.total_weights_included,
        graph.total_weights,
        solution.num_pc_satisfied,
        len(graph.longest_path_constraints),
        solution.termination_condition,
    )
    return solution
 
 
# --------------------------------------------------------------------------- #
# Coercion                                                                     #
# --------------------------------------------------------------------------- #
 
 
def _coerce_graph(graph: object) -> BreakpointGraph:
    """Narrow any structurally compatible graph to CE's input type.
 
    A ``BreakpointGraph`` -- including a subclass, which is what CoRAL's graph
    should be -- passes through untouched, so the caller's container identity
    and its ``node_adjacencies`` iteration order are preserved. Anything else
    is projected field by field; edge lists are shared by reference, never
    copied, which is safe because extraction treats the graph as read-only.
 
    This names no caller and imports nothing from any caller. It is a
    normalizing constructor for CE's own type, not a per-caller adapter -- and
    once CoRAL's graph subclasses CE's, it degenerates to a passthrough.
    """
    if isinstance(graph, BreakpointGraph):
        return graph
 
    missing = [
        field for field in GRAPH_INPUT_FIELDS if not hasattr(graph, field)
    ]
    if missing:
        raise TypeError(
            f"{type(graph).__name__} is not usable as a breakpoint graph; "
            f"missing attribute(s): {', '.join(missing)}."
        )
    logger.debug(
        "Projecting %s onto BreakpointGraph (edges shared by reference).",
        type(graph).__name__,
    )
    return BreakpointGraph(
        **{field: getattr(graph, field) for field in GRAPH_INPUT_FIELDS}
    )
 
 
# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
 
 
def _validate_options(options: CycleExtractionOptions) -> None:
    for name in ("min_fraction_explained", "min_fraction_constraints"):
        value = getattr(options, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {value}.")
    for name in (
        "min_structure_weight",
        "path_constraint_weight",
        "min_marginal_weight",
    ):
        value = getattr(options, name)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
 
 
def _validate_graph(graph: BreakpointGraph) -> None:
    """Check the preconditions the model construction assumes.
 
    Each of these otherwise surfaces as an opaque failure deep in model
    building or in the solver: a KeyError from an adjacency lookup, an
    unbounded relaxation, or an IndexError while reporting constraints.
    """
    if graph.num_edges == 0:
        return  # trivially valid; the caller short-circuits
 
    if graph.max_cn <= 0.0:
        raise ValueError(
            f"max_cn must be positive to bound the copy-number variables, "
            f"got {graph.max_cn}."
        )
 
    edge_groups: tuple[tuple[str, Sequence[Edge]], ...] = (
        ("sequence", graph.sequence_edges),
        ("concordant", graph.concordant_edges),
        ("discordant", graph.discordant_edges),
        ("source", graph.source_edges),
    )
    for kind, edges in edge_groups:
        for idx, edge in enumerate(edges):
            if edge.cn < 0.0:
                raise ValueError(
                    f"{kind} edge {idx} has negative copy number {edge.cn}."
                )
            if edge.cn > graph.max_cn:
                raise ValueError(
                    f"{kind} edge {idx} has copy number {edge.cn} above "
                    f"max_cn {graph.max_cn}."
                )
 
    for idx, edge in enumerate(graph.sequence_edges):
        if edge.start > edge.end:
            raise ValueError(
                f"sequence edge {idx} is inverted: {edge.start} > {edge.end}."
            )
 
    _validate_adjacencies(graph)
    _validate_path_constraints(graph)
 
 
def _validate_adjacencies(graph: BreakpointGraph) -> None:
    bounds = {
        "sequence": graph.num_seq_edges,
        "concordant": graph.num_conc_edges,
        "discordant": graph.num_disc_edges,
        "source": graph.num_src_edges,
    }
    for node, adjacency in graph.node_adjacencies.items():
        for kind, bound in bounds.items():
            for edge_idx in getattr(adjacency, kind):
                if not 0 <= edge_idx < bound:
                    raise ValueError(
                        f"node {node} lists {kind} edge {edge_idx}, but the "
                        f"graph has {bound} such edge(s)."
                    )
 
    for node, edge_idxs in graph.endnode_adjacencies.items():
        for edge_idx in edge_idxs:
            if not 0 <= edge_idx < graph.num_seq_edges:
                raise ValueError(
                    f"end node {node} lists sequence edge {edge_idx}, but the "
                    f"graph has {graph.num_seq_edges} sequence edge(s)."
                )
 
 
def _validate_path_constraints(graph: BreakpointGraph) -> None:
    """``pc_idx`` indexes back into ``path_constraints`` for reporting.
 
    An out-of-range index is an error only when ``path_constraints`` is
    populated. A caller may legitimately supply finalized constraints alone --
    the model only needs their ``edge_counts`` -- so that case is a warning:
    extraction will work, but the cycles file cannot name which read supported
    which constraint.
    """
    if not graph.longest_path_constraints:
        return
 
    if not graph.path_constraints:
        logger.warning(
            "%d finalized subpath constraint(s) supplied without the "
            "corresponding path_constraints; extraction is unaffected but "
            "constraints cannot be reported by origin.",
            len(graph.longest_path_constraints),
        )
        return
 
    num_constraints = len(graph.path_constraints)
    for constraint in graph.longest_path_constraints:
        if not 0 <= constraint.pc_idx < num_constraints:
            raise ValueError(
                f"finalized subpath constraint references pc_idx "
                f"{constraint.pc_idx}, but only {num_constraints} path "
                f"constraint(s) are present."
            )
        if not constraint.edge_counts:
            raise ValueError(
                f"finalized subpath constraint pc_idx {constraint.pc_idx} is "
                f"empty; it constrains nothing."
            )