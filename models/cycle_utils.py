from __future__ import annotations
 
import logging
import pathlib
from typing import TYPE_CHECKING, Any
 
import pyomo.environ as pyo
import pyomo.opt
from pyomo.common.errors import ApplicationError
 
from cycle_extractor.datatypes import (
    BreakpointGraph,
    CycleSolution,
    EdgeId,
    EdgeType,
    ModelMetadata,
    OptimizationWalk,
    Solver,
    SolverOptions,
    WalkData,
)
 
if TYPE_CHECKING:
    from cycle_extractor.models.concrete import CycleModel
 
logger = logging.getLogger(__name__)
 
SELECTED = 0.9
"""A binary at or above this counts as 1. Solvers return 0.9999999 routinely."""
 
INTEGRALITY_TOLERANCE = 1e-4
"""How far ``f / F`` may sit from an integer before it is worth a warning."""
 
 
# --------------------------------------------------------------------------- #
# Solver                                                                       #
# --------------------------------------------------------------------------- #
 
 
class SolverWrapper:
    """A solver plus the log destination its plugin needs.
 
    Pyomo's solver plugins disagree about logging. The shell-based plugins
    accept a ``logfile`` argument; the APPSI plugins raise
    ``NotImplementedError`` on it. ``supports_logfile`` records which kind this
    is, so a log path is passed only where it will be honoured -- otherwise a
    solve that would have succeeded fails on a logging argument.
 
    A solver that raises is converted into an aborted result rather than an
    exception. The peel checks ``termination_condition`` after every round, so
    a failure on round 7 keeps the six walks already extracted; letting the
    exception escape would discard them.
    """
 
    def __init__(
        self,
        solver: pyomo.opt.base.OptSolver,
        log_path: pathlib.Path | None,
        *,
        supports_logfile: bool,
    ) -> None:
        self.solver = solver
        self.log_path = log_path
        self.supports_logfile = supports_logfile
 
    def solve(self, model: pyo.Model, **kwargs: Any) -> pyomo.opt.SolverResults:
        options: dict[str, Any] = {"load_solutions": False, **kwargs}
        if self.log_path is not None and self.supports_logfile:
            options["logfile"] = f"{self.log_path}.log"
        try:
            return self.solver.solve(model, **options)
        except (ValueError, RuntimeError, ApplicationError) as error:
            logger.error(
                "Solver %s raised %s: %s",
                self._describe(),
                type(error).__name__,
                error,
            )
            return _aborted_results()
 
    def _describe(self) -> str:
        """A name for the log. Pyomo's plugin classes disagree about which
        attribute carries it -- the APPSI wrappers have neither ``name`` nor
        ``type`` -- so this falls back to the class."""
        for attribute in ("name", "type"):
            value = getattr(self.solver, attribute, None)
            if isinstance(value, str):
                return value
        return type(self.solver).__name__
 
 
def _aborted_results() -> pyomo.opt.SolverResults:
    info = pyomo.opt.results.solver.SolverInformation()
    info.status = pyo.SolverStatus.aborted
    info.termination_condition = pyo.TerminationCondition.error
    results = pyomo.opt.SolverResults()
    results.solver = info
    return results
 
 
def get_solver(
    solver_options: SolverOptions, log_path: pathlib.Path | None = None
) -> SolverWrapper:
    """Construct the configured solver.
 
    Time limit and thread count are spelled differently by every solver, so
    they are set per plugin rather than passed through. Note there is no
    ``NonConvex`` setting here: CoRAL needs it because its model is quadratic,
    and this one is not -- which is also why HiGHS and CBC are usable at all.
    """
    solver = pyo.SolverFactory(solver_options.solver.value)
    time_limit = solver_options.time_limit_s
    threads = solver_options.num_threads
 
    match solver_options.solver:
        case Solver.GUROBI:
            solver.options["TimeLimit"] = time_limit
            if threads > 0:
                solver.options["Threads"] = threads
        case Solver.SCIP:
            solver.options["limits/time"] = time_limit
        case Solver.CBC:
            solver.options["seconds"] = time_limit
            if threads > 0:
                solver.options["threads"] = threads
        case Solver.HIGHS:
            solver.options["time_limit"] = float(time_limit)
            if threads > 0:
                solver.options["threads"] = threads
 
    supports_logfile = not solver_options.solver.value.startswith("appsi_")
 
    if not solver.available(exception_flag=False):
        raise RuntimeError(
            f"Solver {solver_options.solver.value!r} is not available. Install "
            f"it, or choose another via SolverOptions.solver."
        )
    return SolverWrapper(solver, log_path, supports_logfile=supports_logfile)
 
 
def solve_model(
    model: CycleModel, solver_options: SolverOptions
) -> pyomo.opt.SolverResults:
    """Write the model if asked, then solve it.
 
    The order matters: the ``.lp`` is written *before* the blocking call, so it
    exists even if the solve hangs or the process is killed -- which is exactly
    when it is wanted. ``model.name`` carries the amplicon, model type, and
    round, so peel rounds do not overwrite each other.
    """
    path: pathlib.Path | None = None
    if solver_options.write_model_files:
        path = solver_options.output_dir / "models" / model.name
        model.write_files(path)
 
    solver = get_solver(solver_options, log_path=path)
    results = solver.solve(model)
 
    condition = results.solver.termination_condition
    if condition not in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
    ):
        logger.info(
            "Solver returned %s (%s) for %s.",
            condition,
            results.solver.status,
            model.name,
        )
    return results
 
 
# --------------------------------------------------------------------------- #
# Parsing                                                                      #
# --------------------------------------------------------------------------- #
 
 
def parse_solution(
    results: pyomo.opt.SolverResults,
    model: CycleModel,
    graph: BreakpointGraph,
    *,
    metadata: ModelMetadata | None = None,
) -> CycleSolution:
    """Read the solved variables back into a ``CycleSolution``.
 
    An empty solution comes back when the solver returned nothing usable; the
    status and termination condition are preserved either way, because they are
    the caller's only signal that a partial answer is partial.
    """
    solution = CycleSolution(
        solver_status=results.solver.status,
        termination_condition=results.solver.termination_condition,
        model_metadata=metadata,
    )
    solution.mip_gap, solution.upper_bound = _read_bounds(results)
 
    if not _has_solution(results):
        logger.debug(
            "No solution to load from %s (%s).",
            model.name,
            results.solver.termination_condition,
        )
        return solution
 
    model.solutions.load_from(results)
 
    for i in model.walks:
        if pyo.value(model.sel[i]) < SELECTED:
            continue
        weight = pyo.value(model.F[i])
        walk = _read_walk(model, graph, i, weight)
        if not walk:
            logger.debug("Walk %d selected but uses no edge; skipping.", i)
            continue
 
        satisfied = _satisfied_constraints(graph, walk)
        is_cycle = not any(
            key.type is EdgeType.SOURCE for key in walk
        )
        bucket = "cycles" if is_cycle else "paths"
        getattr(solution.walks, bucket).append(walk)
        getattr(solution.walk_weights, bucket).append(weight)
        getattr(solution.satisfied_pc, bucket).append(satisfied)
        solution.satisfied_pc_set |= set(satisfied)
 
        solution.total_weights_included += sum(
            multiplicity * weight * len(graph.sequence_edges[key.idx])
            for key, multiplicity in walk.items()
            if key.type is EdgeType.SEQUENCE
        )
 
    return solution
 
 
def _has_solution(results: pyomo.opt.SolverResults) -> bool:
    """Whether the results carry a loadable incumbent.
 
    A time limit hit after an incumbent was found still yields a usable walk,
    so this tests for a solution rather than for optimality.
    """
    if not results.solution:
        return False
    return results.solver.termination_condition not in (
        pyo.TerminationCondition.infeasible,
        pyo.TerminationCondition.unbounded,
        pyo.TerminationCondition.error,
    )
 
 
def _read_bounds(
    results: pyomo.opt.SolverResults,
) -> tuple[float | None, float | None]:
    """Absolute gap and bound, when the solver reported them."""
    problem = results.problem
    lower = getattr(problem, "lower_bound", None)
    upper = getattr(problem, "upper_bound", None)
    if lower is None or upper is None:
        return None, None
    try:
        return abs(float(upper) - float(lower)), float(upper)
    except (TypeError, ValueError):
        return None, None
 
 
def _read_walk(
    model: CycleModel, graph: BreakpointGraph, i: int, weight: float
) -> OptimizationWalk:
    """Edge multiplicities for walk ``i``.
 
    Discordant multiplicities are read off ``y``, which names the chosen branch
    exactly. CoRAL divides ``f`` by the walk weight and rounds; that is a float
    quotient, and at small weights it can land near ``.5``. Dividing is only
    needed for the other edge types, where ``ConstraintFlowIsWeight`` pins the
    ratio to 1 and the division is a consistency check rather than a
    measurement.
    """
    walk: OptimizationWalk = {}
    for key in graph.edge_keys():
        if pyo.value(model.used(key, i)) < SELECTED:
            continue
        if key.type is EdgeType.DISCORDANT:
            multiplicity = _read_multiplicity(model, graph, key, i)
        else:
            multiplicity = 1
            _check_unit_flow(model, key, i, weight)
        if multiplicity > 0:
            walk[key] = multiplicity
    return walk
 
 
def _read_multiplicity(
    model: CycleModel, graph: BreakpointGraph, key: EdgeId, i: int
) -> int:
    """The ``m`` whose indicator the solver switched on."""
    chosen = [
        m
        for m in range(1, graph.multiplicity_bound(key) + 1)
        if pyo.value(model.mult_indicator(key, m, i)) >= SELECTED
    ]
    if len(chosen) == 1:
        return chosen[0]
    logger.warning(
        "Discordant edge %s in walk %d has %d active multiplicity "
        "indicators; falling back to 1.",
        key,
        i,
        len(chosen),
    )
    return 1
 
 
def _check_unit_flow(
    model: CycleModel, key: EdgeId, i: int, weight: float
) -> None:
    """Warn if a non-discordant edge does not carry exactly the walk's weight.
 
    ``ConstraintFlowIsWeight`` makes this an identity, so a violation means
    numerical trouble -- a loose solver tolerance, or a big-M large enough to
    swamp it -- and the reconstructed walk will not reproduce the copy numbers.
    """
    if weight <= 0.0:
        return
    ratio = pyo.value(model.flow(key, i)) / weight
    if abs(ratio - 1.0) > INTEGRALITY_TOLERANCE:
        logger.warning(
            "Edge %s in walk %d carries %.4f x the walk weight; expected 1.",
            key,
            i,
            ratio,
        )
 
 
def _satisfied_constraints(
    graph: BreakpointGraph, walk: OptimizationWalk
) -> list[int]:
    """Which subpath constraints this walk actually carries.
 
    Recomputed from the walk rather than read off ``P``. The subpath linkage
    only forces edges in when ``P`` is 1 and never forces ``P`` up when they
    are present, so ``P`` is a lower bound on what a walk satisfies -- and in
    the peel, constraints an earlier round already explained carry no reward,
    leaving their indicators free to come back 0 on a walk that contains them.
    """
    return [
        constraint.pc_idx
        for constraint in graph.longest_path_constraints
        if all(
            walk.get(edge_id, 0) >= count
            for edge_id, count in constraint.edge_counts.items()
        )
    ]