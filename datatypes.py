"""
Data types for Cycle Extractor.
 
Nothing here imports coral: the dependency runs coral -> cycle_extractor only.
Compatibility is by mirrored shape -- same names, same field names, same StrEnum
values -- so ``ce.EdgeType.SEQUENCE == coral.EdgeType.SEQUENCE`` holds, tuple
types round-trip, and ``coral.CycleSolution(**vars(result))`` is the whole
conversion at the boundary.
 
 
with one rule deciding concrete-vs-Protocol: a concrete class when the caller
must construct it or CE returns it; a Protocol when the caller already holds the
object. So the graph and its path constraints are Protocols -- CoRAL passes its
own objects in unconverted -- while options and the result are dataclasses.
"""
 
from __future__ import annotations
 
import os
import pathlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import (
    Any,
    Generic,
    Mapping,
    NamedTuple,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)
 
import pyomo.environ as pyo
 
T = TypeVar("T")
 
EdgeIdx = int
EdgeCount = int
 
 
# --- concrete: reachable from CycleSolution --------------------------------
 
 
class EdgeType(StrEnum):
    """Forced by EdgeId. Values mirror coral.datatypes.EdgeType."""
 
    SEQUENCE = "e"
    CONCORDANT = "c"
    DISCORDANT = "d"
    SOURCE = "s"
    SINK = "t"
    SYNTHETIC_SOURCE = "ns"
    SYNTHETIC_SINK = "nt"
    TERMINAL = "$"
 
    @property
    def is_synthetic(self) -> bool:
        return self in (EdgeType.SYNTHETIC_SOURCE, EdgeType.SYNTHETIC_SINK)
 
 
class EdgeId(NamedTuple):
    """Forced by OptimizationWalk. NamedTuple so ``EdgeId(*coral_edge_id)``
    round-trips and tuple equality holds across the two packages."""
 
    type: EdgeType
    idx: EdgeIdx
 
 
OptimizationWalk = dict[EdgeId, EdgeCount]
"""One extracted structure as an edge-multiplicity map. Same alias as CoRAL;
carries no orientation, so ordering is recovered downstream at reconstruction."""
 
 
@dataclass
class WalkData(Generic[T]):
    """Forced by CycleSolution."""
 
    cycles: list[T]
    paths: list[T]
 
 
# --- protocols: objects the caller already holds ---------------------------
 
 
@runtime_checkable
class FinalizedPathConstraintLike(Protocol):
    """A filtered subpath constraint, as it arrives on the graph.
 
    ``edge_counts`` is an unordered multiset -- the traversal order was dropped
    at finalization. Recover it through the back-pointer:
 
        ordered = graph.path_constraints[fpc.pc_idx].path
 
    CoRAL guarantees the index is valid: ``longest_path_dict`` sets
    ``pc_idx=path_idx`` while enumerating ``path_constraints``, and CoRAL itself
    dereferences it this way in cycle_decomposition, breakpoint_utilities and
    cycle_output. The ordered form is what a contiguity test needs, so any
    block-lift work reads ``path_constraints``, not this multiset.
    """
 
    edge_counts: Mapping[EdgeId, int]
    pc_idx: int
    support: int
 
 
@runtime_checkable
class BreakpointGraphLike(Protocol):
    """Structural contract for the input graph.
 
    Attribute names are verbatim from coral.breakpoint.BreakpointGraph, so a
    CoRAL graph satisfies this as-is: no conversion, no adapter. Edge objects are
    left as Any -- CE reads only ``.cn`` off them, and pinning their classes
    would force a shared type where duck typing suffices.
    """
 
    amplicon_intervals: Sequence[Any]
    sequence_edges: Sequence[Any]
    concordant_edges: Sequence[Any]
    discordant_edges: Sequence[Any]
    source_edges: Sequence[Any]
    node_adjacencies: Mapping[Any, Any]
    endnode_adjacencies: Mapping[Any, list[int]]
    path_constraints: Sequence[Any]
    """Ordered form, indexed by FinalizedPathConstraintLike.pc_idx. Each element
    exposes ``.path``, an alternating list of nodes and edge ids."""
    longest_path_constraints: Sequence[FinalizedPathConstraintLike]
    """Maximality-filtered subset; what the model's constraint rows consume."""
    max_cn: float
 
 
class STStrategy(StrEnum):
    """
    Where terminals attach, for s-t (path) extraction.
    """
    SOURCE_NODES = "source_nodes" # For AA graph
    INTERVAL_ENDS = "interval_ends" # For CoRAL graph
 
 
class SortBy(StrEnum):
    COPY_NUMBER = "CopyNumber"
    LWCN = "LWCN"
 
 
class Solver(StrEnum):
    """
    CoRAL's members + MILP-only solvers.
    """
    GUROBI = "gurobi_direct"
    HIGHS = "appsi_highs"
    CBC = "cbc"
    SCIP = "scip"


class CycleDecompOptions(str, enum.Enum):
    MIN_CYCLES = "min_cycles"
    MAX_WEIGHT = "max_weight"

 
@dataclass(frozen = True)
class CEOptions:
    enforce_connectivity: bool = False
    alpha: float = 0.01
    s_t_strategy: STStrategy = STStrategy.SOURCE_NODES
    sort_by: SortBy = SortBy.COPY_NUMBER
 
 
@dataclass
class SolverOptions:
    """
    Mirror of coral.datatypes.SolverOptions.
    """
    output_dir: pathlib.Path = field(default_factory = pathlib.Path.cwd)
    output_prefix: str = ""
    num_threads: int = field(default_factory = lambda: os.cpu_count() or 1)
    time_limit_s: int = 7200
    model_prefix: str = "pyomo"
    solver: Solver = Solver.HIGHS # default to HIGHS


class ModelMetadata(NamedTuple):
    """Mirrors coral.datatypes.ModelMetadata. `k` and `alpha` stay for shape
    compatibility and are None in CE, which extracts one structure per solve."""
    model_type: CycleDecompOptions
    k: int | None = None
    path_constraint_weight: float | None = None
    total_weights: float | None = None
    resolution: float | None = None
    num_path_constraints: int = 0
    mip_gap: float | None = None

    def to_output_str(self) -> str:
        return text_utils.MODEL_METADATA_TEMPLATE.format(
            model_type = self.model_type.value,
            k = self.k,
            alpha = self.alpha,
            total_weights = self.path_constraint_weight,
            resolution = self.resolution,
            num_path_constraints = self.num_path_constraints,
            mip_gap = self.mip_gap,
        )


@dataclass
class CycleSolution:
    """
    Container for storing Pyomo solution state.
    Mirror of coral.datatypes.CycleSolution.
    """
    solver_status: pyo.SolverStatus
    termination_condition: pyo.TerminationCondition
    total_weights_included: float = 0.0
    walks: WalkData[OptimizationWalk] = field(default_factory = lambda: WalkData([], []))
    walk_weights: WalkData[float] = field(default_factory = lambda: WalkData([], []))
    satisfied_pc: WalkData[list[int]] = field(default_factory = lambda: WalkData([], []))
    satisfied_pc_set: set[int] = field(default_factory = set)
    mip_gap: float | None = None
    upper_bound: float | None = None

    model_metadata: ModelMetadata | None = None

    @property
    def relative_mip_gap(self) -> float | None:
        if self.mip_gap is None or self.upper_bound is None:
            return None
        return (self.mip_gap / self.upper_bound) * 100.0

    @property
    def num_pc_satisfied(self) -> int:
        return len(self.satisfied_pc_set)

    @property
    def num_cycles(self) -> int:
        return len(self.walks.cycles)
 
    @property
    def num_paths(self) -> int:
        return len(self.walks.paths)
 
    @property
    def num_walks(self) -> int:
        return self.num_cycles + self.num_paths

