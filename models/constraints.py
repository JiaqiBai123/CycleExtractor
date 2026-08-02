from __future__ import annotations
 
import logging
from typing import TYPE_CHECKING
 
import pyomo.environ as pyo
 
from cycle_extractor.datatypes import (
    BreakpointGraph,
    CycleExtractionOptions,
    EdgeId,
    EdgeToCN,
    EdgeType,
    ModelType,
    Node,
)
 
if TYPE_CHECKING:
    from cycle_extractor.models.concrete import CycleModel
 
logger = logging.getLogger(__name__)
 
 
def add_all_constraints(
    model: CycleModel,
    graph: BreakpointGraph,
    residual_cn: EdgeToCN,
    *,
    k: int,
    options: CycleExtractionOptions,
    model_type: ModelType,
) -> None:
    walks = range(k)
 
    add_walk_exsitence_constraints(model, graph, walks)
    add_capacity_constraints(model, graph, residual_cn, walks)
    add_unit_flow_constraints(model, graph, walks)
    add_discordant_multiplicity_constraints(model, graph, walks)
    add_balance_constraints(model, graph, walks)
    add_source_constraint(model, graph, walks)
    add_subpath_constraints(model, graph, walks, model_type = model_type)
 
    if model_type == ModelType.MIN_CYCLES:
        add_weight_floor_constraint(model, graph, options, walks)
 
    if options.enforce_connectivity:
        add_connectivity(model, graph, walks)
    else:
        logger.debug(
            "Connectivity enforcement disabled; walks may decompose into "
            "disconnected components."
        )
 
 
def add_walk_exsitence_constraints(
    model: CycleModel, graph: BreakpointGraph, walks: range
) -> None:
    model.HasEdgeSelected = pyo.ConstraintList()
    for key in graph.edge_keys():
        for i in walks:
            model.HasEdgeSelected.add(
                model.used(key, i) <= model.e[i]
            )

    model.HasWeight = pyo.Constraint(
        model.walks, rule = lambda m, i: m.F[i] <= graph.max_cn * m.e[i]
    )
 
 
def add_capacity_constraints(
    model: CycleModel,
    graph: BreakpointGraph,
    residual_cn: EdgeToCN,
    walks: range,
) -> None:
    """
    Walks cannot send more copy number through an edge than it has left.
    """
    # TBD - see "1. capacity" in CE paper, also need to consider multiple cycles/walks for MIN_CYCLES
    #
    #

 
def add_weight_floor_constraint(
    model: CycleModel,
    graph: BreakpointGraph,
    options: CycleExtractionOptions,
    walks: range,
) -> None:
    """
    MIN_CYCLES: the solution must explain a fraction of the graph's CN.
    """
    # TBD - this is not implemented in CE, please check CoRAL's implementation
    #
    #
 
 
def add_unit_flow_constraints(
    model: CycleModel, graph: BreakpointGraph, walks: range
) -> None:
    # TBD - see "3. copy number F with multiplicity one" in CE paper
    #
    #
 
 
def add_discordant_multiplicity_constraints(
    model: CycleModel, graph: BreakpointGraph, walks: range
) -> None:
    """
    On a discordant edge, f[u, v, i] == m * F[i] for some integer m.
    """
    # TBD - see "4. discordant edge multiplicity constraint" in CE paper
    #
    #
 
 
def add_balance_constraints(
    model: CycleModel, graph: BreakpointGraph, walks: range
) -> None:
    """
    At every node, sequence-edge flow equals breakpoint and terminal flow.
    """
    # TBD - see "2. balance" in CE paper, also need to consider MIN_CYCLES (for each i)
    #
    #
 
 
def add_source_constraint(
    model: CycleModel, graph: BreakpointGraph, walks: range
) -> None:
    # TBD - two constraints
    # 1. sum(x_sui) = sum(x_vti) (# source edges = # sink edges)
    # 2. sum(x_sui) <= e[i] (if a walk exists, it is either a cycle or has only 1 source edge)
 
 
def add_subpath_constraints(
    model: CycleModel,
    graph: BreakpointGraph,
    walks: range,
    *,
    model_type: ModelType,
) -> None:
    constraints = graph.longest_path_constraints
    if not constraints:
        return
 
    # TBD - please check CoRAL implementation
    # CoRAL has a bug that subpath constraints were not considered
    # you can require >= 0.5 (50%) subpath constraint satisfied by default in MIN_CYCLES.
    # MAX_WEIGHT does not need R[p]. they can be doppped automatically by pyomo.

 
 
def add_connectivity_constraint(
    model: CycleModel, graph: BreakpointGraph, walks: range
) -> None:
    if not graph.node_adjacencies:
        logger.debug("Empty node set; no connectivity rows to add.")
        return
 
    # TBD - may use https://link.springer.com/chapter/10.1007/3-540-36626-1_5
    # can also use what is described in CE
    # you can define any helper functions needed separately


