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
        add_connectivity_constraint(model, graph, walks)
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
    model.EdgeCapacity = pyo.ConstraintList()
    edge_keys = list(graph.edge_keys())

    def capacity(key: EdgeId) -> float | None:
        if key.type == EdgeType.SEQUENCE:
            return residual_cn.sequence[key.idx]
        if key.type == EdgeType.CONCORDANT:
            return residual_cn.concordant[key.idx]
        if key.type == EdgeType.DISCORDANT:
            return residual_cn.discordant[key.idx]
        if key.type in (EdgeType.SOURCE, EdgeType.SINK):
            return residual_cn.source[key.idx]
        return None

    for key in edge_keys:
        edge_capacity = capacity(key)
        if edge_capacity is None:
            continue
        for i in walks:
            model.EdgeCapacity.add(
                model.flow(key, i) <= edge_capacity * model.used(key, i)
            )

    for key in graph.structural_keys():
        edge_capacity = capacity(key)
        if edge_capacity is None:
            continue
        model.EdgeCapacity.add(
            sum(model.flow(key, i) for i in walks) <= edge_capacity
        )

    for srci in range(len(graph.source_edges)):
        source_key = EdgeId(EdgeType.SOURCE, srci)
        sink_key = EdgeId(EdgeType.SINK, srci)

        model.EdgeCapacity.add(
            sum(
                model.flow(source_key, i) + model.flow(sink_key, i)
                for i in walks
            )
            <= residual_cn.source[srci]
        )   
    
 
 
def add_weight_floor_constraint(
    model: CycleModel,
    graph: BreakpointGraph,
    options: CycleExtractionOptions,
    walks: range,
) -> None:
    """
    MIN_CYCLES: the solution must explain a fraction of the graph's CN.
    """
    explained_weight = sum(
        model.flow(key, i) * len(graph.sequence_edges[key.idx])
        for key in graph.sequence_keys()
        for i in walks
    )

    fraction = 0.9
    model.WeightFloor = pyo.Constraint(
        expr = explained_weight >= fraction * graph.total_weights
    )
 
 
def add_unit_flow_constraints(
    model: CycleModel, graph: BreakpointGraph, walks: range
) -> None:
    model.UnitFlow = pyo.ConstraintList()
    big_m = graph.max_cn

    for key in graph.edge_keys():
        for i in walks:
            model.UnitFlow.add(model.F[i] <= model.flow(key, i) + big_m * (1 - model.used(key, i)))

 
def add_discordant_multiplicity_constraints(
    model: CycleModel, graph: BreakpointGraph, walks: range
) -> None:
    """
    On a discordant edge, f[u, v, i] == m * F[i] for some integer m.
    """
    model.DiscordantMultiplicity = pyo.ConstraintList()

    C_max = graph.max_cn

    for key in graph.structural_keys():
        if key.type != EdgeType.DISCORDANT:
            continue

        multiplicities = range(
            1, graph.multiplicity_bound(key) + 1
        )

        for i in walks:
            model.DiscordantMultiplicity.add(
                sum(
                    model.mult_indicator(key, m, i)
                    for m in multiplicities
                )
                == model.used(key, i)
            )

            for m in multiplicities:
                model.DiscordantMultiplicity.add(
                    model.mult_flow(key, m, i) <= m * model.F[i]
                )

                model.DiscordantMultiplicity.add(
                    model.mult_flow(key, m, i) <= C_max * model.mult_indicator(key, m, i)
                )

                model.DiscordantMultiplicity.add(
                    model.mult_flow(key, m, i) >= m * model.F[i] - C_max
                    * (1 - model.mult_indicator(key, m, i))
                )

                model.DiscordantMultiplicity.add(
                    model.mult_flow(key, m, i) >= 0
                )

            model.DiscordantMultiplicity.add(
                model.flow(key, i)
                == sum(
                    model.mult_flow(key, m, i)
                    for m in multiplicities
                )
            )
 
 
def add_balance_constraints(
    model: CycleModel, graph: BreakpointGraph, walks: range
) -> None:
    """
    At every node, sequence-edge flow equals concordant and discordant flow.
    """
    model.NodeBalance = pyo.ConstraintList()

    for node, adjacency in graph.node_adjacencies.items():

        sequence_keys = [
            EdgeId(EdgeType.SEQUENCE, seqi)
            for seqi in adjacency.sequence
        ]

        breakpoint_keys = [
            EdgeId(EdgeType.CONCORDANT, ci)
            for ci in adjacency.concordant
        ]

        breakpoint_keys.extend(
            EdgeId(EdgeType.DISCORDANT, di)
            for di in adjacency.discordant
        )

        for i in walks:
            model.NodeBalance.add(
                sum(
                    model.flow(key, i)
                    for key in breakpoint_keys
                )
                ==
                sum(
                    model.flow(key, i)
                    for key in sequence_keys
                )
            )
 
 
def add_source_constraint(
    model: CycleModel, graph: BreakpointGraph, walks: range
) -> None:

    model.SourceConstraint = pyo.ConstraintList()

    for i in walks:
        model.SourceConstraint.add(
            sum(
                model.used(EdgeId(EdgeType.SOURCE, srci), i)
                for srci in range(len(graph.source_edges))
            )
            ==
            sum(
                model.used(EdgeId(EdgeType.SINK, srci), i)
                for srci in range(len(graph.source_edges))
            )
        )

        model.SourceConstraint.add(
            sum(
                model.used(EdgeId(EdgeType.SOURCE, srci), i)
                for srci in range(len(graph.source_edges))
            )
            <= model.e[i]
        )
 
 
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

    model.SubpathConstraints = pyo.ConstraintList()

    for pi, path_constraint in enumerate(constraints):
        for key, edge_count in path_constraint.edge_counts.items():
            for i in walks:

                if key.type == EdgeType.DISCORDANT:
                    multiplicities = range(
                        1,
                        graph.multiplicity_bound(key) + 1
                    )

                    model.SubpathConstraints.add(
                        sum(
                            m * model.mult_indicator(key, m, i)
                            for m in multiplicities
                        )
                        >= edge_count * model.r[pi, i]
                    )

                else:
                    model.SubpathConstraints.add( 
                        model.used(key, i) >= edge_count * model.r[pi, i] 
                    )


    if model_type == ModelType.MIN_CYCLES:

        for pi in range(len(constraints)):

            for i in walks:
                model.SubpathConstraints.add(
                    model.R[pi] >= model.r[pi, i]
                )

            model.SubpathConstraints.add(
                sum(
                    model.r[pi, i]
                    for i in walks
                )
                >= model.R[pi]
            )

        model.SubpathConstraints.add(
            sum(
                model.R[pi]
                for pi in range(len(constraints))
            )
            >= 0.5 * len(constraints)
        )
 
 
def add_connectivity_constraint(
    model: CycleModel, graph: BreakpointGraph, walks: range
) -> None:
    if not graph.node_adjacencies:
        logger.debug("Empty node set; no connectivity rows to add.")
        return

    model.Connectivity = pyo.ConstraintList()

    s = graph.start_node
    num_nodes = graph.num_nodes
    structural_keys = list(graph.structural_keys())

    def endpoints(key: EdgeId) -> tuple[Node, Node]:
        if key.type == EdgeType.SEQUENCE:
            edge = graph.sequence_edges[key.idx]
            return edge.start_node, edge.end_node

        if key.type == EdgeType.CONCORDANT:
            edge = graph.concordant_edges[key.idx]
            return edge.node1, edge.node2

        if key.type == EdgeType.DISCORDANT:
            edge = graph.discordant_edges[key.idx]
            return edge.node1, edge.node2

    edge_endpoints = {
        key: endpoints(key)
        for key in structural_keys
    }

    for i in walks:

        model.Connectivity.add(
            model.d_order(s, i) == 1
        )
  
        incoming_s = []

        for key in structural_keys:
            u, v = edge_endpoints[key]

            if s == u:
                incoming_s.append(model.d_rev(key, i))

            elif s == v:
                incoming_s.append(model.d_fwd(key, i))

        model.Connectivity.add(
            sum(incoming_s) == 0
        )

        incident_s = [
            key
            for key in structural_keys
            if s in edge_endpoints[key]
        ]

        selected_from_s = sum(
            model.used(key, i)
            for key in incident_s
        )

        for key in incident_s:
            u, v = edge_endpoints[key]

            if s == u:
                outgoing = model.d_fwd(key, i)
            else:
                outgoing = model.d_rev(key, i)

            model.Connectivity.add(
                outgoing <= selected_from_s
            )

       
        for node in graph.node_adjacencies:

            incident_keys = [
                key
                for key in structural_keys
                if node in edge_endpoints[key]
            ]

            model.Connectivity.add(
                model.d_order(node, i)
                <= num_nodes
                * sum(
                    model.used(key, i)
                    for key in incident_keys
                )
            )

       
        for key in structural_keys:
            model.Connectivity.add(
                model.d_fwd(key, i) + model.d_rev(key, i) <= model.used(key, i)
            )

       
        for node in graph.node_adjacencies:

            if node == s:
                continue

            incoming = []
            incident_keys = []

            for key in structural_keys:
                u, v = edge_endpoints[key]

                if node == u:
                    incident_keys.append(key)
                    incoming.append(model.d_rev(key, i))

                elif node == v:
                    incident_keys.append(key)
                    incoming.append(model.d_fwd(key, i))

            model.Connectivity.add(
                sum(incoming) 
                >= (1 / num_nodes)
                * sum(
                    model.used(key, i)
                    for key in incident_keys
                )
            )

        
        for key in structural_keys:
            u, v = edge_endpoints[key]

            model.Connectivity.add(
                model.d_order(v, i)- model.d_order(u, i)
                >= 1 - num_nodes * (1 - model.d_fwd(key, i))
            )

            model.Connectivity.add(
                model.d_order(u, i) - model.d_order(v, i)
                >= 1 - num_nodes * (1 - model.d_rev(key, i))
            )


