"""
Define model variables, objectives, and add constraints.
 
Variable families, indexed by walk i in [0, k-1]:
 
    e[i]	walk i is selected (for CoRAL, this is z[i]), binary
    F[i]        copy number carried by walk i (for CoRAL, this is w[i]), continuous
    x[u, v, i]  times column (u, v) is traversed by walk i, binary (for CoRAL this is integer)
    f[u, v, i]  times column (u, v) is traversed by walk i (= CoRAL's x[u, v, i] * w[i])
    y[u, v, m, i]  times column (u, v) is traversed by walk i, CE only, binary
    z[u, v, m, i]  times column (u, v) is traversed by walk i, CE only, continuous
    r[p, i]     subpath constraint p is carried by walk i, binary
    R[p]	some walk carries constraint p, binary
    d[v, i]     depth of node n in walk i's spanning tree, integer
    t1/t2[u, v,i]  column (u, v) is a tree edge, forward / reverse, (= CoRAL's y1, y2)

MAX_WEIGHT requires k = 1. It extracts one walk, and the caller peels.
MIN_CYCLES uses full k walks. Both go through the same builder.
"""
 
from __future__ import annotations
 
import logging
import pathlib
 
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
from cycle_extractor.models import constraints as constraint_rows
 
logger = logging.getLogger(__name__)
 
 
class CycleModel(pyo.ConcreteModel):
    """
    Variables for one extraction model.  
    """
    def __init__(
        self,
        name: str,
        graph: BreakpointGraph,
        k: int,
        options: CycleExtractionOptions,
    ) -> None:
        super().__init__(name = name)
        self._index = index

        edge_keys = graph.edge_keys(include_terminals = options.allow_paths)
        mult_keys = [
            (key.type, key.idx, m)
            for key in edge_keys
            if key.type is EdgeType.DISCORDANT
            for m in range(1, graph.multiplicity_bound(key) + 1)
        ]
 
        def by_type(edge_type: EdgeType) -> list[EdgeId]:
            return [key for key in edge_keys if key.type is edge_type]
 
        self.walks = pyo.RangeSet(0, k - 1)
        self.edges = pyo.Set(initialize = edge_keys, dimen = 2)
        self.seq_edges = pyo.Set(initialize = by_type(EdgeType.SEQUENCE), dimen = 2)
        self.conc_edges = pyo.Set(
            initialize = by_type(EdgeType.CONCORDANT), dimen = 2
        )
        self.disc_edges = pyo.Set(
            initialize = by_type(EdgeType.DISCORDANT), dimen = 2
        )
        self.non_source_edges = pyo.Set(
            initialize = graph.structural_keys(), dimen = 2
        )
        self.pc_idx = pyo.RangeSet(0, len(graph.longest_path_constraints) - 1)
        self.mult_keys = pyo.Set(initialize = mult_keys, dimen = 3)
 
        self.e = pyo.Var(self.walks, domain = pyo.Binary)
        self.F = pyo.Var(
            self.walks, domain = pyo.NonNegativeReals, bounds = (0.0, graph.max_cn)
        )
        self.x = pyo.Var(self.edges, self.walks, domain = pyo.Binary)
        self.f = pyo.Var(self.edges, self.walks, domain = pyo.NonNegativeReals)
        self.r = pyo.Var(self.pc_idx, self.walks, domain = pyo.Binary)
        self.y = pyo.Var(self.mult_keys, self.walks, domain = pyo.Binary)
        self.z = pyo.Var(
            self.mult_keys, self.walks, domain = pyo.NonNegativeReals
        )

        if options.enforce_connectivity:
            self.node_keys = pyo.Set(
                initialize = list(graph.node_adjacencies), dimen = 3
            )
            self.t1 = pyo.Var(
                self.non_source_edges, self.walks, domain = pyo.Binary
            )
            self.t2 = pyo.Var(
                self.non_source_edges, self.walks, domain = pyo.Binary
            )
            self.d = pyo.Var(
                self.node_keys,
                self.walks,
                domain = pyo.NonNegativeIntegers,
                bounds = (0, graph.num_nodes),
            )
 
    def edge_used(self, key: EdgeId, i: int) -> pyo.Var:
        return self.x[key.type, key.idx, i]
 
    def flow(self, key: EdgeId, i: int) -> pyo.Var:
        return self.f[key.type, key.idx, i]
 
    def mult_indicator(self, key: EdgeId, m: int, i: int) -> pyo.Var:
        return self.y[key.type, key.idx, m, i]
 
    def mult_flow(self, key: EdgeId, m: int, i: int) -> pyo.Var:
        return self.z[key.type, key.idx, m, i]
 
    def remaining_weights(
        self, graph: BreakpointGraph, residual_cn: EdgeToCN
    ) -> float:
        """
        Length-weighted copy number still unexplained.
        """
        return sum(
            residual_cn.sequence[key.idx]
            * len(graph.sequence_edges[key.idx])
            for key in graph.sequence_keys()
        )
 
    def d_fwd(self, key: EdgeId, i: int) -> pyo.Var:
        return self.t1[key.type, key.idx, i]
 
    def d_rev(self, key: EdgeId, i: int) -> pyo.Var:
        return self.t2[key.type, key.idx, i]
 
    def d_order(self, node: Node, i: int) -> pyo.Var:
        return self.d[node.chr, node.pos, node.strand, i]
 
    def explained(
        self, graph: BreakpointGraph, walks: range
    ) -> pyo.Expression:
        """
        Length-weighted copy number the given walks account for.
        """
        sequence_keys = graph.sequence_keys()
        return sum(
            self.f(key, i) * len(graph.sequence_edges[key.idx])
            for i in walks
            for key in sequence_keys
        )
 
    def write_files(self, path: str | pathlib.Path) -> None:
        """
        Dump the model as <path>.lp and <path>_ampl.nl.
        """
        path = pathlib.Path(path)
        path.parent.mkdir(parents = True, exist_ok = True)
        self.write(
            f"{path}.lp", io_options = {"symbolic_solver_labels": True}
        )
        self.write(f"{path}_ampl.nl", format = "nl")
        logger.debug("Wrote model to %s.lp and %s_ampl.nl.", path, path)
 
 
def build_model(
    graph: BreakpointGraph,
    *,
    k: int,
    model_type: ModelType,
    options: CycleExtractionOptions,
    residual_cn: EdgeToCN | None = None,
    is_pc_unsatisfied: list[bool] | None = None,
    name: str = "cycle_extraction",
) -> CycleModel:
    """
    Generate the MILP for one call to the solver.
    Declares the variables, installs the objective "model_type" selects, and
    attaches every constraint row. 
    Returns a pyomo ConcreteModel object 
    """
    if model_type == ModelType.MAX_WEIGHT and k != 1:
        raise ValueError(
            f"The max-weight model extracts one walk per round; got k = {k}."
        )
    if k < 1:
        raise ValueError(f"k must be at least 1; got {k}.")

    residual_cn = residual_cn or EdgeToCN.from_graph(graph)
 
    model = CycleModel(
        name = f"{name}_k={k}",
        graph = graph,
        k = k,
        options = options,
    )

    match model_type:
        case ModelType.MIN_CYCLES:
            model.Objective = ce_minimize_cycles_objective(
                model, graph, options, k
            )
        case ModelType.MAX_WEIGHT:
            model.Objective = ce_max_weight_objective(
                model, graph, options, residual_cn, is_pc_unsatisfied
            )
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")

    constraint_rows.add_all_constraints(
        model,
        graph,
        residual_cn = residual_cn,   
        k = k,
        options = options,
        model_type = model_type,
    )
 
    _log_model_shape(model, graph, options, model_type, k)
    return model
 

def _log_model_shape(
    model: CycleModel,
    graph: BreakpointGraph,
    options: CycleExtractionOptions,
    model_type: ModelType,
    k: int,
) -> None:
    """
    Record the model at debug level. 
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
 
    num_terminals = len(model.edges) - len(model.structural_edges)
    logger.debug(
        "Amplicon %d: %s model, k = %d.",
        graph.amplicon_idx + 1,
        model_type.value,
        k,
    )
    logger.debug(
        "  graph: %d nodes, %d edges (%d seq, %d conc, %d disc, %d src), "
        "max CN %.2f, total length-weighted CN %.2f.",
        graph.num_nodes,
        graph.num_edges,
        graph.num_seq_edges,
        graph.num_conc_edges,
        graph.num_disc_edges,
        graph.num_src_edges,
        graph.max_cn,
        graph.total_weights,
    )
    logger.debug(
        "  model: %d edge columns (%d structural + %d terminal), "
        "%d multiplicity keys, %d subpath constraints.",
        len(model.edges),
        len(model.structural_edges),
        num_terminals,
        len(model.mult_keys),
        len(graph.longest_path_constraints),
    )
    logger.debug(
        "  variables: %d total across %d families (%d binary, %d continuous, "
        "%d integer).",
        *variable_census(model),
    )
    logger.debug(
        "  connectivity %s, terminals %s.",
        "on" if options.enforce_connectivity else "off",
        "on" if options.allow_paths else "off",
    )
 
 
def variable_census(model: CycleModel) -> tuple[int, int, int, int, int]:
    """
    total, families, binary, continuous, integer) over the model.
    """
    total = families = binary = continuous = integer = 0
    for variable in model.component_objects(pyo.Var):
        families += 1
        for element in variable.values():
            total += 1
            if element.domain is pyo.Binary:
                binary += 1
            elif element.is_integer():
                integer += 1
            else:
                continuous += 1
    return total, families, binary, continuous, integer


# Objectives
def ce_minimize_cycles_objective(
    model: CycleModel,
    graph: BreakpointGraph,
    options: CycleExtractionOptions,
    k: int,
    *,
) -> pyo.Objective:
    """
    Fewest walks, tie-broken toward explaining more copy number.
    """
    total_weights = graph.total_weights
    if total_weights <= 0.0:
        raise ValueError(
            "total_weights must be positive; the graph has no sequence edge "
            "copy number to explain."
        )

    num_walks = sum(model.e[i] for i in range(k))
    cn_ratio = model.explained(range(k)) / total_weights
    return pyo.Objective(
        sense = pyo.minimize, expr = 1.0 + num_walks - cn_ratio
    )


def ce_max_weight_objective(
    model: CycleModel,
    graph: BreakpointGraph,
    options: CycleExtractionOptions,
    residual_cn: EdgeToCN,
    *,
    is_pc_unsatisfied: list[bool] | None,
) -> pyo.Objective:
    """
    Heaviest single walk, with a bonus for explaining new subpaths.
    """
    cn_explained = model.explained(range(1))
    constraints = graph.longest_path_constraints

    if is_pc_unsatisfied is None:
        is_pc_unsatisfied = [True] * len(constraints)
    elif len(is_pc_unsatisfied) != len(constraints):
        raise ValueError(
            f"is_pc_unsatisfied has {len(is_pc_unsatisfied)} entries for "
            f"{len(constraints)} subpath constraint(s)."
        )
 
    num_pc_unsatisfied = sum(is_pc_unsatisfied)
    if num_pc_unsatisfied == 0:
        return pyo.Objective(sense = pyo.maximize, expr = cn_explained)
 
    bonus_per_pc = max(
        options.path_constraint_weight * model.remaining_weights(residual_cn) / num_pc_unsatisfied,
        1.0,
    )
    obj_pc = sum(
        model.r[position, 0] * bonus_per_pc
        for position, unsatisfied in enumerate(is_pc_unsatisfied)
        if unsatisfied
    )
    return pyo.Objective(sense = pyo.maximize, expr = cn_explained + obj_pc)
 
 
