"""Building a breakpoint graph edge by edge.
 
``datatypes.BreakpointGraph`` is the finished container -- a plain dataclass the
model reads and never writes, and the type CoRAL's own graph subclasses. This
module is the other half: assembling one while keeping the adjacency index in
step, then deriving the pieces that are functions of the finished edge set.
 
Nothing here infers copy numbers. CE takes ``cn`` as given, from the file or
from the caller; fitting copy numbers to coverage is upstream work that needs
alignments CE never sees.
"""
 
from __future__ import annotations
 
import logging
from collections import defaultdict
 
from cycle_extractor.datatypes import (
    AdjacencyMatrix,
    AmpliconInterval,
    BreakpointGraph,
    ConcordantEdge,
    DiscordantEdge,
    EdgeId,
    FinalizedPathConstraint,
    Node,
    PathConstraint,
    SequenceEdge,
    SourceEdge,
    Strand,
)
 
logger = logging.getLogger(__name__)
 
 
class GraphBuilder:
    """Accumulates edges, then hands back a finished graph.
 
    Separate from ``BreakpointGraph`` because the two want opposite things: the
    container is read-only and its ``Mapping`` fields say so, while building
    needs a ``defaultdict`` and repeated mutation. Keeping the mutation here
    means the graph the model receives has no half-built states.
 
    Edge order is preserved exactly as added. That is not cosmetic -- path
    constraints reference edges by index, and ``node_adjacencies`` insertion
    order fixes the root-candidate order in the connectivity rows, so sorting
    after the fact would silently change both.
    """
 
    def __init__(self, amplicon_idx: int = 0) -> None:
        self.amplicon_idx = amplicon_idx
        self.sequence_edges: list[SequenceEdge] = []
        self.concordant_edges: list[ConcordantEdge] = []
        self.discordant_edges: list[DiscordantEdge] = []
        self.source_edges: list[SourceEdge] = []
        self.path_constraints: list[PathConstraint] = []
        self.node_adjacencies: defaultdict[Node, AdjacencyMatrix] = defaultdict(
            AdjacencyMatrix
        )
 
    # -- edges ------------------------------------------------------------- #
 
    def add_sequence_edge(
        self, chr: str, start: int, end: int, cn: float = 0.0
    ) -> None:
        """Add a genomic segment. Its endpoints are ``start-`` and ``end+``."""
        if start > end:
            raise ValueError(
                f"Sequence edge {chr}:{start}-{end} is inverted."
            )
        edge = SequenceEdge(chr=chr, start=start, end=end, cn=cn)
        index = len(self.sequence_edges)
        self.sequence_edges.append(edge)
        self.node_adjacencies[edge.node1].sequence.append(index)
        self.node_adjacencies[edge.node2].sequence.append(index)
 
    def add_concordant_edge(
        self, node1: Node, node2: Node, cn: float = 0.0
    ) -> None:
        """Join two adjacent segments across their reference-order boundary."""
        edge = ConcordantEdge(node1=node1, node2=node2, cn=cn)
        index = len(self.concordant_edges)
        self.concordant_edges.append(edge)
        self.node_adjacencies[node1].concordant.append(index)
        self.node_adjacencies[node2].concordant.append(index)
 
    def add_discordant_edge(
        self, node1: Node, node2: Node, cn: float = 0.0
    ) -> None:
        """Add a structural-variant junction.
 
        A self-loop registers once, not twice: appending the index to both
        endpoints of ``node1 == node2`` would make the node's degree count it
        twice and break the conservation row.
        """
        edge = DiscordantEdge(node1=node1, node2=node2, cn=cn)
        index = len(self.discordant_edges)
        self.discordant_edges.append(edge)
        self.node_adjacencies[node1].discordant.append(index)
        if node1 != node2:
            self.node_adjacencies[node2].discordant.append(index)
 
    def add_source_edge(self, node: Node, cn: float = 0.0) -> None:
        """Attach a terminal at ``node``, letting a walk begin or end there."""
        index = len(self.source_edges)
        self.source_edges.append(SourceEdge(node=node, cn=cn))
        self.node_adjacencies[node].source.append(index)
 
    def add_path_constraint(self, path: list[Node | EdgeId], support: int) -> None:
        """Record a long-read subwalk the decomposition should reproduce."""
        self.path_constraints.append(
            PathConstraint(
                path=path, support=support, amplicon_id=self.amplicon_idx
            )
        )
 
    # -- finishing --------------------------------------------------------- #
 
    def build(self) -> BreakpointGraph:
        """Derive the remaining fields and return the finished graph."""
        intervals, end_nodes = self._infer_intervals()
        endnode_adjacencies = {
            node: list(self.node_adjacencies[node].sequence)
            for node in end_nodes
        }
        max_cn = max(
            (edge.cn for edge in self.sequence_edges), default=0.0
        )
 
        graph = BreakpointGraph(
            amplicon_intervals=intervals,
            sequence_edges=self.sequence_edges,
            concordant_edges=self.concordant_edges,
            discordant_edges=self.discordant_edges,
            source_edges=self.source_edges,
            node_adjacencies=dict(self.node_adjacencies),
            endnode_adjacencies=endnode_adjacencies,
            path_constraints=self.path_constraints,
            longest_path_constraints=longest_path_constraints(
                self.path_constraints
            ),
            # The model bounds every flow by max_cn, so an edge sitting exactly
            # at the maximum would be pinned to its bound; CoRAL adds the same
            # slack for the same reason.
            max_cn=max_cn + 1.0,
            amplicon_idx=self.amplicon_idx,
        )
        logger.info(
            "Built amplicon %d: %d nodes, %d edges (%d seq, %d conc, %d disc, "
            "%d src), %d intervals, %d/%d subpath constraints kept.",
            self.amplicon_idx + 1,
            graph.num_nodes,
            graph.num_edges,
            graph.num_seq_edges,
            graph.num_conc_edges,
            graph.num_disc_edges,
            graph.num_src_edges,
            len(intervals),
            len(graph.longest_path_constraints),
            len(self.path_constraints),
        )
        return graph
 
    def _infer_intervals(
        self,
    ) -> tuple[list[AmpliconInterval], list[Node]]:
        """Group sequence edges into contiguous runs, and take their ends.
 
        A run breaks where the next edge is on another chromosome or does not
        start one base after the previous one ends. The two outer endpoints of
        each run are the amplicon's boundary nodes -- the ones that get
        terminal columns, so a walk can begin or end at the edge of the
        amplified region rather than only inside it.
 
        Relies on the sequence edges being in reference order, which is how
        they appear in a graph file and how ``add_sequence_edge`` preserves
        them. Sorting them here would renumber the indices that path
        constraints and adjacency lists already point at.
        """
        if not self.sequence_edges:
            return [], []
 
        intervals: list[AmpliconInterval] = []
        end_nodes: list[Node] = []
 
        def close(chr: str, start: int, end: int) -> None:
            intervals.append(
                AmpliconInterval(
                    chr=chr, start=start, end=end, amplicon_id=self.amplicon_idx
                )
            )
            end_nodes.append(Node(chr, start, Strand.REVERSE))
            end_nodes.append(Node(chr, end, Strand.FORWARD))
 
        first = self.sequence_edges[0]
        run_chr, run_start = first.chr, first.start
        for previous, edge in zip(
            self.sequence_edges, self.sequence_edges[1:]
        ):
            if edge.chr != previous.chr or edge.start != previous.end + 1:
                close(run_chr, run_start, previous.end)
                run_chr, run_start = edge.chr, edge.start
        close(run_chr, run_start, self.sequence_edges[-1].end)
        return intervals, end_nodes
 
 
def longest_path_constraints(
    constraints: list[PathConstraint],
) -> list[FinalizedPathConstraint]:
    """Reduce subwalks to edge multiplicities, keeping only the maximal ones.
 
    A constraint contained in another is redundant: satisfying the longer one
    satisfies it too, so dropping it removes a model row without changing the
    feasible set. Containment is per-edge -- every edge present at least as
    often. Ties (mutually contained, i.e. equal) keep the first, so an exact
    duplicate is dropped rather than both being discarded.
    """
    finalized = [
        FinalizedPathConstraint(
            edge_counts=_edge_counts(constraint.path),
            pc_idx=index,
            support=constraint.support,
        )
        for index, constraint in enumerate(constraints)
    ]
 
    def contains(outer: FinalizedPathConstraint, inner: FinalizedPathConstraint) -> bool:
        return all(
            outer.edge_counts.get(edge, 0) >= count
            for edge, count in inner.edge_counts.items()
        )
 
    maximal: list[FinalizedPathConstraint] = []
    for index, candidate in enumerate(finalized):
        if not candidate.edge_counts:
            continue
        redundant = any(
            contains(other, candidate)
            and (other_index < index or not contains(candidate, other))
            for other_index, other in enumerate(finalized)
            if other_index != index and other.edge_counts
        )
        if not redundant:
            maximal.append(candidate)
    return maximal
 
 
def _edge_counts(path: list[Node | EdgeId]) -> dict[EdgeId, int]:
    """How often each edge appears in an alternating node/edge walk."""
    counts: dict[EdgeId, int] = {}
    for element in path:
        if isinstance(element, EdgeId):
            counts[element] = counts.get(element, 0) + 1
    return counts