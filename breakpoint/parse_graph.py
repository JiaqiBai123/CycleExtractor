"""Reading a breakpoint graph from a ``*_graph.txt`` file.
 
The format is CoRAL's, which is AmpliconArchitect's with a path-constraint
section added. Sections appear in order, each introduced by a header line that
this parser ignores -- the record type is the first field of every data line::
 
    sequence         chr7:54659673-  chr7:54763281+  4.15  45.9  103609  576
    concordant       chr7:54763281+->chr7:54763282-  4.15  26
    discordant       chr7:55155021-->chr7:55127266+  86.50  978
    source           chr7:54659673-  2.31
    path_constraint  e2+:1,c2-:1,e3+:1  6
 
Only the copy number is read from each edge; the trailing coverage and
read-count columns are evidence for a copy number CE takes as given. Parsing
does not infer or refit anything.
 
The ``-->`` in a discordant line is genuinely ambiguous to split on -- a
reverse-strand node ends in ``-`` and the separator is ``->``. Splitting on
``->`` from the right is what disambiguates it.
"""
 
from __future__ import annotations
 
import logging
import pathlib
from collections.abc import Iterable, Iterator
 
from cycle_extractor.breakpoint.breakpoint_graph import GraphBuilder
from cycle_extractor.datatypes import (
    BreakpointGraph,
    EdgeId,
    EdgeType,
    Node,
    Strand,
)
 
logger = logging.getLogger(__name__)
 
SEPARATOR = "->"
 
 
def parse_graph(
    path: str | pathlib.Path, amplicon_idx: int = 0
) -> BreakpointGraph:
    """Read one graph file.
 
    Path constraints are applied after every edge, because they reference edges
    by index and the indices only exist once the edges are in.
    """
    path = pathlib.Path(path)
    with path.open() as handle:
        graph = parse_graph_lines(handle, amplicon_idx=amplicon_idx)
    logger.info("Parsed %s.", path)
    return graph
 
 
def parse_graph_lines(
    lines: Iterable[str], amplicon_idx: int = 0
) -> BreakpointGraph:
    """Parse from any line source -- a file, a stream, a list."""
    builder = GraphBuilder(amplicon_idx=amplicon_idx)
    constraints: list[tuple[str, int]] = []
 
    for number, line in enumerate(lines, start=1):
        fields = line.split()
        if not fields or fields[0].endswith(":"):
            continue  # blank, or a section header like "SequenceEdge:"
        try:
            _parse_record(builder, fields, constraints)
        except (ValueError, IndexError) as error:
            raise ValueError(
                f"Line {number} is not a valid graph record: {line.strip()!r} "
                f"({error})"
            ) from error
 
    for token, support in constraints:
        builder.add_path_constraint(_parse_path(builder, token), support)
 
    if not constraints:
        logger.warning(
            "No path constraints in the graph file; extraction will be "
            "unconstrained by long-read subwalks."
        )
    return builder.build()
 
 
def parse_graphs(directory: str | pathlib.Path) -> list[BreakpointGraph]:
    """Read every ``*_graph.txt`` in a directory, sorted by amplicon number.
 
    Files are named ``<prefix>_amplicon<N>_graph.txt``; sorting on the parsed
    ``N`` rather than the filename keeps amplicon 2 ahead of amplicon 10.
    """
    directory = pathlib.Path(directory)
    paths = sorted(directory.glob("*graph.txt"), key=_amplicon_number)
    if not paths:
        raise FileNotFoundError(f"No *graph.txt files under {directory}.")
    return [
        parse_graph(path, amplicon_idx=index)
        for index, path in enumerate(paths)
    ]
 
 
# --------------------------------------------------------------------------- #
# Records                                                                      #
# --------------------------------------------------------------------------- #
 
 
def _parse_record(
    builder: GraphBuilder,
    fields: list[str],
    constraints: list[tuple[str, int]],
) -> None:
    match fields[0]:
        case "sequence":
            start, end = _parse_node(fields[1]), _parse_node(fields[2])
            builder.add_sequence_edge(
                start.chr, start.pos, end.pos, cn=float(fields[3])
            )
        case "concordant":
            node1, node2 = _parse_edge_nodes(fields[1])
            builder.add_concordant_edge(node1, node2, cn=float(fields[2]))
        case "discordant":
            node1, node2 = _parse_edge_nodes(fields[1])
            builder.add_discordant_edge(node1, node2, cn=float(fields[2]))
        case "source":
            builder.add_source_edge(
                _parse_node(fields[1]), cn=float(fields[2])
            )
        case "path_constraint":
            constraints.append((fields[1], int(fields[2])))
        case unknown:
            logger.debug("Ignoring unrecognized record type %r.", unknown)
 
 
def _parse_node(token: str) -> Node:
    """``chr7:54659673-`` into a node."""
    chromosome, _, rest = token.partition(":")
    return Node(chromosome, int(rest[:-1]), Strand(rest[-1]))
 
 
def _parse_edge_nodes(token: str) -> tuple[Node, Node]:
    """``chr7:55155021-->chr7:55127266+`` into its two nodes.
 
    Split from the right: the left node's strand character is itself a ``-``,
    so a left split would cut inside it.
    """
    left, separator, right = token.rpartition(SEPARATOR)
    if not separator:
        raise ValueError(f"No {SEPARATOR!r} in breakpoint edge {token!r}")
    return _parse_node(left), _parse_node(right)
 
 
def _parse_path(builder: GraphBuilder, token: str) -> list[Node | EdgeId]:
    """``e2+:1,c2-:1,e3+:1`` into an alternating node/edge walk.
 
    Each piece is ``<type><1-based index><strand>:<count>``. The strand on a
    sequence edge says which way the walk crosses it, which is what orders the
    two nodes around it; breakpoint edges contribute no node of their own,
    since their endpoints are the sequence edges' nodes on either side.
 
    Indices in the file are 1-based, and repeats are recorded by the edge
    appearing again rather than by the ``:count`` suffix -- so the count is
    read but not used to weight anything here; ``_edge_counts`` derives
    multiplicity from the walk itself.
    """
    walk: list[Node | EdgeId] = []
    for piece in token.split(","):
        identifier, _, _count = piece.partition(":")
        edge_type = EdgeType(identifier[0])
        index = int(identifier[1:-1]) - 1
        strand = Strand(identifier[-1])
        edge_id = EdgeId(edge_type, index)
 
        if edge_type is not EdgeType.SEQUENCE:
            walk.append(edge_id)
            continue
 
        edge = builder.sequence_edges[index]
        first, second = (
            (edge.node1, edge.node2)
            if strand is Strand.FORWARD
            else (edge.node2, edge.node1)
        )
        walk.extend([first, edge_id, second])
    return walk
 
 
def _amplicon_number(path: pathlib.Path) -> tuple[int, str]:
    """Sort key pulling the amplicon number out of the filename."""
    for part in path.stem.split("_"):
        if part.startswith("amplicon") and part[8:].isdigit():
            return int(part[8:]), path.name
    return 0, path.name

