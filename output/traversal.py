from __future__ import annotations
 
import logging
from collections import defaultdict
 
from cycle_extractor.datatypes import (
    BreakpointGraph,
    DirectedEdge,
    DirectedWalk,
    EdgeId,
    EdgeType,
    Node,
    OptimizationWalk,
    Strand,
)
 
logger = logging.getLogger(__name__) 


def eulerian_traversal(
    g: BreakpointGraph, edge_counts_next_cycle: OptimizationWalk
) -> list[DirectedWalk]:
    """Return an eulerian traversal of a cycle/walk, represented by a list of
     directed edges.

    g: breakpoint graph (object)
    edges_counts_next_cycle: subgraph induced by the cycle, as a dict that maps an edge
        to its multiplicity
    """ 
    components: list[DirectedWalk] = []
    
    # TBD - implement the Hierholzer's algorithm that CE implemented (line 1092-)
    # You can define any helper functions by yourself
    #
    if len(components) > 1:
        logger.warning(
            "Extracted edges form %d disconnected components; reporting them "
            "as separate walks. Enable enforce_connectivity to prevent this.",
            len(components),
        )
    return components
 
 
