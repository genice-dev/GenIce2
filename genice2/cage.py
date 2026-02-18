import string
from logging import getLogger

import networkx as nx
import numpy as np
from cycless.cycles import centerOfMass, cycles_iter
from cycless.polyhed import cage_to_graph, polyhedra_iter

# for cage assessment
from graphstat import GraphStat


def assign_unused_label(basename, labels):
    enum = 0
    label = f"A{basename}"
    while label in labels:
        char = string.ascii_lowercase[enum]
        label = f"A{basename}{char}"
        enum += 1
    return label


def make_cage_expression(ring_ids, ringlist):
    ringcount = [0 for i in range(9)]
    for ring in ring_ids:
        ringcount[len(ringlist[ring])] += 1
    index = []
    for i in range(9):
        if ringcount[i] > 0:
            index.append(f"{i}^{ringcount[i]}")
    index = " ".join(index)
    return index


def center_of_graph(g: nx.Graph, node_frac: np.ndarray) -> np.ndarray:
    # 辺をたどりながらグラフの中心を計算する。
    first_node = next(iter(g.nodes()))
    positions = {first_node: node_frac[first_node]}

    def dfs(node):
        for nei in g.neighbors(node):
            origin = positions[node]
            if nei not in positions:
                delta = node_frac[nei] - origin
                delta -= np.floor(delta + 0.5)
                positions[nei] = origin + delta
                dfs(nei)

    dfs(first_node)
    return np.mean(list(positions.values()), axis=0)


def assess_cages(graph, node_frac):
    """Assess cages from  the graph topology.

    Args:
        graph (graph-like): HB network
        nodepos (np.Array): Positions of the nodes
    """
    logger = getLogger()

    # Prepare the list of rings
    # taking the positions in PBC into account.
    ringlist = [
        [int(x) for x in ring]
        for ring in cycles_iter(nx.Graph(graph), 8, pos=node_frac)
    ]

    # Positions of the centers of the rings.
    ringpos = np.array([centerOfMass(ringnodes, node_frac) for ringnodes in ringlist])

    MaxCageSize = 22
    cage_fracs = []
    cagetypes = []
    # data storage of the found cages
    db = GraphStat()
    labels = set()
    g_id2label = dict()

    # Detect cages and classify
    cages = [cage for cage in polyhedra_iter(ringlist, MaxCageSize)]
    cage_graphs = [cage_to_graph(cage, ringlist) for cage in cages]
    cage_fracs = [center_of_graph(g, node_frac) for g in cage_graphs]
    FrankKasper = True
    for cage, g in zip(cages, cage_graphs):
        cagesize = len(cage)
        g_id = db.query_id(g)
        # if it is a new cage type
        if g_id < 0:
            # new type!
            # register the last query
            g_id = db.register()

            # prepare a new label
            label = assign_unused_label(cagesize, labels)
            g_id2label[g_id] = label
            labels.add(label)

        else:
            label = g_id2label[g_id]
        cagetypes.append(label)
        # cage expression
        faces = make_cage_expression(cage, ringlist)
        logger.info(f"    Cage type: {label} ({faces})")
        if faces not in ("5^12", "5^12 6^2", "5^12 6^3", "5^12 6^4"):
            FrankKasper = False
    if FrankKasper:
        logger.info("    Frank-Kasper type.")
        cagecount = dict(A12=0, A14=0, A15=0, A16=0)
        for label in cagetypes:
            cagecount[label] += 1
        logger.info(f"    Cage count: {cagecount}")
        # Three canonical structure types: CS1, CS2, HS1
        M = np.array(
            [
                [2 / 46, 6 / 46, 0, 0],
                [16 / 136, 0, 0, 8 / 136],
                [3 / 40, 2 / 40, 2 / 40, 0],
            ]
        ).T
        b = np.array(
            [cagecount["A12"], cagecount["A14"], cagecount["A15"], cagecount["A16"]]
        )
        Mplus = np.linalg.inv(M.T @ M) @ M.T
        x = Mplus @ b
        x /= np.sum(x)
        logger.info(
            f"    Composition of the canonical structure types (CS1, CS2, HS1): {x}"
        )
    if len(cage_fracs) == 0:
        logger.info("    No cages detected.")
    return np.array(cage_fracs), cagetypes
