from collections import defaultdict

import torch

from sigmadock.chem.processing import EDGE_ENTITY, NODE_ENTITY


def summarize_degrees_by_node_entity(graph: dict) -> None:
    """
    Prints degree statistics (total + per-edge-id) for each node-entity category,
    grouping together all edge-names that share the same integer ID.
    """
    edge_index = graph.edge_index  # [2, E]
    edge_entity = graph.edge_entity  # [E]
    node_entity = graph.node_entity  # [N], each value in NODE_ENTITY.values()
    N = graph.x.size(0)

    # 1) Build a map:  id -> [list of edge-names]
    id2names = defaultdict(list)
    for name, eid in EDGE_ENTITY.items():
        id2names[eid].append(name)

    # 2) Compute degree per node per unique ID
    max_id = max(id2names.keys())
    deg_by_id = torch.zeros((max_id + 1, N), dtype=torch.long, device=edge_entity.device)

    for eid, _ in id2names.items():
        mask = edge_entity == eid
        ei = edge_index[:, mask]
        if ei.numel() == 0:
            continue
        nodes, counts = torch.unique(ei.flatten(), return_counts=True)
        deg_by_id[eid, nodes] = counts // 2  # undirected edges

    # total degree per node = sum over unique IDs
    total_deg = deg_by_id.sum(dim=0).float()

    # 3) For each node-entity group, summarize
    for nename, neidx in NODE_ENTITY.items():
        nodes = (node_entity == neidx).nonzero(as_tuple=True)[0]
        if nodes.numel() == 0:
            continue

        td = total_deg[nodes]
        print(f"\n=== Node Entity: {nename} (code={neidx}) ===")
        print(f" Nodes: {nodes.numel():4d}   Total Degree  avg={td.mean():.2f}, min={td.min():.0f}, max={td.max():.0f}")

        # breakdown per unique edge-ID
        for eid, names in id2names.items():
            d = deg_by_id[eid, nodes].float()
            if d.sum() == 0:
                continue
            label = "/".join(names)
            print(f"   • {label:20s}  avg={d.mean():.2f}, min={d.min():.0f}, max={d.max():.0f}")
