import torch
import cvxpy as cp
import scipy.sparse as sp
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from torch.nn.functional import cosine_similarity
import heapq


def retrieval_path(graph, pred, rel_emb, ent_emb, ent2id, rel2id, q_entity, question_emb):
    sub_pred = []
    sub_rel_emb = []
    sub_ent_emb = []
    inv_entity_vocab = {}
    inv_rel_vocab = {}
    triples = []
    entity_cnt, rel_cnt = 0, 0  

    # Step 1: analyze the graph & compute edge embedding
    for triple in graph:  
        u, r, v = triple[0], triple[1], triple[2]  

        if u not in inv_entity_vocab:
            inv_entity_vocab[u] = entity_cnt
            entity_cnt += 1
            u_id = ent2id[u]
            sub_pred.append(pred[u_id])
            sub_ent_emb.append(ent_emb[u_id])

        if v not in inv_entity_vocab:
            inv_entity_vocab[v] = entity_cnt
            entity_cnt += 1
            v_id = ent2id[v]
            sub_pred.append(pred[v_id])
            sub_ent_emb.append(ent_emb[v_id])

        if r not in inv_rel_vocab:
            inv_rel_vocab[r] = rel_cnt
            rel_cnt += 1
            r_id = rel2id[r]
            sub_rel_emb.append(rel_emb[r_id])

        u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]
        triples.append((u, v, r))

    sub_ent2id = inv_entity_vocab
    sub_rel2id = inv_rel_vocab
    sub_id2ent = {v: k for k, v in sub_ent2id.items()}
    sub_id2rel = {v: k for k, v in sub_rel2id.items()}

    # Step 2: compute cos sim
    edge_weights = []
    device = question_emb.device
    for (u, v, r) in triples:
        rel_vec = sub_rel_emb[r].clone().detach().to(device)
        weight = cosine_similarity(rel_vec.unsqueeze(0), question_emb.unsqueeze(0)).item()
        weight = (weight + 1) / 2
        edge_weights.append(1-weight)  

    # Step 3: choose top-k entities
    retrieved_ent = []
    retrieved_ent_idx = torch.topk(torch.tensor(sub_pred), 3, dim=-1).indices
    retrieved_ent = [sub_id2ent[i.item()] for i in retrieved_ent_idx]

    # Step 4: compute shortest path
    G = nx.DiGraph()
    for i, (u, v, r) in enumerate(triples):
        G.add_edge(sub_id2ent[u], sub_id2ent[v], weight=edge_weights[i])

    # Step 5: path formation
    paths = []
    for item in q_entity:
        if item in sub_ent2id:
            for target in retrieved_ent:
                if target in sub_ent2id:
                        path_nodes = a_star_search(G, item, target, question_emb, sub_ent_emb, sub_ent2id, sub_pred)
                        if path_nodes:
                            path_with_rel = []
                            for i in range(len(path_nodes) - 1):
                                u, v = path_nodes[i], path_nodes[i + 1]
                                rel = None
                                for (src, dst, r) in triples:
                                    if sub_id2ent[src] == u and sub_id2ent[dst] == v:
                                        rel = sub_id2rel[r]
                                        break
                                if rel:
                                    if v == target:
                                        path_with_rel.append(f"{u} -> {rel} -> {v}")
                                    else:
                                        path_with_rel.append(f"{u} -> {rel} ->")
                                else:
                                    path_with_rel.append(f"{u} -> {v}")  

                            paths.append(" ".join(path_with_rel))
                        else:
                            paths.append(f"{item} -> {target}")
    print(paths)      
    return paths

# A*
def a_star_search(G, start, goal, question_emb, sub_ent_emb, sub_ent2id, sub_pred):
    def heuristic(u, v):
        u_emb = sub_ent_emb[sub_ent2id[u]].clone().detach().to(question_emb.device).unsqueeze(0)
        v_emb = sub_ent_emb[sub_ent2id[v]].clone().detach().to(question_emb.device).unsqueeze(0)
        return (1 - cosine_similarity(u_emb, v_emb).item()) / 2  
    queue = [(0, start, [])]  
    visited = set()
    best_cost = {start: 0}

    while queue:
        cost, node, path = heapq.heappop(queue)

        if node in visited:
            continue
        visited.add(node)

        path = path + [node]
        if node == goal:
            return path  
        
        for neighbor in G.neighbors(node):
            edge_weight = G[node][neighbor]['weight']
            g_cost = cost + edge_weight  
            h_cost = heuristic(neighbor, goal)  
            total_cost = g_cost + h_cost  
            
            if neighbor not in best_cost or total_cost < best_cost[neighbor]:
                best_cost[neighbor] = total_cost
                heapq.heappush(queue, (g_cost, neighbor, path))

    return None  
