import networkx as nx
import numpy as np
import torch
from scipy.linalg import block_diag
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class S2VGraph(object):
    def __init__(self, g, node_features, adj):
        self.g = g
        self.node_features = node_features
        self.adj = adj

def construct_graph(sc, fc):
    num_subjs, num_nodes = sc.shape[0], sc.shape[1]
    g_list = []

    for subj in range(num_subjs):
        g = nx.Graph()
        node_features = []

        for node in range(num_nodes):
            g.add_node(node)
            row = fc[subj, node]
            row[node] =  1
            attr = np.array(row, dtype=np.float32)
            node_features.append(attr)

        # sc prepare
        sc_content = sc[subj]
        bottom5 = np.quantile(np.unique(sc_content), 0.05)
        sc_content[sc_content < bottom5] = 0.0
        # rows, cols = np.where(sc_content > 0)
        # g.add_weighted_edges_from(zip(rows, cols, sc[subj, rows, cols]))

        g_list.append(S2VGraph(g, np.array(node_features), sc_content))

    return g_list

def get_batch_data(graphs, labels, batch_size, train_state):
    num_samples = len(graphs)
    indeces = np.arange(num_samples)
    if train_state:
        np.random.shuffle(indeces)
    for start_idx in range(0, num_samples, batch_size):
        # end_idx = min(start_idx + batch_size, num_samples)
        end_idx = start_idx + batch_size
        if end_idx <= num_samples:
            batch_indices = indeces[start_idx:end_idx]

            batch_node_features = []
            batch_edge_adj = []
            for i in batch_indices:
                g = graphs[i]
                batch_node_features.append(g.node_features)
                batch_edge_adj.append(g.adj)
            batch_edge_adj = block_diag(*batch_edge_adj)
            batch_node_features = torch.tensor(batch_node_features, dtype=torch.float32).to(device)
            batch_edge_adj = torch.tensor(batch_edge_adj, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(labels[batch_indices], dtype=torch.float32).to(device)

            yield batch_node_features, batch_edge_adj, batch_labels

