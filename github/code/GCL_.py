import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv  
import pandas as pd  
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class GCL_GCN(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(GCL_GCN, self).__init__()
        self.conv1 = GCNConv(feature_dim, 128)
        self.conv2 = GCNConv(128, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def feature_masking(data, mask_rate=0.2, transaction_data=None):

    transaction_counts = transaction_data['total_transactions'].values
    mask_prob = mask_rate * (1 - transaction_counts / transaction_counts.max())

    mask_prob = torch.tensor(mask_prob, dtype=torch.float32)

    mask = torch.rand(size=(data.num_nodes, data.x.size(1))) > mask_prob[:, None]
    data.x = data.x * mask.float()
    return data

def edge_perturbation(data, edges_df, perturb_rate=0.2, beta_T=0.5, beta_A=0.5):
    edge_index = data.edge_index
    from_nodes = edge_index[0]
    to_nodes = edge_index[1]

    timestamps = edges_df['TimeStamp'].values
    values = edges_df['Value'].values

    delete_probabilities = []

    unique_edges = edges_df.groupby(['From', 'To'])

    for i in range(edge_index.size(1)):
        u = from_nodes[i].item()
        v = to_nodes[i].item()

        edge_transactions = unique_edges.get_group((u, v))

        t_first = edge_transactions['TimeStamp'].min()
        t_last = edge_transactions['TimeStamp'].max()

        delta_t = abs(edge_transactions['TimeStamp'] - t_first)

        T_e = delta_t.mean()

        total_value = edge_transactions['Value'].sum()
        v_uv = edge_transactions.iloc[0]['Value']

        A_e = v_uv / total_value

        P_delete = beta_T * T_e + beta_A * A_e
        delete_probabilities.append(P_delete)

    delete_probabilities = torch.tensor(delete_probabilities)

    mask = torch.rand(delete_probabilities.size(0)) < perturb_rate * delete_probabilities
    data.edge_index = edge_index[:, ~mask]

    return data

def contrastive_loss(embedding_1, embedding_2, margin=1.0):

    cosine_similarity = F.cosine_similarity(embedding_1, embedding_2)

    positive_loss = torch.mean(1 - cosine_similarity)
    negative_loss = torch.mean(torch.clamp(margin - cosine_similarity, min=0))

    loss = positive_loss + negative_loss
    return loss
