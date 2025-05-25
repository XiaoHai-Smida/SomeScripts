import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from gcn_layers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TSGR(nn.Module):
    def __init__(self, input_size, feature_embedding_size, hidden_size, output_size, nhead):
        super(TSGR, self).__init__()
        self.feature_size = input_size
        self.feature_embedding_size = feature_embedding_size
        self.ff_hidden_size = hidden_size
        self.output_size = output_size

        # Transformer
        self.former_layers = nn.ModuleList()
        self.former_layers.append(nn.Linear(self.feature_size,self.feature_embedding_size))
        self.former_layers.append(
            TransformerEncoder(TransformerEncoderLayer(
                d_model=self.feature_embedding_size, nhead=nhead,
                dim_feedforward=self.ff_hidden_size, dropout=0.5), num_layers=2))

        # GCN
        self.GCN_layers = GraphBaseBlock(
            [self.feature_embedding_size, self.feature_embedding_size], 1, dropout=0.5, withloop=False)

        # linear
        self.linear = nn.Linear(self.feature_embedding_size, self.output_size)

        self.bn = nn.BatchNorm1d(self.feature_embedding_size)

        self.Pooling = nn.Sequential(
            nn.Linear(self.feature_embedding_size, 32),
            nn.ReLU(),
            Transpose(1, 2),
            nn.AvgPool1d(kernel_size=3, stride=3),   # batch,90,x -> batch,30,x
            Transpose(1, 2),
            nn.Linear(32,16),
            Transpose(1, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16,self.output_size)
        )

    def forward(self, node_features, edge_features):
        # node_features = torch.reshape(node_features, (-1, node_features.size(2)))
        # edge_features = torch.reshape(edge_features, (-1, edge_features.size(2)))
        node_features = self.former_layers[0](node_features)
        node_features = node_features.permute(1, 0, 2)
        self_score = self.former_layers[1](node_features).permute(1, 0, 2)
        self_score = torch.reshape(self_score,(-1, node_features.shape[-1]))
        gcn_output = self.GCN_layers(self_score, edge_features)

        node_features = self.bn(gcn_output)
        node_features = torch.reshape(node_features, (-1, self.feature_size, self.feature_embedding_size)).permute(1, 0, 2)
        self_score = self.former_layers[1](node_features).permute(1, 0, 2)
        self_score = torch.reshape(self_score, (-1, node_features.shape[-1]))
        gcn_output = self.GCN_layers(self_score, edge_features)

        output = torch.reshape(gcn_output, (-1,  self.feature_size, self.feature_embedding_size))
        output = self.Pooling(output)

        return output

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)
