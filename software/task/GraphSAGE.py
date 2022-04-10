import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from task.common import *


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden)) # input layer
        for i in range(n_layers - 2): # hidden layers
            self.layers.append(SAGEConv(n_hidden, n_hidden))
        self.layers.append(SAGEConv(n_hidden, n_classes)) # output layer

    def forward(self, features, edge_index):
        x = features
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x, edge_index))

        return F.log_softmax(x, dim=-1)


def partition_model(model):
    group_list = [[child] for child in model.children()]
    return group_list


def get_comp_size(graph, para_shape_list):
    computation_peak_bytes, computation_active_bytes = 0, 0

    num_nodes, num_edges, feat_length = graph.features.shape[0], graph.edge_index.shape[1], graph.features.shape[1]

    edge_index_size = graph.edge_index.nelement() * graph.edge_index.element_size()
    feat_size = graph.features.nelement() * graph.features.element_size()
    label_size = graph.labels.nelement() * graph.labels.element_size()

    # padded size
    padded_edge_index_size, padded_feat_size, padded_label_size = calc_pad(edge_index_size), calc_pad(feat_size), calc_pad(label_size)
    padded_graph_size = [padded_edge_index_size, padded_feat_size, padded_label_size] # edge_index, features and labels

    for padded_size in padded_graph_size:
        computation_active_bytes += padded_size
        computation_peak_bytes += padded_size
    
    sage_counter = 0
    for shape in para_shape_list[0]: # assuming float32 type (output/features)
        if len(shape) == 2 and sage_counter % 3 == 0: # 2-D weights
            padded_aggr_output_size = calc_pad(num_nodes * shape[1] * 4) # features after aggregation

            # propagate node features to edges
            edge_feat_size = (feat_size * num_edges / num_nodes) * shape[1] / feat_length
            padded_edge_feat_size = calc_pad(edge_feat_size)
            computation_active_bytes += padded_edge_feat_size + padded_aggr_output_size 

            if computation_active_bytes > computation_peak_bytes:
                computation_peak_bytes = computation_active_bytes
            computation_active_bytes -= padded_edge_feat_size # only save features

            # lin_r and lin_l (linear layers) output
            padded_lin_output_size = calc_pad(num_nodes * shape[0] * 4)

            computation_active_bytes += padded_lin_output_size * 2
            if computation_active_bytes > computation_peak_bytes:
                computation_peak_bytes = computation_active_bytes
        sage_counter += 1

    # multiply the threshold to ensure safe memory
    computation_peak_bytes = calc_pad(int(computation_peak_bytes * comp_ratio))

    return computation_peak_bytes
