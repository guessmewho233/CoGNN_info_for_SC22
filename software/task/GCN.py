import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from task.common import *


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_feats, n_hidden)) # input layer
        for i in range(n_layers - 2): # hidden layers
            self.layers.append(GCNConv(n_hidden, n_hidden))
        self.layers.append(GCNConv(n_hidden, n_classes)) # output layer

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
    edge_weight_size = num_edges * 4 # float32
    feat_size = graph.features.nelement() * graph.features.element_size()
    label_size = graph.labels.nelement() * graph.labels.element_size()

    graph_size = [edge_index_size, edge_weight_size, feat_size, label_size] # edge_index, features and labels

    for size in graph_size:
        padded_size = calc_pad(size)
        computation_active_bytes += padded_size
        computation_peak_bytes += padded_size
    
    # add self loops from PyG
    mask = graph.edge_index[0] == graph.edge_index[1]
    add_loop_num_edges = num_nodes + num_edges - torch.count_nonzero(mask).item()
    loop_edge_index_size = edge_index_size * add_loop_num_edges / num_edges
    loop_edge_weight_size = edge_weight_size * add_loop_num_edges / num_edges
    padded_loop_edge_index_size, padded_loop_edge_weight_size = calc_pad(loop_edge_index_size), calc_pad(loop_edge_weight_size)
    
    for shape in para_shape_list[0]: # assuming float32 type (output/features)
        if len(shape) == 2: # 2-D weights
            padded_output_size = calc_pad(num_nodes * shape[1] * 4)
            computation_active_bytes += padded_output_size + padded_loop_edge_index_size + padded_loop_edge_weight_size # after GCN layer

            # propagate node features to edges
            edge_feat_size = (feat_size * add_loop_num_edges * 2 / num_nodes) * shape[1] / feat_length
            padded_edge_feat_size = calc_pad(edge_feat_size)

            computation_active_bytes += padded_edge_feat_size + padded_output_size # during propagation
            if computation_active_bytes > computation_peak_bytes:
                computation_peak_bytes = computation_active_bytes
            computation_active_bytes -= padded_edge_feat_size # only save features

    # multiply the threshold to ensure safe memory
    computation_peak_bytes = calc_pad(int(computation_peak_bytes * comp_ratio))

    return computation_peak_bytes
