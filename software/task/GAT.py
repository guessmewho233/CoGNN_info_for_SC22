import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch.nn import Sequential, Linear
from task.common import *

class GAT(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_feats, 8, heads=8, dropout=0.6)) # input layer
        for i in range(n_layers - 2): # hidden layer
            self.layers.append(GATConv(8 * 8, 8, heads=8, dropout=0.6))
        self.layers.append(GATConv(8 * 8, n_classes, heads=1, concat=False, dropout=0.6))

    def forward(self, features, edge_index):
        x = features
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
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

    graph_size = [edge_index_size, feat_size, label_size] # edge_index, features and labels

    for size in graph_size:
        padded_size = calc_pad(size)
        computation_active_bytes += padded_size
        computation_peak_bytes += padded_size
    
    # add self loops from PyG
    mask = graph.edge_index[0] == graph.edge_index[1]
    add_loop_num_edges = num_nodes + num_edges - torch.count_nonzero(mask).item()
    padded_loop_edge_index_size = calc_pad(edge_index_size * add_loop_num_edges / num_edges)

    for i in range(len(para_shape_list[0])):
        shape = para_shape_list[0][i]
        heads = 8   # hidden layers
        if i == len(para_shape_list[0]) - 1: # output layer
            back_lin_output_size = padded_lin_output_size
            back_edge_alpha_size = padded_edge_alpha_size
            heads = 1

        # calculate lin and alpha output
        padded_lin_output_size, padded_alpha_size = calc_pad(num_nodes * shape[0] * 4), calc_pad(num_nodes * heads * 4)

        computation_active_bytes += 2 * (padded_lin_output_size + padded_alpha_size)
        if computation_active_bytes > computation_peak_bytes:
            computation_peak_bytes = computation_active_bytes
        computation_active_bytes -= padded_lin_output_size

        # add self loop to edge_index
        computation_active_bytes += padded_loop_edge_index_size
        if computation_active_bytes > computation_peak_bytes:
            computation_peak_bytes = computation_active_bytes

        # propagate node features to edges
        edge_feat_size, edge_alpha_size = add_loop_num_edges * shape[0] * 4, add_loop_num_edges * heads * 4
        padded_edge_feat_size, padded_edge_alpha_size = calc_pad(edge_feat_size), calc_pad(edge_alpha_size)
        computation_active_bytes += padded_edge_feat_size + 2 * padded_edge_alpha_size
        if computation_active_bytes > computation_peak_bytes:
            computation_peak_bytes = computation_active_bytes

        # process alpha and generate message (leaky_relu)
        computation_active_bytes += 2 * padded_edge_alpha_size 
        if computation_active_bytes > computation_peak_bytes:
            computation_peak_bytes = computation_active_bytes

        # softmax details
        padded_add_alpha_size = calc_pad((add_loop_num_edges - num_edges) * heads * 4)

        computation_active_bytes += 2 * padded_edge_alpha_size # scatter/index_select
        if computation_active_bytes > computation_peak_bytes:
            computation_peak_bytes = computation_active_bytes
        computation_active_bytes -= padded_add_alpha_size

        computation_active_bytes += 2 * padded_edge_alpha_size # exp
        if computation_active_bytes > computation_peak_bytes:
            computation_peak_bytes = computation_active_bytes
        computation_active_bytes -= padded_edge_alpha_size

        computation_active_bytes += padded_edge_alpha_size # scatter/index_select
        if computation_active_bytes > computation_peak_bytes:
            computation_peak_bytes = computation_active_bytes
        computation_active_bytes -= (padded_edge_alpha_size - padded_add_alpha_size)
        computation_active_bytes += padded_edge_alpha_size
        if computation_active_bytes > computation_peak_bytes:
            computation_peak_bytes = computation_active_bytes
        computation_active_bytes -= padded_add_alpha_size

        computation_active_bytes += 2 * padded_edge_alpha_size # softmax return
        if computation_active_bytes > computation_peak_bytes:
            computation_peak_bytes = computation_active_bytes
        computation_active_bytes -= 3 * padded_edge_alpha_size

        # aggregation and update
        computation_active_bytes += padded_edge_feat_size + padded_lin_output_size
        if computation_active_bytes > computation_peak_bytes:
            computation_peak_bytes = computation_active_bytes
        computation_active_bytes -= (padded_edge_feat_size + 2 * (padded_edge_alpha_size + padded_alpha_size))

    # output gradient - backward
    gradient_output_size = 2 * back_lin_output_size + 16 * back_edge_alpha_size

    computation_active_bytes += gradient_output_size
    if computation_active_bytes > computation_peak_bytes:
        computation_peak_bytes = computation_active_bytes
    computation_active_bytes -= gradient_output_size # release gradients

    # multiply the threshold to ensure safe memory
    computation_peak_bytes = calc_pad(int(computation_peak_bytes * comp_ratio))

    return computation_peak_bytes
