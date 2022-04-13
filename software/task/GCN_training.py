import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch_geometric.nn import GCNConv

from task.common import *
from task.GCN import *

def import_func(model, graph):
    def train(model, graph):
        print(threading.currentThread().getName(),
            'GCN training {} >>>>>>>>>>'.format(graph.name), time.time())
#        before_time = time.time()
        model = model.cuda()
        features = graph.features.cuda(non_blocking=True)
        labels = graph.labels.cuda(non_blocking=True)
        edge_index = graph.edge_index.to(gpu, non_blocking=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss = None
        for epoch in range(num_epochs):
            model.train()
            output = model(features, edge_index)
            loss = F.nll_loss(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#            print('GCN {} Epoch {:05d}'.format(graph.name, epoch))
#        training_time = time.time() - before_time    
#        return training_time * 1000

        return loss.item()

    return train


def import_model(data, num_layers):
    # feats and classes of graph data
    feat_dim, num_classes = 32, 2
    for idata, ifeat, iclasses in gnndatasets:
        if idata == data:
            feat_dim, num_classes = ifeat, iclasses
            break
    
    # load graph data
    name, edge_index, features, labels = load_graph_data(data)
    graph = GraphSummary(name, edge_index, features, labels)

    model = GCN(feat_dim, hidden_dim, num_classes, num_layers, 0.5)

    # set full name to disguish
    FULL_NAME = 'GCN_{}'.format(data)
    set_fullname(model, FULL_NAME)

    return model, graph


def import_task(data, num_layers):
    model, graph = import_model(data, num_layers)
    func = import_func(model, graph)
    group_list = partition_model(model)
    shape_summary_list = [group_to_shape(group) for group in group_list]
    return model, func, shape_summary_list


def import_parameters(data, num_layers):
    model, graph = import_model(data, num_layers)
    group_list = partition_model(model)
    batch_list = [group_to_batch(group) for group in group_list]
    para_shape_list = [group_to_para_shape(group) for group in group_list]
    comp_total_bytes = get_comp_size(graph, para_shape_list)
    return batch_list, comp_total_bytes
