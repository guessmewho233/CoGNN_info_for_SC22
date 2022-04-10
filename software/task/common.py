import os
import dgl
import torch

# gnn architecture configuration
gpu = 0
num_epochs = 20
hidden_dim = 64

# gnn threshold
comp_ratio = 1.15

gnndatasets = [
    ('citeseer', 3703, 6), # small datasets (< 64MB)
    ('cora', 1433, 7),
    ('pubmed', 500, 3),
    ('ppi', 50, 121),
    ('PROTEINS_full', 29, 2),
    ('artist', 100, 12),
    ('soc-BlogCatalog', 128, 39),
    ('ddi', 512, 32),

    ('OVCAR-8H', 66, 2), # medium datasets (64MB ~ 1GB)
    ('Yeast', 74, 2),
    ('DD', 89, 2),
    ('SW-620H', 66, 2),
    ('com-amazon', 96, 22),
    ('amazon0601', 96, 22),
    ('arxiv', 128, 40),
    ('collab', 128, 32),
    ('ppa', 58, 16),

    ('reddit.dgl', 602, 50), # large datasets (> 1GB)
    ('products', 100, 47),
    ('protein', 8, 2),
    ('citation', 1433, 7),
    ('TWITTER-Real-Graph-Partial', 1323, 2)
]


class GraphSummary(): # graph data summary
    def __init__(self, name, edge_index, features, labels):
        self.name = name
        self.edge_index = edge_index
        self.features = features
        self.labels = labels


def load_graph_data(data):
    g = dgl.load_graphs("../data/graph/{}.graph".format(data))[0][0]
    edge_index = torch.stack((g.edges()[0], g.edges()[1]), 0).long()
    features = g.ndata['feat']
    labels = g.ndata['label']

    return data, edge_index, features, labels

## get the shapes of tensors and parameters
kMinBlockSize = 512

def calc_pad(tsize):
    return kMinBlockSize * ((tsize + kMinBlockSize - 1) // kMinBlockSize) 

def set_fullname(mod, fullname):
    mod.fullname = fullname
    if len(list(mod.children())) == 0:
        for index, p in enumerate(mod.parameters()):
            p.reserved_name = '%s->p%d' % (fullname, index)
    for child_name, child in mod.named_children():
        child_fullname = '%s->%s' % (fullname, child_name)
        set_fullname(child, child_fullname)


def group_to_shape(group):
    shape_list = []
    param_list = []
    buf_list = []
    mod_list = []

    def travel_layer(mod):
        if len(list(mod.children())) == 0:
            mod_list.append(mod)
        else:
            for child in mod.children():
                travel_layer(child)
    for mod in group:
        travel_layer(mod)

    for mod in mod_list:
        for p in mod.parameters():
            shape_list.append(p.shape)
            param_list.append(p)
    for mod in mod_list:
        for key, buf in mod._buffers.items():
            if buf is not None and buf.dtype is torch.float32:
                shape_list.append(buf.shape)
                buf_list.append((mod, key))

    return shape_list, param_list, buf_list, mod_list


def group_to_batch(group):
    mod_list = []
    def travel_layer(mod):
        if len(list(mod.children())) == 0:
            mod_list.append(mod)
        else:
            for child in mod.children():
                travel_layer(child)
    for mod in group:
        travel_layer(mod)

    def pad(t, blockSize):
        length = t.nelement()
        size = length * t.element_size()
        padded_size = blockSize * ((size + blockSize - 1) // blockSize)
        padded_length = padded_size // t.element_size()
        t_padded = torch.zeros(padded_length)
        t_padded[:length] = t
        return t_padded

    tensor_list = []
    for mod in mod_list:
        for p in mod.parameters():
            tensor_list.append(pad(p.view(-1), kMinBlockSize))
    for mod in mod_list:
        for _, buf in mod._buffers.items():
            if buf is not None and buf.dtype is torch.float32:
                tensor_list.append(pad(buf.view(-1), kMinBlockSize))
    
    if len(tensor_list) > 0:
        batched_tensor = torch.cat(tensor_list)
    else:
        batched_tensor = None
    
    modname_list = [mod.fullname for mod in mod_list]

    return batched_tensor, modname_list


def group_to_para_shape(group):
    mod_list = []
    shape_list = []

    def travel_layer(mod):
        if len(list(mod.children())) == 0:
            mod_list.append(mod)
        else:
            for child in mod.children():
                travel_layer(child)
    for mod in group:
        travel_layer(mod)
    
    for mod in mod_list:
        for p in mod.parameters():
            shape_list.append(p.shape)

    return shape_list
