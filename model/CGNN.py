from torch_geometric.nn import (
    SAGEConv,
    GCNConv,
    GATConv,
    JumpingKnowledge,
)
import torch.nn as nn
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import sum as sparsesum
from torch_sparse import mul
import torch.nn.functional as F

def row_norm(adj):
    """
    Applies the row-wise normalization:
        \mathbf{D}_{out}^{-1} \mathbf{A} 
    """
    row_sum = sparsesum(adj, dim=1)

    return mul(adj, 1 / row_sum.view(-1, 1))


def get_norm_CT(CT, edge_weight):
    return gcn_norm(CT, edge_weight ,add_self_loops=False)


class MyGCNConv(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(MyGCNConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lin = nn.Linear(input_dim, output_dim)
        self.adj_norm = None
        self.C_norm = None
    
    def forward(self, x, edge_index, C = None):
        # if self.adj_norm is None:
        #     row, col = edge_index
        #     num_nodes = x.shape[0]

        #     adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        #     self.adj_norm = get_norm_adj(adj, self.conv_norm)
        if self.C_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]
            C_adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.C_norm = get_norm_CT(C_adj, edge_weight=C._values())
        
        # prop_x = self.adj_norm @ x
        prop_x  = self.C_norm @ x
        trans_x = self.lin(prop_x)
        return trans_x


class CGNN(nn.Module):
    def __init__(self, in_dim, hid_dim, n_cls, num_layers, dropout, jk, norm):
        super(CGNN, self).__init__()
        out_dim = hid_dim if jk else n_cls

        if num_layers == 1:
            self.convs = nn.ModuleList([MyGCNConv(in_dim, out_dim)])
        else:
            self.convs = nn.ModuleList([MyGCNConv(in_dim, hid_dim)])
            for _ in range(num_layers - 2):
                self.convs.append(MyGCNConv(hid_dim, hid_dim))
            self.convs.append(MyGCNConv(hid_dim, out_dim))
        
        if jk is not None:
            jk_in_dim = hid_dim * num_layers if jk == "cat" else hid_dim
            self.jk_lin = nn.Linear(jk_in_dim, n_cls)
            self.jump = JumpingKnowledge(mode=jk, channels=hid_dim, num_layers=num_layers)

        self.num_layers = num_layers
        self.dropout = dropout
        self.jk = jk
        self.norm = norm
    
    def forward(self, x, edge_index, C):
        h = x
        hs = []
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, C)
            if i != len(self.convs) - 1 or self.jk:
                h = F.relu(h)
                h = F.dropout(h, p = self.dropout, training = self.training)
                if self.norm:
                    h = F.normalize(h, p = 2, dim = 1)
            hs += [h]
        if self.jk is not None:
            h = self.jump(hs)
            h = self.jk_lin(h)

        return h