import yaml
import torch
import os.path as osp
import numpy as np
import random
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import add_self_loops, to_dense_adj, to_undirected
from torch_scatter import scatter_add
import scipy

def set_random_seed(seed):		
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def config2args(args, dataset_name):
    config_path = osp.join('config', f'{args.model}.yaml')
    set_random_seed(1234)
    with open(config_path, "r") as file:
        hyperparams = yaml.safe_load(file)
    for name, value in hyperparams[dataset_name].items():
        if hasattr(args, name):
            setattr(args, name, value)
        else:
            setattr(args, name, value)
    return args

def config2args_lp(args, dataset_name):
    config_path = osp.join('config_lp', f'{args.model}.yaml')
    with open(config_path, "r") as file:
        hyperparams = yaml.safe_load(file)
    for name, value in hyperparams[dataset_name].items():
        if hasattr(args, name):
            setattr(args, name, value)
        else:
            setattr(args, name, value)
    return args

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def path_graph(node_squence, k: int, n_nodes: int):
    adj_dict = {node : [] for node in node_squence}
    if k == 1:
        # k-regular path graph is a undirected path graph, but the head and tail nodes links to 2 nodes
        for i, node in enumerate(node_squence):
            if i == 0:
                adj_dict[node].append(node_squence[i+1])
                adj_dict[node].append(node_squence[i+2])
            elif i == n_nodes-1:
                adj_dict[node].append(node_squence[i-1])
                adj_dict[node].append(node_squence[i-2])
            else:
                adj_dict[node].extend([node_squence[i-1], node_squence[i+1]])
    elif k>=2:
        for i, node in enumerate(node_squence): 
            if i < k:
                adj_dict[node].extend(node_squence[:i])    # k = 5 i=2   0,1
                adj_dict[node].extend(node_squence[i+1:i+k])  # 3,4,5  
            elif i > n_nodes-k-1:
                adj_dict[node].extend(node_squence[i:])
                adj_dict[node].extend(node_squence[i-k:i])
            else:
                adj_dict[node].extend(node_squence[i-k:i])
                adj_dict[node].extend(node_squence[i+1:i+k+1])
    
    g = nx.from_dict_of_lists(adj_dict,create_using=nx.DiGraph)
    return adj_dict, g

def get_CT(args, data):
    num_nodes = data.num_nodes
    gen_data = data.clone()
    edge_index = data.edge_index
    x = data.x
    x_avg = torch.mean(x, dim = 0)
    x_avg_norm = F.normalize(x_avg, p = 2, dim=-1)
    x_norm = F.normalize(x, p = 2, dim=-1)
    sim   = torch.matmul(x_norm, x_avg_norm)
    node_sort = sim.sort()[1].numpy()
    adj_dict, fpg = path_graph(node_sort,k = 1, n_nodes=num_nodes)
    edges = np.array(list(fpg.edges))
    edge_list = torch.from_numpy(edges[np.argsort(edges[:, 0])].transpose())
    edge_list = to_undirected(edge_list)

    new_edge_index = torch.cat((edge_index, edge_list), dim = -1).long()
    gen_data.edge_index = new_edge_index    
    gen_data.coalesce()
    new_edge_weight  = torch.ones((gen_data.edge_index.size(1),), dtype=edge_index.dtype)

    gen_data.edge_weight = new_edge_weight

    fill_values = 1
    edge_index, edge_weight = add_self_loops(gen_data.edge_index, gen_data.edge_weight, fill_values, num_nodes)    

    # compute transition matrid D^-1A
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes).float()
    deg_inv = deg.pow(-1)  # D^-1
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight   # * D^-1 A
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense()

    # compute stationary distribution
    # eig_value, left_vector = scipy.linalg.eig(p_dense.numpy(),left=True,right=False)
    # eig_value = torch.from_numpy(eig_value.real)
    # left_vector = torch.from_numpy(left_vector.real) 
    # dom_val, ind = eig_value.sort(descending=True)
    # pi = left_vector[:,ind[0]]
    # pi = pi/pi.sum()
    p_dense = p_dense.numpy()
    p_dense_t = p_dense.T
    eigenvals, eigenvects = np.linalg.eig(p_dense_t)
    close_to_1_idx = np.isclose(eigenvals,1)
    target_eigenvect = eigenvects[:,close_to_1_idx]
    target_eigenvect = target_eigenvect[:,0]
    pi = target_eigenvect / sum(target_eigenvect) 
    pi[pi == float('inf')] = 0
    pi[pi <= 0] = 0
    
    # Compute R
    pi = pi / max(pi)
    pi = pi.real
    Pi = np.diag(pi)
    deg_inv_np = deg_inv.numpy()
    DiLap = Pi @ (deg_inv_np - p_dense)

    pi_pow_0 = np.power(np.where(pi==0, 1, pi), -0.5)
    # pi_pow_0[pi<=0] = 0
    Pi_pow = np.diag(pi_pow_0)
    # R = (Pi_pow.dot(DiLap)).dot(Pi_pow) - deg_inv_np + np.eye(num_nodes)
    R = (Pi_pow.dot(DiLap)).dot(Pi_pow) + np.eye(num_nodes)
    R_pinv = np.linalg.pinv(R)
    

    # pi_pow_1 = np.power(pi, 0.5)
    pi_pow_1 = np.power(np.where(pi==0, 1, pi), -1)
    # pi_pow_1[pi<=0] = 0 
    e = np.ones(num_nodes)
    I = np.eye(num_nodes)
    H = (np.outer(e, pi_pow_1)).dot(R_pinv * I) - R_pinv.dot(np.outer(pi_pow_0, pi_pow_0))
    H = H.real
    # print(H)

    C = H + H.T
    # convert distance to similarity

    np.fill_diagonal(C, 0)
    C = C / np.max(C)

    C[C <= 0] = 0

    # neighbor specific CT 
    undir_edge_index = to_undirected(data.edge_index)
    dense_adj = to_dense_adj(undir_edge_index)
    dense_adj = dense_adj.numpy()

    
    
    
    # # CT to similarity 
    # zero_mask = (C <= 0)
    # C_sim = np.exp(-C)
    # C_sim[zero_mask] = 0

    # # similarity filter by adj
    # CT_adj = C_sim * dense_adj
    

    # edge_attr: distance
    CT_adj = C * dense_adj
    
    CT_adj_norm = CT_adj / CT_adj.max(axis=1).reshape(-1,1)
    CT_adj_norm = CT_adj_norm.squeeze()
    # print(CT_adj_norm.shape)
    CT = CT_adj_norm + CT_adj_norm.T
    # print(CT.shape)
    
    np.fill_diagonal(CT, 0)

    rows, cols = np.where(CT != 0)
    values = CT[rows, cols]
    indices = torch.LongTensor([rows, cols])
    # print(data.edge_index.shape)
    values = torch.FloatTensor(values)
    size = torch.Size(CT.shape)

    # sparse_CT = torch.sparse.FloatTensor(indices, values, size)


    # CT to similarity 
    sim_values = max(values) + 0.01 - values
    sim_values = sim_values/ max(sim_values)
    sparse_CT = torch.sparse.FloatTensor(indices, sim_values, size)
    # print(sim_values)
    # print(max(sim_values))
    # print(min(sim_values))
    torch.save(sparse_CT, osp.join('R', f'{args.dataset}.pt'))

    return sparse_CT





