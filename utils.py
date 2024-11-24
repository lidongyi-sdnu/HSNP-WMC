import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from normalization import normalized_adjacency, row_normalize


def encode_onehot(labels):
    classes = set(labels)  # set()函数创建一个无序不重复元素集,标签直接简化成了7类（Cora数据集）
    # enumerate()函数生成序列，带有索引i和值c
    # 这一句将string类型的label变为int类型的label，建立映射关系
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    # map()会根据提供的函数对指定序列做映射
    # 返回int类型的label（2708，7）
    return labels_onehot


def preprocess_citation(adj, features):

    adj = normalized_adjacency(adj)
    features = row_normalize(features)
    return adj, features


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(dataset_str="cora"):
    print("Loading {} dataset...".format(dataset_str))
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    # 索引乱序，读取测试节点的索引
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    # 将索引从小到大排序，就是从1708到2707
    test_idx_range = np.sort(test_idx_reorder)
    # 每个数据集中出现的一些脱离模型的点叫做孤立点。处理孤立点的方式是找到test.index中没有对应的索引，
    # 一共有15个，把这些点当作特征全为0的节点加入到测试集中，并且对应的标签在ty中
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    # 将allx和tx叠起来并转化成lil格式的feature，即输入一张整图
    features = sp.vstack((allx, tx)).tolil()
    # vstack:将数组堆叠成一列
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # 邻接矩阵也是lil的，并且shape为（2708,2708）
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # from_dict_of_lists(graph)图转化为字典
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist() # （1708,1709，...,2707）
    idx_train = range(len(y)) # range（0,140）
    idx_val = range(len(y), len(y)+500) #（140,640）

    adj, features = preprocess_citation(adj, features)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    # adj = sp.coo_matrix(adj, shape=(2708,2708), dtype=np.float32)
    #adj = torch.FloatTensor(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    adj = adj.to_dense()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)