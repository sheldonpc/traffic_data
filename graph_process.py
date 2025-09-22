import argparse
import os
import datetime
import time
import matplotlib.pyplot as plt
from torchinfo import summary
import yaml
import json
import sys
from datetime import datetime as lh_datatime
import smtplib
from email.mime.text import MIMEText

import numpy as np
import scipy.sparse as sp
import pickle
import pandas as pd
import torch
import torch.nn as nn

def pre_graph_dict(args):
    """
    预处理图数据，构建邻接矩阵、归一化邻接矩阵和拉普拉斯矩阵，并将结果存储到 args 中。

    参数:
        args: 命令行参数对象，包含以下属性：
            - dataset_graph: 数据集名称，用于确定加载哪个图数据。
            - device: 指定计算设备（如 CPU 或 GPU）。
            - use_lpls: 是否使用拉普拉斯矩阵。

    返回值:
        无。该函数将处理后的图数据以字典形式保存在 args 的以下属性中：
            - A_dict_np: 存储原始图数据的归一化邻接矩阵（numpy 格式）。
            - A_dict: 存储归一化后的邻接矩阵（torch.Tensor 格式并转移到指定设备）。
            - lpls_dict: 存储图的拉普拉斯矩阵（torch.Tensor 格式并转移到指定设备）。
    """

    # 初始化存储图相关数据的字典
    A_dict_np = {}
    A_dict = {}
    lap_dict = {}
    node_dict = {}

    # 定义部分数据集中图节点的数量
    node_dict['PEMS08'], node_dict['PEMS07'], node_dict['PEMS04'], node_dict['PEMS03'], node_dict['England'] = 170, 883, 307, 358, 248

    # 获取当前使用的图数据名称
    data_graph = args.dataset_graph

    # 根据不同数据集加载对应的邻接矩阵或距离矩阵
    if True:
        if data_graph == 'PEMS08' or data_graph == 'PEMS04' or data_graph == 'PEMS07' or data_graph == 'England':
            # 加载标准格式的邻接矩阵与距离矩阵
            A, Distance = get_adjacency_matrix(
                distance_df_filename='../data/' + data_graph + '/' + data_graph + '.csv',
                num_of_vertices=node_dict[data_graph]
            )
        elif data_graph == 'PEMS03':
            # PEMS03 数据集需要额外提供节点 ID 文件
            A, Distance = get_adjacency_matrix(
                distance_df_filename='../data/' + data_graph + '/' + data_graph + '.csv',
                num_of_vertices=node_dict[data_graph],
                id_filename='../data/' + data_graph + '/' + data_graph + '.txt'
            )
        elif data_graph == 'PEMS07M':
            # 加载带权重的邻接矩阵，并添加自环
            A = weight_matrix('../data/' + data_graph + '/' + data_graph + '.csv').astype(np.float32)
            A = A + np.eye(A.shape[0])
        elif data_graph == 'NYC_BIKE':
            # 从 CSV 文件中读取邻接矩阵
            A = pd.read_csv('../data/' + data_graph + '/' + data_graph + '.csv', header=None).values.astype(np.float32)
        elif data_graph == 'chengdu_didi':
            # 从 .npy 文件中读取邻接矩阵
            A = np.load('../data/' + data_graph + '/' + 'matrix.npy').astype(np.float32)
        elif data_graph == 'CA_District5':
            # 从 .npy 文件中读取邻接矩阵
            A = np.load('../data/' + data_graph + '/' + data_graph + '.npy').astype(np.float32)
        elif data_graph == 'METRLA':
            # 从 pickle 文件中加载邻接矩阵
            A = load_adj2('../data/METRLA/adj_mx_la.pkl', num_nodes=207).numpy().astype(np.float32)
        elif data_graph == 'PEMSBAY':
            # 从 pickle 文件中加载邻接矩阵并提取第一个元素
            A, _ = load_adj2('../data/PEMSBAY/adj_mx_bay.pkl', num_nodes=325)
            A = A[0].astype(np.float32)

        # 计算图的拉普拉斯矩阵
        lpls = cal_lape(A.copy())
        lpls = torch.FloatTensor(lpls).to(args.device)

        # 如果不使用拉普拉斯矩阵，则对其进行 Xavier 初始化
        if not args.use_lpls:
            nn.init.xavier_uniform_(lpls)

        # 将拉普拉斯矩阵存入字典
        lap_dict[data_graph] = lpls

        # 对邻接矩阵进行归一化处理
        A = get_normalized_adj(A)

        # 存储 numpy 格式的归一化邻接矩阵
        A_dict_np[data_graph] = A

        # 转换为 PyTorch Tensor 并转移到指定设备
        A = torch.FloatTensor(A).to(args.device)
        A_dict[data_graph] = A

    # 将处理好的图数据保存到 args 中
    args.A_dict_np = A_dict_np
    args.A_dict = A_dict
    args.lpls_dict = lap_dict


def load_adj2(pkl_filename,num_nodes):
    """
    加载邻接矩阵并进行预处理

    该函数从pickle文件中加载传感器ID和邻接矩阵，然后对邻接矩阵进行处理，
    减去单位矩阵以去除自连接关系

    参数:
        pkl_filename (str): 包含传感器ID和邻接矩阵的pickle文件路径
        num_nodes (int): 图中节点的数量

    返回:
        torch.Tensor: 处理后的邻接矩阵，类型为torch张量
    """
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    # 从邻接矩阵中减去单位矩阵，去除节点的自连接关系
    adj = torch.tensor(adj) - torch.eye(num_nodes)
    return adj


def load_adj(file_path, adj_type):
    """
    加载邻接矩阵并根据指定类型进行处理

    参数:
        file_path (str): 邻接矩阵文件路径
        adj_type (str): 邻接矩阵处理类型，可选值包括:
            - "scalap": 缩放拉普拉斯矩阵
            - "normlap": 对称归一化拉普拉斯矩阵
            - "symnadj": 对称消息传递邻接矩阵
            - "transition": 转移矩阵
            - "doubletransition": 双向转移矩阵
            - "identity": 单位矩阵
            - "original": 原始邻接矩阵

    返回:
        tuple: (adj, adj_mx)
            - adj: 处理后的邻接矩阵列表
            - adj_mx: 原始邻接矩阵
    """
    try:
        # METR和PEMS_BAY数据集的加载方式
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(file_path)
    except:
        # PEMS04数据集的加载方式
        adj_mx = load_pickle(file_path)

    # 根据不同的邻接矩阵类型进行相应处理
    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "normlap":
        adj = [calculate_symmetric_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "symnadj":
        adj = [symmetric_message_passing_adj(adj_mx).astype(np.float32).todense()]
    elif adj_type == "transition":
        adj = [transition_matrix(adj_mx).T]
    elif adj_type == "doubletransition":
        adj = [transition_matrix(adj_mx).T, transition_matrix(adj_mx.T).T]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32).todense()]
    elif adj_type == 'original':
        adj = adj_mx
    else:
        error = 0
        assert error, "adj type not defined"
    return adj, adj_mx


import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import torch

def check_nan_inf(tensor, raise_ex=True):
    """
    检查张量中是否包含NaN或无穷大值

    参数:
        tensor: 要检查的张量
        raise_ex: 布尔值，当为True时，如果发现NaN或无穷大值则抛出异常，默认为True

    返回:
        tuple: 包含两个元素的元组
            - dict: 包含"nan"和"inf"键的字典，对应值为布尔值表示是否存在NaN或无穷大
            - bool: 表示是否存在NaN或无穷大值的布尔值
    """
    # 检查张量中是否存在NaN值
    nan = torch.any(torch.isnan(tensor))
    # 检查张量中是否存在无穷大值
    inf = torch.any(torch.isinf(tensor))
    # 根据参数决定是否抛出异常
    if raise_ex and (nan or inf):
        raise Exception({"nan":nan, "inf":inf})
    return {"nan":nan, "inf":inf}, nan or inf


def remove_nan_inf(tensor):
    """
    移除张量中的NaN（非数字）和Inf（无穷大）值，将其替换为0

    参数:
        tensor: 输入的torch张量，可能包含NaN或Inf值

    返回:
        处理后的torch张量，其中所有的NaN和Inf值都被替换为0
    """
    # 将张量中的NaN值替换为0
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    # 将张量中的Inf值替换为0
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor


def calculate_symmetric_normalized_laplacian(adj):
    """
    计算对称归一化拉普拉斯矩阵

    对称归一化拉普拉斯矩阵定义为: L_sym = I - D^{-1/2} * A * D^{-1/2}
   其中I是单位矩阵，D是度矩阵，A是邻接矩阵

    参数:
        adj: 邻接矩阵，可以是numpy数组或scipy稀疏矩阵

    返回:
        symmetric_normalized_laplacian: 对称归一化拉普拉斯矩阵，scipy coo_matrix格式
    """
    adj                                 = sp.coo_matrix(adj)
    D                                   = np.array(adj.sum(1))
    D_inv_sqrt = np.power(D, -0.5).flatten()    # 计算度矩阵对角线元素的负二分之一次方
    D_inv_sqrt[np.isinf(D_inv_sqrt)]    = 0.     # 处理无穷大值，将其设为0
    matrix_D_inv_sqrt                   = sp.diags(D_inv_sqrt)   # 构造度矩阵的负二分之一次方对角矩阵
    # 计算对称归一化拉普拉斯矩阵: L_sym = I - D^{-1/2} * A * D^{-1/2}
    symmetric_normalized_laplacian      = sp.eye(adj.shape[0]) - matrix_D_inv_sqrt.dot(adj).dot(matrix_D_inv_sqrt).tocoo()
    return symmetric_normalized_laplacian


def calculate_scaled_laplacian(adj, lambda_max=2, undirected=True):
    """
    计算缩放的拉普拉斯矩阵

    该函数首先根据邻接矩阵计算对称归一化的拉普拉斯矩阵，然后对其进行缩放处理，
    使得特征值范围被映射到[-1, 1]区间，这在图神经网络中常用于提升收敛性。

    参数:
        adj: 邻接矩阵，表示图的连接关系
        lambda_max: 最大特征值，用于缩放因子计算。如果为None，则会自动计算
        undirected: 布尔值，指示图是否为无向图。如果是，则会对邻接矩阵进行对称化处理

    返回值:
        缩放后的拉普拉斯矩阵，其特征值范围被映射到[-1, 1]区间
    """
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    L       = calculate_symmetric_normalized_laplacian(adj)
    if lambda_max is None:  # manually cal the max lambda
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L       = sp.csr_matrix(L)
    M, _    = L.shape
    I       = sp.identity(M, format='csr', dtype=L.dtype)
    L_res   = (2 / lambda_max * L) - I
    return L_res

def symmetric_message_passing_adj(adj):
    """
    计算对称消息传递的邻接矩阵

    该函数实现了一种对称归一化的消息传递机制，通过对邻接矩阵进行对称归一化处理，
    使得消息在图中的传播更加平衡和稳定。

    参数:
        adj: 邻接矩阵，要求已经添加了自环连接

    返回值:
        对称归一化后的消息传递邻接矩阵
    """
    print("calculating the renormalized message passing adj, please ensure that self-loop has added to adj.")
    adj         = sp.coo_matrix(adj)
    rowsum      = np.array(adj.sum(1))
    d_inv_sqrt  = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt  = sp.diags(d_inv_sqrt)
    # 执行对称归一化：D^(-1/2) * A * D^(-1/2)
    mp_adj          = d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()
    return mp_adj

def transition_matrix(adj):
    """
    计算转移矩阵（随机游走矩阵）

    该函数将邻接矩阵转换为转移矩阵，其中每个元素表示从一个节点转移到另一个节点的概率。
    这在图上的随机游走和PageRank等算法中非常有用。

    参数:
        adj: 邻接矩阵

    返回值:
        转移矩阵，每行的和为1，表示从该节点出发的转移概率分布
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    # 计算转移矩阵：P = D^(-1) * A
    P = d_mat.dot(adj).astype(np.float32).todense()
    return P



def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    根据距离文件生成邻接矩阵和距离矩阵。

    该函数支持两种格式的距离文件：.npy格式的预处理矩阵文件，或CSV格式的节点距离列表。
    如果提供了id_filename，则会根据文件中的节点ID映射关系重新排列矩阵索引。

    参数:
        distance_df_filename (str): 距离数据文件路径，支持.npy或.csv格式
        num_of_vertices (int): 图中顶点的数量
        id_filename (str, optional): 节点ID映射文件路径，默认为None

    返回:
        tuple: 包含两个元素的元组
            - A (numpy.ndarray): 邻接矩阵，形状为(num_of_vertices, num_of_vertices)
            - distaneA (numpy.ndarray): 距离矩阵，形状为(num_of_vertices, num_of_vertices)
    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:
            # 读取节点ID到索引的映射关系
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            # 根据映射关系构建邻接矩阵和距离矩阵
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:
            # 直接使用节点索引构建邻接矩阵和距离矩阵
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA

def load_pickle(pickle_file):
    '''
    加载pickle格式的文件数据。

    该函数尝试以不同编码方式加载pickle文件，处理可能的Unicode解码错误。

    参数:
        pickle_file (str): pickle文件路径

    返回:
        object: 从pickle文件中加载的数据对象

    异常:
        UnicodeDecodeError: 当默认编码无法解码时，尝试使用latin1编码
        Exception: 其他加载异常会打印错误信息并重新抛出
    '''
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data




import numpy as np
import pandas as pd
import scipy.sparse as sp


def calculate_scaled_laplacian(adj):
    """
    计算缩放的拉普拉斯矩阵（Scaled Laplacian），用于图卷积网络中的谱域处理。

    参数:
        adj (np.ndarray): 图的邻接矩阵，形状为 (n, n)，其中 n 是节点数。

    返回:
        np.ndarray: 缩放后的拉普拉斯矩阵，形状为 (n, n)。
    """
    n = adj.shape[0]
    d = np.sum(adj, axis=1)  # 计算度向量 D
    lap = np.diag(d) - adj     # 构造未归一化的拉普拉斯矩阵 L = D - A

    # 对拉普拉斯矩阵进行对称归一化
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                lap[i, j] /= np.sqrt(d[i] * d[j])

    # 处理可能存在的无穷大或 NaN 值
    lap[np.isinf(lap)] = 0
    lap[np.isnan(lap)] = 0

    # 计算最大特征值并进行缩放
    lam = np.linalg.eigvals(lap).max().real
    return 2 * lap / lam - np.eye(n)


def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    """
    根据输入文件路径读取邻接矩阵，并根据参数决定是否进行权重缩放。

    参数:
        file_path (str): 邻接矩阵文件路径。
        sigma2 (float): 用于计算权重的高斯核参数，默认为 0.1。
        epsilon (float): 权重阈值，小于该值的边将被置零，默认为 0.5。
        scaling (bool): 是否进行权重缩放，默认为 True。

    返回:
        np.ndarray: 处理后的邻接矩阵。
    """
    try:
        W = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    # 检查是否为 0/1 矩阵，若是则关闭缩放
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 10000.
        W2, WMASK = W * W, np.ones([n, n]) - np.identity(n)
        # 使用高斯核计算权重矩阵，参考论文中的 Eq.10
        A = np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * WMASK
        return A
    else:
        return W


def first_approx(W, n):
    """
    对邻接矩阵进行一阶近似处理，用于图卷积网络中。

    参数:
        W (np.ndarray): 输入的邻接矩阵。
        n (int): 节点数量。

    返回:
        np.matrix: 一阶近似处理后的矩阵。
    """
    A = W + np.identity(n)  # 添加自连接
    d = np.sum(A, axis=1)
    sinvD = np.sqrt(np.mat(np.diag(d)).I)
    # 参考论文中的 Eq.5 进行矩阵变换
    return np.mat(np.identity(n) + sinvD * A * sinvD)


def get_normalized_adj(A):
    """
    对邻接矩阵进行对称归一化处理。

    参数:
        A (np.ndarray): 输入的邻接矩阵。

    返回:
        np.ndarray: 归一化后的邻接矩阵。
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))  # 添加自连接
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # 防止除以零导致无穷大
    diag = np.reciprocal(np.sqrt(D))
    # 执行对称归一化 A_wave = D^{-1/2} * A * D^{-1/2}
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def asym_adj(adj):
    """
    对邻接矩阵进行非对称归一化处理（行归一化）。

    参数:
        adj (np.ndarray): 输入的邻接矩阵。

    返回:
        np.ndarray: 非对称归一化后的邻接矩阵。
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.  # 处理度为0的情况
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def idEncode(x, y, col):
    """
    将二维坐标编码为一维索引。

    参数:
        x (int): 行索引。
        y (int): 列索引。
        col (int): 总列数。

    返回:
        int: 编码后的一维索引。
    """
    return x * col + y


def constructGraph(row, col):
    """
    构造一个二维网格图的邻接矩阵，每个节点与其8个方向的邻居相连。

    参数:
        row (int): 网格行数。
        col (int): 网格列数。

    返回:
        np.ndarray: 构造的邻接矩阵，形状为 (row*col, row*col)。
    """
    # 8个方向的偏移量（包括自身）
    mx = [-1, 0, 1, 0, -1, -1, 1, 1, 0]
    my = [0, -1, 0, 1, -1, 1, -1, 1, 0]

    areaNum = row * col

    def illegal(x, y):
        """判断坐标是否越界"""
        return x < 0 or y < 0 or x >= row or y >= col

    W = np.zeros((areaNum, areaNum))
    for i in range(row):
        for j in range(col):
            n1 = idEncode(i, j, col)
            for k in range(len(mx)):
                temx = i + mx[k]
                temy = j + my[k]
                if illegal(temx, temy):
                    continue
                n2 = idEncode(temx, temy, col)
                W[n1, n2] = 1
    return W

def calculate_normalized_laplacian(adj):
    """
    计算归一化拉普拉斯矩阵

    参数:
        adj: 邻接矩阵，可以是稀疏矩阵或密集矩阵

    返回:
        tuple: (normalized_laplacian, isolated_point_num)
            - normalized_laplacian: 归一化拉普拉斯矩阵（稀疏矩阵格式）
            - isolated_point_num: 孤立点的数量
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1)).flatten()
    isolated_point_num = np.sum(np.where(d == 0, 1, 0))
    print(f"Number of isolated points: {isolated_point_num}")

    # 添加一个小的正数以避免除以零
    epsilon = 1e-10
    d = np.where(d == 0, epsilon, d)

    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian, isolated_point_num

def cal_lape(adj_mx):
    """
    计算拉普拉斯位置编码

    参数:
        adj_mx: 邻接矩阵

    返回:
        numpy.ndarray: 拉普拉斯位置编码矩阵，形状为(n_nodes, lape_dim)
    """
    lape_dim = 32
    L, isolated_point_num = calculate_normalized_laplacian(adj_mx)

    # 计算拉普拉斯矩阵的特征值和特征向量
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    # 提取除孤立点外的前lape_dim个特征向量作为位置编码
    laplacian_pe = EigVec[:, isolated_point_num + 1: lape_dim + isolated_point_num + 1]
    return laplacian_pe
