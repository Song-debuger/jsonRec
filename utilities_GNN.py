import os
import json
import dgl
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder


def json_to_dgl_graph(file_path, feature_encoders=None, fit_encoders=False):
    """
    将 JSON 文件中的图数据转换为 DGL 图，并处理节点特征。

    参数：
    file_path (str): JSON 文件的路径。
    feature_encoders (dict): 特征编码器字典，用于类别型特征编码。
    fit_encoders (bool): 是否拟合编码器。

    返回：
    tuple: 包含 DGL 图和特征编码器字典的元组。
    """
    with open(file_path, 'r') as f:
        graph_data = json.load(f)
    G = dgl.graph(([], []))  # 初始化空图
    node_features = []
    feature_keys = None

    if feature_encoders is None:
        feature_encoders = {}

    for node in graph_data["nodes"]:
        G.add_nodes(1)  # 添加节点
        node_feature = []

        if feature_keys is None:
            feature_keys = list(node.keys())

        for key in feature_keys:
            value = node.get(key, None)
            if key not in feature_encoders:
                if isinstance(value, str):
                    feature_encoders[key] = LabelEncoder()
                    if fit_encoders:
                        feature_encoders[key].fit([value])
                    else:
                        feature_encoders[key].fit([])
                else:
                    feature_encoders[key] = None

            if feature_encoders[key] is not None:
                if isinstance(value, str):
                    try:
                        encoded_value = feature_encoders[key].transform([value])[0]
                    except ValueError:
                        encoded_value = -1  # 未见过的标签用 -1 处理
                    node_feature.append(encoded_value)
                else:
                    node_feature.append(value if value is not None else 0)  # 将 None 替换为 0
            else:
                node_feature.append(value if value is not None else 0)  # 将 None 替换为 0

        node_features.append(node_feature)

    src, dst = [], []
    for edge in graph_data["links"]:
        src.append(edge["source"])
        dst.append(edge["target"])
    G.add_edges(src, dst)  # 添加边
    G.ndata['feat'] = torch.tensor(node_features).float()  # 将节点特征转换为张量并添加到图中
    return G, feature_encoders


def mask_graph(graph, mask_ratio=0.2):
    """
    随机移除图中的部分节点，以生成不完整的图。

    参数：
    graph (DGLGraph): 输入的 DGL 图。
    mask_ratio (float): 要移除的节点比例。

    返回：
    DGLGraph: 移除部分节点后的图。
    """
    G = graph.clone()
    num_nodes = G.num_nodes()
    num_mask = int(num_nodes * mask_ratio)
    nodes_to_remove = torch.randperm(num_nodes)[:num_mask].tolist()
    G.remove_nodes(nodes_to_remove)
    return G


def load_dgl_graphs_from_folder(folder_path, fit_encoders=False):
    """
    从文件夹中加载所有 JSON 文件并转换为 DGL 图。

    参数：
    folder_path (str): 文件夹路径。
    fit_encoders (bool): 是否拟合编码器。

    返回：
    tuple: 包含图列表、文件名列表和特征编码器的元组。
    """
    graphs = []
    file_names = []
    feature_encoders = None
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            graph, feature_encoders = json_to_dgl_graph(file_path, feature_encoders, fit_encoders=fit_encoders)
            graphs.append(graph)
            file_names.append(file_name)
    return graphs, file_names, feature_encoders
