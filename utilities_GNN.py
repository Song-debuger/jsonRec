import os
import json
import dgl
import torch
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from tqdm import tqdm
import pickle

def save_data(file_path, data):
    """
    保存数据到指定文件路径。如果目录不存在，创建目录。

    参数：
    file_path (str): 要保存数据的文件路径。
    data (any): 要保存的数据。
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_data(file_path):
    """
    从指定文件路径加载数据。

    参数：
    file_path (str): 要加载数据的文件路径。

    返回：
    any: 加载的数据。
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def normalize_features(features, method="minmax"):
    """
    归一化节点特征。

    参数：
    features (torch.Tensor): 节点特征张量。
    method (str): 归一化方法，可以是 "minmax" 或 "standard"。

    返回：
    torch.Tensor: 归一化后的节点特征张量。
    """
    features = features.numpy()
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid normalization method")

    features = scaler.fit_transform(features)
    return torch.tensor(features)

def json_to_dgl_graph(file_path, feature_encoders=None, fit_encoders=False):
    """
    将 JSON 文件中的图数据转换为 DGL 图，并处理节点和边特征。

    参数：
    file_path (str): JSON 文件的路径。
    feature_encoders (dict): 特征编码器字典，用于类别型特征编码。
    fit_encoders (bool): 是否拟合编码器。

    返回：
    tuple: 包含 DGL 图和特征编码器字典的元组。
    """
    with open(file_path, 'r') as f:
        graph_data = json.load(f)

    # 初始化空图
    G = dgl.graph(([], []))
    node_features = []
    feature_keys = None

    if feature_encoders is None:
        feature_encoders = {}

    # 处理节点
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
                        if fit_encoders:
                            feature_encoders[key].fit(feature_encoders[key].classes_.tolist() + [value])
                            encoded_value = feature_encoders[key].transform([value])[0]
                        else:
                            encoded_value = -1  # 未见过的标签用 -1 处理
                    node_feature.append(encoded_value)
                else:
                    node_feature.append(value if value is not None else 0)  # 将 None 替换为 0
            else:
                node_feature.append(value if value is not None else 0)  # 将 None 替换为 0

        node_features.append(node_feature)

    # 处理边
    src, dst = [], []
    for edge in graph_data["links"]:
        src.append(edge["source"])
        dst.append(edge["target"])
    G.add_edges(src, dst)  # 添加边

    # 将节点特征转换为张量并添加到图中
    G.ndata['feat'] = torch.tensor(node_features).float()

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
    nodes_to_remove = set(torch.randperm(num_nodes)[:num_mask].tolist())

    new_src, new_dst = [], []

    src, dst = G.edges()
    for s, d in zip(src.tolist(), dst.tolist()):
        if s in nodes_to_remove or d in nodes_to_remove:
            continue
        new_src.append(s)
        new_dst.append(d)

    new_graph = dgl.graph((new_src, new_dst), num_nodes=num_nodes)
    new_graph.ndata['feat'] = G.ndata['feat']
    new_graph.remove_nodes(list(nodes_to_remove))

    return new_graph

def load_dgl_graphs_from_folder(folder_path, fit_encoders=False):
    """
    从文件夹中加载 DGL 图并处理节点特征。

    参数：
    folder_path (str): 包含 JSON 图文件的文件夹路径。
    fit_encoders (bool): 是否拟合特征编码器。

    返回：
    tuple: 包含 DGL 图列表、文件名列表和特征编码器字典的元组。
    """
    graphs = []
    file_names = []
    feature_encoders = {}

    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    for json_file in tqdm(json_files, desc="Loading graphs"):
        file_path = os.path.join(folder_path, json_file)
        try:
            graph, feature_encoders = json_to_dgl_graph(file_path, feature_encoders, fit_encoders)
            graphs.append(graph)
            file_names.append(json_file)
        except Exception as e:
            print(f"Error processing file: {file_path}")
            print(e)

    return graphs, file_names, feature_encoders

def normalize_graph_features(graphs, method="minmax"):
    """
    对图的节点特征进行归一化。

    参数：
    graphs (list): 图列表。
    method (str): 归一化方法，可以是 "minmax" 或 "standard"。

    返回：
    list: 归一化后的图列表。
    """
    for graph in graphs:
        graph.ndata['feat'] = normalize_features(graph.ndata['feat'], method)
    return graphs