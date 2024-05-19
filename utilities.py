import os
import json
import random
import networkx as nx
import matplotlib.pyplot as plt
import dgl
import torch


# 从 JSON 文件中读取图数据并转换为 NetworkX 图
def json_to_networkx(file_path):
    with open(file_path, 'r') as f:
        graph_data = json.load(f)
    G = nx.MultiDiGraph() if graph_data.get("directed", False) else nx.MultiGraph()
    for node in graph_data["nodes"]:
        node_attributes = {key: str(value) for key, value in node.items()}
        label = "|".join(f"{key}:{value}" for key, value in node_attributes.items())
        G.add_node(node["id"], label=label, **node_attributes)
    for edge in graph_data["links"]:
        G.add_edge(edge["source"], edge["target"])
    return G


# 从文件夹中读取所有图
def load_graphs_from_folder(folder_path):
    graphs = []
    file_names = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            graphs.append(json_to_networkx(file_path))
            file_names.append(file_name)
    return graphs, file_names


# 随机掩码生成不完整的图
def mask_graph(graph, mask_ratio=0.2):
    G = graph.copy()
    num_nodes = len(G.nodes)
    num_mask = int(num_nodes * mask_ratio)
    nodes_to_remove = random.sample(list(G.nodes), num_mask)

    for node in nodes_to_remove:
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 1:
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    G.add_edge(neighbors[i], neighbors[j])
        G.remove_node(node)

    return G

# # 可视化推荐结果
# def visualize_network(recommendations, ground_truth, results_folder):
#     plt.figure(figsize=(12, 12))
#     G = nx.DiGraph()
#
#     for test_graph, recommended_files in recommendations.items():
#         G.add_node(test_graph, color='blue')
#         for recommended_file in recommended_files:
#             G.add_node(recommended_file, color='green')
#             G.add_edge(test_graph, recommended_file, color='gray')
#
#     pos = nx.spring_layout(G)
#     colors = nx.get_node_attributes(G, 'color').values()
#     nx.draw(G, pos, with_labels=True, node_color=colors, node_size=3000, font_size=10, font_color='white')
#     edges = G.edges()
#     colors = [G[u][v]['color'] for u, v in edges]
#     nx.draw_networkx_edges(G, pos, edge_color=colors)
#     plt.title('Recommendation Network')
#     plt.savefig(os.path.join(results_folder, 'recommendation_network.png'))
#     plt.show()

def visualize_network(recommendations, results_folder, top_n=3):
    plt.figure(figsize=(12, 12))
    G = nx.DiGraph()

    for test_graph, recommended_files in recommendations.items():
        G.add_node(test_graph, color='blue', label='Test Graph')
        for recommended_file in recommended_files[:top_n]:
            G.add_node(recommended_file, color='green', label='Recommended Graph')
            G.add_edge(test_graph, recommended_file, color='gray')

    pos = nx.spring_layout(G)
    colors = [G.nodes[node]['color'] for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=3000, font_size=10, font_color='white')
    edges = G.edges()
    edge_colors = [G[u][v]['color'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors)
    plt.title('Recommendation Network (Top 3)')
    plt.savefig(os.path.join(results_folder, 'recommendation_network.png'))
    plt.show()

# def json_to_dgl_graph(file_path):
#     with open(file_path, 'r') as f:
#         graph_data = json.load(f)
#     G = dgl.graph(([], []))
#     node_features = []
#     for node in graph_data["nodes"]:
#         G.add_nodes(1)
#         node_features.append(node['id'])
#     src, dst = [], []
#     for edge in graph_data["links"]:
#         src.append(edge["source"])
#         dst.append(edge["target"])
#     G.add_edges(src, dst)
#     G.ndata['feat'] = torch.tensor(node_features).float().unsqueeze(1)
#     return G
#
# def load_dgl_graphs_from_folder(folder_path):
#     graphs = []
#     file_names = []
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith('.json'):
#             file_path = os.path.join(folder_path, file_name)
#             graphs.append(json_to_dgl_graph(file_path))
#             file_names.append(file_name)
#     return graphs, file_names