dataset_folder = 'C:\\Users\\98157\\Desktop\\毕设思路\\jsonRec\\Datasets\\jsonGraph'
import os
import json
import random
import networkx as nx
from grakel import graph_from_networkx, GraphKernel
from sklearn.model_selection import train_test_split
import numpy as np

import os
import json
import random
import networkx as nx
from grakel import graph_from_networkx, GraphKernel
import numpy as np


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


# 推荐 n 个最相似的训练集图给测试集中的每个图
def recommend_for_test(K_test_train, test_file_names, train_file_names, top_n=3):
    recommendations = {}
    for i, similarities in enumerate(K_test_train):
        most_similar_indices = similarities.argsort()[-top_n:][::-1]
        recommended_files = [train_file_names[idx] for idx in most_similar_indices]
        recommendations[test_file_names[i]] = recommended_files
    return recommendations


# 评估推荐系统性能
def evaluate_recommendations(recommendations, ground_truth):
    all_precisions = []
    all_recalls = []
    all_f1s = []
    successful_recommendations = 0

    for test_graph, recommended_files in recommendations.items():
        if test_graph in ground_truth:
            true_files = ground_truth[test_graph]
            true_set = set(true_files)
            recommended_set = set(recommended_files)

            # 检查推荐结果中是否包含任何一个 ground truth 图
            true_positives = 1 if true_set & recommended_set else 0
            if true_positives > 0:
                successful_recommendations += 1

            precision = true_positives / len(recommended_set) if recommended_set else 0
            recall = true_positives / len(true_set) if true_set else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
        else:
            print(f"Warning: {test_graph} not in ground truth")

    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1s)
    hit_rate = successful_recommendations / len(recommendations)

    return avg_precision, avg_recall, avg_f1, hit_rate


# 读取数据集
# dataset_folder = 'dataset'
dataset_folder = 'C:\\Users\\98157\\Desktop\\毕设思路\\jsonRec\\Datasets\\jsonGraph'
graphs, file_names = load_graphs_from_folder(dataset_folder)

# 检查数据集大小
print(f"Total graphs: {len(graphs)}")
print(f"Total file names: {len(file_names)}")

# 从训练集中随机选择一部分作为 ground truth
ground_truth_indices = random.sample(range(len(graphs)), int(len(graphs) * 0.2))
ground_truth_graphs = [graphs[i] for i in ground_truth_indices]
ground_truth_file_names = [file_names[i] for i in ground_truth_indices]

# 检查 ground truth 集合
print(f"Ground truth graphs: {len(ground_truth_graphs)}")
print(f"Ground truth file names: {len(ground_truth_file_names)}")

# 构建 ground truth 集合
ground_truth = {file_name: [file_name] for file_name in ground_truth_file_names}

# 生成不同 mask 程度的不完整图作为测试集
mask_ratios = [0.3, 0.5, 0.7]
incomplete_test_graphs = []
incomplete_test_file_names = []

for graph, file_name in zip(ground_truth_graphs, ground_truth_file_names):
    for mask_ratio in mask_ratios:
        incomplete_graph = mask_graph(graph, mask_ratio)
        incomplete_test_graphs.append(incomplete_graph)
        incomplete_test_file_names.append(f"{file_name}_mask_{mask_ratio}")

# 检查生成的不完整图集合
print(f"Incomplete test graphs: {len(incomplete_test_graphs)}")
print(f"Incomplete test file names: {len(incomplete_test_file_names)}")

# 将 NetworkX 图转换为 grakel 库使用的图格式
G_train = list(graph_from_networkx(graphs, node_labels_tag='label'))
G_test = list(graph_from_networkx(incomplete_test_graphs, node_labels_tag='label'))

# 使用 grakel 计算图核
gk = GraphKernel(kernel={"name": "weisfeiler_lehman", "n_iter": 5}, normalize=True)
K_train = gk.fit_transform(G_train)  # 训练集图核矩阵
K_test_train = gk.transform(G_test)  # 测试集与训练集之间的图核相似度矩阵

# 推荐
n_recommendations = 4  # 推荐的相似图数量
recommendations = recommend_for_test(K_test_train, incomplete_test_file_names, file_names, top_n=n_recommendations)

# 更新 ground truth 集合以包含所有生成的不完整图
for file_name in ground_truth_file_names:
    for mask_ratio in mask_ratios:
        ground_truth[f"{file_name}_mask_{mask_ratio}"] = [file_name]

# 调试输出
print("Recommendations:", recommendations)
print("Ground truth:", ground_truth)

# 评估推荐
precision, recall, f1, hit_rate = evaluate_recommendations(recommendations, ground_truth)

# 输出评估结果
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Hit Rate: {hit_rate}")


