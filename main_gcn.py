import random
import os
import numpy as np
from datetime import datetime
from gcn_model import GCN, train_gcn_model
from utilities_GNN import load_dgl_graphs_from_folder, mask_graph
import torch
from sklearn.metrics.pairwise import cosine_similarity
import json

# 读取数据集
dataset_folder = 'Datasets/jsonGraph'
train_graphs, train_file_names, feature_encoders = load_dgl_graphs_from_folder(dataset_folder, fit_encoders=True)

# 分割训练集和测试集
random.shuffle(train_graphs)
train_split = int(0.8 * len(train_graphs))
test_graphs = train_graphs[train_split:]
train_graphs = train_graphs[:train_split]
test_file_names = train_file_names[train_split:]
train_file_names = train_file_names[:train_split]

# 生成不完整的测试集
incomplete_test_graphs = [mask_graph(g, mask_ratio=0.2) for g in test_graphs]

# 训练 GCN 模型
in_feats = len(train_graphs[0].ndata['feat'][0])  # 输入特征维度
h_feats = 16  # 隐藏层维度

model = train_gcn_model(train_graphs, in_feats, h_feats)

# 评估模型
def get_graph_embeddings(model, graphs):
    """
    获取图的嵌入表示。

    参数：
    model (GCN): 训练好的 GCN 模型。
    graphs (list): 图列表。

    返回：
    np.array: 图的嵌入表示数组。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            logits = model(g, g.ndata['feat'])
            embedding = logits.mean(dim=0).cpu().numpy()
            embeddings.append(embedding)
    return np.array(embeddings)

def evaluate_recommendations(train_embeddings, test_embeddings, train_file_names, test_file_names, ground_truth, top_n=3):
    """
    评估推荐结果。

    参数：
    train_embeddings (np.array): 训练图的嵌入表示。
    test_embeddings (np.array): 测试图的嵌入表示。
    train_file_names (list): 训练图的文件名列表。
    test_file_names (list): 测试图的文件名列表。
    ground_truth (dict): 测试图的 ground truth 字典。
    top_n (int): 推荐的前 N 个结果。

    返回：
    tuple: 包含推荐结果、命中率和 MRR 的元组。
    """
    similarities = cosine_similarity(test_embeddings, train_embeddings)
    recommendations = {}
    hits = 0
    total_reciprocal_rank = 0

    for i, test_file in enumerate(test_file_names):
        most_similar_indices = similarities[i].argsort()[-top_n:][::-1]
        recommended_files = [train_file_names[idx] for idx in most_similar_indices]
        recommendations[test_file] = recommended_files

        # 计算命中率和平均排名倒数
        if test_file in ground_truth:
            true_files = ground_truth[test_file]
            hit = False
            for rank, recommended_file in enumerate(recommended_files, start=1):
                if recommended_file in true_files:
                    hits += 1
                    total_reciprocal_rank += 1 / rank
                    hit = True
                    break
            if not hit:
                total_reciprocal_rank += 0

    hit_rate = hits / len(test_file_names)
    mrr = total_reciprocal_rank / len(test_file_names)
    return recommendations, hit_rate, mrr

# 获取训练集和测试集的图表示
train_embeddings = get_graph_embeddings(model, train_graphs)
test_embeddings = get_graph_embeddings(model, incomplete_test_graphs)

# 创建 ground_truth 字典，假设每个测试图对应一个 ground truth 图
ground_truth = {test_file_names[i]: [test_file_names[i]] for i in range(len(test_graphs))}

# 评估推荐结果
recommendations, hit_rate, mrr = evaluate_recommendations(train_embeddings, test_embeddings, train_file_names, test_file_names, ground_truth)

# 保存结果到现有的 results 文件夹中
results_folder = 'results'

output_folder = 'results_GNN'

# 保存结果
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_folder = os.path.join(output_folder, timestamp)
os.makedirs(results_folder)

# 保存推荐结果
with open(os.path.join(results_folder, 'recommendations.json'), 'w') as f:
    json.dump(recommendations, f, indent=4)
#
# # 保存推荐结果
# recommendations_file = os.path.join(results_folder, 'recommendations.json')
# with open(recommendations_file, 'w') as f:
#     json.dump(recommendations, f, indent=4)

# 保存命中率和 MRR
# metrics_file = os.path.join(results_folder, 'metrics.json')
metrics = {
    'Hit Rate': hit_rate,
    'MRR': mrr
}

with open(os.path.join(results_folder, 'results.json'), 'w') as f:
    json.dump(metrics, f, indent=4)
# with open(metrics_file, 'w') as f:
#     json.dump(metrics, f, indent=4)

print(f"Hit Rate: {hit_rate}")
print(f"Mean Reciprocal Rank (MRR): {mrr}")
print("Recommendations:", recommendations)
