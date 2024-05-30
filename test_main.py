import random
import os
import numpy as np
from datetime import datetime
import torch
from sklearn.metrics.pairwise import cosine_similarity
import json
from tqdm import tqdm
from gcn_model import train_gcn_model
from utilities_GNN import (
    load_dgl_graphs_from_folder,
    mask_graph,
    load_data,
    save_data,
    normalize_graph_features
)

# 设置路径和文件名
dataset_folder = 'Datasets/test'
data_file = 'Datasets/cache/test_saved_data.pkl'
normalized_data_file = 'Datasets/cache/test_saved_normalized_data.pkl'
output_folder = 'results_GNN_test'

def load_and_process_data(dataset_folder, data_file):
    if os.path.exists(data_file):
        print(f"Loading data from {data_file}...")
        train_graphs, train_file_names, feature_encoders = load_data(data_file)
    else:
        print(f"{data_file} not found. Loading data from {dataset_folder} and creating {data_file}...")
        train_graphs, train_file_names, feature_encoders = load_dgl_graphs_from_folder(dataset_folder, fit_encoders=True)
        save_data(data_file, (train_graphs, train_file_names, feature_encoders))
    return train_graphs, train_file_names, feature_encoders

def normalize_and_save_data(graphs, file_names, feature_encoders, normalization_method="minmax"):
    graphs = normalize_graph_features(graphs, method=normalization_method)
    save_data(normalized_data_file, (graphs, file_names, feature_encoders))
    return graphs, file_names, feature_encoders

def get_graph_embeddings(model, graphs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for g in tqdm(graphs, desc="Computing embeddings"):
            g = g.to(device)
            logits = model(g, g.ndata['feat'])
            embedding = logits.mean(dim=0).cpu().numpy()
            embeddings.append(embedding)
    return np.array(embeddings)

def evaluate_recommendations(train_embeddings, test_embeddings, train_file_names, test_file_names, ground_truth, top_n=3):
    similarities = cosine_similarity(test_embeddings, train_embeddings)
    recommendations = {}
    hits = 0
    total_reciprocal_rank = 0

    for i, test_file in enumerate(test_file_names):
        most_similar_indices = similarities[i].argsort()[-top_n:][::-1]
        recommended_files = [train_file_names[idx] for idx in most_similar_indices]
        recommendations[test_file] = recommended_files

        # 计算命中率和平均排名倒数
        true_files = ground_truth.get(test_file.split('_mask_')[0], [])
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

# 主程序
print("Loading graphs...")
train_graphs, train_file_names, feature_encoders = load_and_process_data(dataset_folder, data_file)

# 归一化
if os.path.exists(normalized_data_file):
    print(f"Loading normalized data from {normalized_data_file}...")
    train_graphs, train_file_names, feature_encoders = load_data(normalized_data_file)
else:
    print(f"{normalized_data_file} not found. Normalizing data and creating {normalized_data_file}...")
    train_graphs, train_file_names, feature_encoders = normalize_and_save_data(train_graphs, train_file_names, feature_encoders, normalization_method="minmax")

# ground truth 集合包含在训练集中
ground_truth = {file_name: [file_name] for file_name in train_file_names}

# 生成不同 mask 程度的不完整图作为测试集
mask_ratios = [0.1, 0.2, 0.3]
incomplete_test_graphs = []
incomplete_test_file_names = []

for graph, file_name in zip(train_graphs, train_file_names):
    for mask_ratio in mask_ratios:
        incomplete_graph = mask_graph(graph, mask_ratio)
        incomplete_test_graphs.append(incomplete_graph)
        incomplete_test_file_names.append(f"{file_name}_mask_{mask_ratio}")

# 训练 GCN 模型
in_feats = len(train_graphs[0].ndata['feat'][0])
h_feats = 16
out_feats = in_feats
model = train_gcn_model(train_graphs, in_feats, h_feats, 50)

# 获取训练集和测试集的图表示
train_embeddings = get_graph_embeddings(model, train_graphs)
test_embeddings = get_graph_embeddings(model, incomplete_test_graphs)

# 评估推荐结果
n_recommendations = 5
recommendations, hit_rate, mrr = evaluate_recommendations(train_embeddings, test_embeddings, train_file_names, incomplete_test_file_names, ground_truth, top_n=n_recommendations)

# 保存结果到现有的 results 文件夹中
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_folder = os.path.join(output_folder, timestamp)
os.makedirs(results_folder)

with open(os.path.join(results_folder, 'recommendations.json'), 'w') as f:
    json.dump(recommendations, f, indent=4)

metrics = {
    'n_recommendations': n_recommendations,
    'mask_ratios': mask_ratios,
    'Hit Rate': hit_rate,
    'MRR': mrr
}

with open(os.path.join(results_folder, 'results.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

# 保存模型
model_path = os.path.join(results_folder, 'trained_gcn_model.pth')
torch.save(model.state_dict(), model_path)

print(f"Hit Rate: {hit_rate}")
print(f"Mean Reciprocal Rank (MRR): {mrr}")
# print("Recommendations:", recommendations)