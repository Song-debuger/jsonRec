import random
import os
from datetime import datetime
import json

from GNN_engine import recommend_for_test, evaluate_recommendations, train_and_evaluate
from utilities import load_graphs_from_folder, mask_graph, visualize_network

# 创建测试结果文件夹
output_folder = 'results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取数据集
dataset_folder = 'Datasets/jsonGraph'
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

# 设置推荐数量
n_recommendations = 4

# 训练和评估
hit_rate, mrr, recommendations, similarity_matrix = train_and_evaluate(
    graphs, incomplete_test_graphs, file_names, incomplete_test_file_names, ground_truth_file_names, ground_truth, mask_ratios, n_recommendations
)

# 输出评估结果
print(f"Hit Rate: {hit_rate}")
print(f"Mean Reciprocal Rank: {mrr}")

# 保存结果
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_folder = os.path.join(output_folder, timestamp)
os.makedirs(results_folder)

# 保存推荐结果
with open(os.path.join(results_folder, 'recommendations.json'), 'w') as f:
    json.dump(recommendations, f, indent=4)

# # 保存训练参数和评估指标
# results = {
#     'n_recommendations': n_recommendations,
#     'mask_ratios': mask_ratios,
#     'precision': precision,
#     'recall': recall,
#     'f1_score': f1,
#     'hit_rate': hit_rate
# }

# 保存训练参数和评估指标
results = {
    'n_recommendations': n_recommendations,
    'mask_ratios': mask_ratios,
    'hit_rate': hit_rate,
    'mean_reciprocal_rank': mrr
}

with open(os.path.join(results_folder, 'results.json'), 'w') as f:
    json.dump(results, f, indent=4)

# # 保存相似度矩阵
# pd.DataFrame(similarity_matrix).to_csv(os.path.join(results_folder, 'similarity_matrix.csv'))
#
# # 可视化推荐结果和相似度矩阵
# visualize_network(recommendations, results_folder, top_n=n_recommendations)
# sns.heatmap(similarity_matrix[:10, :10], cmap='coolwarm')  # 只展示前10个图的相似度矩阵
# plt.title('Similarity Matrix Heatmap (Top 10)')
# plt.savefig(os.path.join(results_folder, 'similarity_matrix_heatmap.png'))
# plt.show()