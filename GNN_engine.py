from grakel import graph_from_networkx, GraphKernel
import numpy as np

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
    successful_recommendations = 0
    reciprocal_ranks = []

    for test_graph, recommended_files in recommendations.items():
        if test_graph in ground_truth:
            true_files = ground_truth[test_graph]
            true_set = set(true_files)

            # 检查推荐结果中是否包含任何一个 ground truth 图
            hit = any(file in true_set for file in recommended_files)
            if hit:
                successful_recommendations += 1

            # 计算 Reciprocal Rank
            # 高MRR表示推荐系统能够在前几个推荐结果中包含正确结果，推荐效果较好。
            for rank, recommended_file in enumerate(recommended_files, start=1):
                if recommended_file in true_set:
                    reciprocal_ranks.append(1 / rank)
                    break
            else:
                reciprocal_ranks.append(0)

    hit_rate = successful_recommendations / len(recommendations)
    mrr = np.mean(reciprocal_ranks)

    return hit_rate, mrr

# 训练和评估函数
def train_and_evaluate(graphs, incomplete_test_graphs, file_names, incomplete_test_file_names, ground_truth_file_names, ground_truth, mask_ratios, n_recommendations):
    # 将 NetworkX 图转换为 grakel 库使用的图格式
    G_train = list(graph_from_networkx(graphs, node_labels_tag='label'))
    G_test = list(graph_from_networkx(incomplete_test_graphs, node_labels_tag='label'))

    # 使用 grakel 计算图核
    gk = GraphKernel(kernel={"name": "weisfeiler_lehman", "n_iter": 5}, normalize=True)
    K_train = gk.fit_transform(G_train)  # 训练集图核矩阵
    K_test_train = gk.transform(G_test)  # 测试集与训练集之间的图核相似度矩阵

    # 推荐
    recommendations = recommend_for_test(K_test_train, incomplete_test_file_names, file_names, top_n=n_recommendations)

    # 更新 ground truth 集合以包含所有生成的不完整图
    for file_name in ground_truth_file_names:
        for mask_ratio in mask_ratios:
            ground_truth[f"{file_name}_mask_{mask_ratio}"] = [file_name]

    # 评估推荐
    hit_rate, mrr = evaluate_recommendations(recommendations, ground_truth)
    return hit_rate, mrr, recommendations, K_test_train
