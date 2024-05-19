import os
import shutil
import random
from sklearn.model_selection import train_test_split

def split_json_files(source_dir, train_dir, test_dir, groundtruth_dir, test_size=0.2, groundtruth_size=0.1):
    # 确保目标目录存在
    for dir_path in [train_dir, test_dir, groundtruth_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 收集所有 JSON 文件
    json_files = [f for f in os.listdir(source_dir) if f.endswith(".json")]

    # 先从数据中划分出 ground truth 部分
    train_files, groundtruth_files = train_test_split(json_files, test_size=groundtruth_size, random_state=42)

    # 再从剩余的文件中划分训练集和测试集
    train_files, test_files = train_test_split(train_files, test_size=test_size, random_state=42)

    # 定义复制文件的函数
    def copy_files(file_list, destination_dir):
        for file_name in file_list:
            source_file = os.path.join(source_dir, file_name)
            target_file = os.path.join(destination_dir, file_name)
            shutil.copy2(source_file, target_file)
            print(f"已复制: {source_file} 到 {target_file}")

    # 复制文件到相应的目录
    copy_files(train_files, train_dir)
    copy_files(test_files, test_dir)
    copy_files(groundtruth_files, groundtruth_dir)

source_dir = "C:/Users/98157/Documents/MORGAN-e706d1474d24fe53c5c3e0ad47e0c1b545904637/MORGAN/Datasets/jsonGraph"
train_dir = "C:/Users/98157/Documents/MORGAN-e706d1474d24fe53c5c3e0ad47e0c1b545904637/MORGAN/Datasets/D_json/train"
test_dir = "C:/Users/98157/Documents/MORGAN-e706d1474d24fe53c5c3e0ad47e0c1b545904637/MORGAN/Datasets/D_json/test"
groundtruth_dir = "C:/Users/98157/Documents/MORGAN-e706d1474d24fe53c5c3e0ad47e0c1b545904637/MORGAN/Datasets/D_json/gt"

split_json_files(source_dir, train_dir, test_dir, groundtruth_dir)
