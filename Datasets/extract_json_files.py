import os
import shutil


def extract_json_files(source_dir, target_dir):
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录及其子目录
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".json"):
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_dir, file)

                # 如果目标目录中已存在同名文件，可以选择重命名或覆盖
                if os.path.exists(target_file):
                    target_file = os.path.join(target_dir,
                                               f"{os.path.splitext(file)[0]}_copy{os.path.splitext(file)[1]}")

                # 复制文件到目标目录
                shutil.copy2(source_file, target_file)
                print(f"已复制: {source_file} 到 {target_file}")


source_dir = "C:/Users/98157/Desktop/毕设思路/modelset/modelset/graph/repo-genmymodel-uml/data"

target_dir = "C:/Users/98157/Documents/MORGAN-e706d1474d24fe53c5c3e0ad47e0c1b545904637/MORGAN/Datasets/jsonGraph"
extract_json_files(source_dir, target_dir)
