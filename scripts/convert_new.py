import os
import json
import pandas as pd

folder_path = "D:/study_space/AI-class/Program/LLM_Fine-tuning_training_coursedesign/data/Classical-Modern/Classical-Modern-main/双语数据"


def find_files_recursive(folder_path, source_file="source.txt", target_file="target.txt"):
    """
    递归遍历指定路径下的所有子文件夹，寻找 source.txt 和 target.txt。
    返回包含文件路径的元组列表。
    """
    file_pairs = []

    for root, _, files in os.walk(folder_path):  # 遍历文件夹及子文件夹
        if source_file in files and target_file in files:
            source_path = os.path.join(root, source_file)
            target_path = os.path.join(root, target_file)
            file_pairs.append((source_path, target_path))

    return file_pairs


def get_files(folder_path):
    """
    从指定路径下递归查找 source.txt 和 target.txt，并将内容组合成 dataset。
    """
    file_pairs = find_files_recursive(folder_path)
    print(f"发现 {len(file_pairs)} 对文件")

    dataset = []

    for source_path, target_path in file_pairs:
        try:
            # 读取文件内容
            with open(source_path, "r", encoding="utf-8") as f:
                source_content = f.read().strip().split("\n")
            with open(target_path, "r", encoding="utf-8") as f:
                target_content = f.read().strip().split("\n")

            # 检查内容长度是否一致
            if len(source_content) != len(target_content):
                print(f"警告：文件内容长度不匹配，跳过文件：{source_path}, {target_path}")
                continue

            # 将内容添加到数据集中
            for src, tgt in zip(source_content, target_content):
                dataset.append([src, tgt])

        except Exception as e:
            print(f"处理文件 {source_path}, {target_path} 时出错：{e}")

    return dataset


def save_dataset(dataset, output_file):
    """
    将 dataset 保存为 JSONL 格式文件。
    """
    # 转换为 DataFrame
    df = pd.DataFrame(dataset, columns=["source", "target"])
    df["instruction"] = "请把现代汉语翻译成古文"

    # 重命名列
    df.rename(columns={"source": "output", "target": "input"}, inplace=True)

    # 打印数据集长度
    print(f"数据集长度：{len(df)}")

    # 保存为 JSONL 文件
    df.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"数据集已保存到 {output_file}")


# 主程序
if __name__ == "__main__":
    dataset = get_files(folder_path)
    output_dir = os.path.join("D:/study_space/AI-class/Program/LLM_Fine-tuning_training_coursedesign/data", "processed")
    os.makedirs(output_dir, exist_ok=True)  # 创建目录（如果不存在）
    output_file = os.path.join(output_dir, "dataset.jsonl")
    if dataset:
        save_dataset(dataset, output_file)
    else:
        print("未生成数据集，请检查源数据文件。")
