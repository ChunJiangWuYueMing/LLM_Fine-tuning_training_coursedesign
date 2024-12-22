'''
convert_dataset.py is a script that reads the content of the source and target files in the subfolders of the folder_path
and combines them into a dataset. The source file is the input field, and the target file is the output field.
'''
import os
import json
import pandas as pd

folder_path = "D:\study_space\AI-class\Program\LLM_Fine-tuning_training_coursedesign\data\Classical-Modern\Classical-Modern-main\双语数据"


# get all subfolders in the folder, then for each subfolder, get source file and target file,
# then read the content of the files then combine them into a dateset, the source file is the input field,
# the target file is the output field, and the target file is the output field

def get_files(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    print(subfolders)
    dataset = []

    source_file = "source.txt"
    target_file = "target.txt"
    for x in subfolders:
        with open(os.path.join(x, source_file), "r", encoding="utf-8") as f:
            source_content = f.read()
        with open(os.path.join(x, target_file), "r", encoding="utf-8") as f:
            target_content = f.read()

        # source and target needs to be split by "\n"
        source_content = source_content.split("\n")
        target_content = target_content.split("\n")

        # source and target should be saved into dateset line by line
        for i in range(len(source_content)):
            dataset.append([source_content[i], target_content[i]])

    return dataset


dataset = get_files(folder_path)

# add one column "instruction" with the content "请把古文翻译成现代汉语" to the dataset
df = pd.DataFrame(dataset, columns=["source", "target"])
df["instruction"] = "请把现代汉语翻译成古文"

# rename the columns: source -> output, target -> input
df.rename(columns={"source": "output", "target": "input"}, inplace=True)

# print length of the dataset
print(len(df))

# save the dataset into a jsonl file
df.to_json("dataset.jsonl", orient="records", lines=True, force_ascii=False)