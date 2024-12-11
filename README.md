# LLM_Fine-tuning_training_coursedesign
# 基于小规模LLM的多种微调方法比较与分析

## 项目简介
本项目旨在深入探讨基于小规模中文大语言模型（LLM）的多种微调方法，并对其性能进行系统性比较与分析。  
通过本研究，期望为实际应用中的模型优化提供参考，并为用户选择合适的微调策略提供理论依据。

---

## 研究背景
大型语言模型 (LLM) 微调是采用预训练模型并在较小的特定数据集上进一步训练它们以满足特定任务需求的过程。  
微调方法能够弥合通用模型与特定应用之间的差距，使语言模型的功能更符合期望。

---

## 项目目标
本研究的主要目标包括：

1. **复现现有微调算法**  
   比较不同微调方法（如监督微调、无监督微调、冻结层微调等）的优劣。

2. **创新性探索微调策略**  
   尝试结合多种微调方法，提出更加高效的混合微调方案。

3. **建立性能评估系统**  
   从任务性能、参数量、推理速度等多个维度构建一套全面的微调性能评估体系。

---
## 项目结构
```plaintext
├── data/                # 数据集存放路径
├── models/              # 预训练模型与微调后的模型
├── scripts/             # 微调相关脚本
├── results/             # 微调实验结果与评估报告
├── docs/                # 项目文档及参考资料
├── experiments/         # 实验配置及运行记录
└── README.md            # 项目介绍文件
```

---

## 实验内容
**微调方法**
监督微调
无监督微调
冻结层微调
方法融合（创新性探索）

## 评估维度
**任务性能**：模型在特定任务中的精确性与效果。
**参数量**：模型的存储与计算开销。
**推理速度**：微调后模型在不同设备上的运行效率。

## 安装依赖
请先安装以下工具：
```requirements.txt
Python 版本 >= 3.8
PyTorch
其他依赖见 requirements.txt
```

## 准备数据
将数据集放置于 data/ 目录下，格式要求详见 数据说明。

## 运行脚本
微调脚本位于 scripts/ 目录中，具体运行方式详见对应文件的注释。

---
## 进展展示

### 学习记录（csdn博客记录）：
https://blog.csdn.net/2301_80686939/article/details/144414221?spm=1001.2014.3001.5502
### 数据集准备：
huggingface选取有关文言文的数据集
### 微调方法实现：
主要尝试PEFT（Parameter-Efficient Fine-Tuning） 的微调方法
### 实验结果分析：
使用gpt-4o对结果进行分析，评估相应的性能，在训练过程中使用visdom对训练数据进行记录

---

## 后续计划
完成不同方法的实验并记录详细结果。
撰写项目报告及总结。
发布可复现的代码和结果供社区参考。

### 贡献者
Tomhow
### 联系方式
如果您对本项目有任何疑问或建议，请通过以下方式联系我：

Email: [1465276650@qq.com]
GitHub: [https://github.com/ChunJiangWuYueMing]
