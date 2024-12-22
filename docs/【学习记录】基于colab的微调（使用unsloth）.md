简介
鉴于在本地部署更高性能LLM的困难，我初步尝试在colab上进行我的微调目标，尝试使用unsloth来加速微调的进程，主要使用了PEFT（Parameter-Efficient Fine-Tuning） 微调方法，希望在修改代码后能够实现文言文的翻译工作（实例化微调的性能），同时也希望在本地部署（使用参数量较小的LLM）。

在学习的过程中借鉴了unsloth的官方文档，colab的.ipynb文件。本学习记录主要记录逐句解读官方文件中代码的过程，仅供参考。

代码逐步分析
%%capture
!pip install unsloth
# Also get the latest nightly Unsloth!
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
1.上述代码的作用是安装并升级 Unsloth 框架。

（1）%%capture

这是 Jupyter Notebook 的魔法命令，用于捕获单元格输出，防止在安装过程中产生的大量日志污染笔记本界面。

（2）安装 unsloth：
json
!pip install unsloth

这一步安装 unsloth，是用于大模型微调的 Python 库。

（3）升级为最新版本：

!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

!pip uninstall unsloth -y：卸载已安装的 unsloth。
!pip install --upgrade --no-cache-dir --no-deps：从 GitHub 上安装最新的开发版本，确保使用最新的框架特性。
git+https://github.com/unslothai/unsloth.git：直接从 Unsloth 的 GitHub 仓库获取最新代码。
--no-deps：不安装依赖（假设系统已有依赖，且只需升级核心库）。--no-cache-dir：禁用缓存，避免安装旧版本。

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
 
# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
    "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
 
    "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
 
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit" # NEW! Llama 3.3 70B!
] # More models at https://huggingface.co/unsloth
 
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct", # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
2.这段代码用于加载 Llama 3.2 模型并准备进行推理或微调。

1. 导入模块
from unsloth import FastLanguageModel import torch

FastLanguageModel：Unsloth 的核心工具，用于快速加载语言模型。
torch：PyTorch库，用于深度学习模型的加载和操作。
2. 参数配置
max_seq_length = 2048 dtype = None load_in_4bit = True

max_seq_length：定义模型输入的最大序列长度，2048 是常见的值，可根据硬件性能调整。
dtype：
如果设为 None，会根据设备自动选择最佳数据类型。
可手动指定：
float16：适用于 Tesla T4、V100 等支持半精度的显卡。
bfloat16：适用于 Ampere 架构（A100）及更新显卡，适合更高性能的半精度计算。
load_in_4bit：启用 4bit 量化，显著降低内存使用，适合在资源受限的设备上运行。
3. 可用模型列表
fourbit_models = [ ... ]

提供多个预量化的模型，支持 Hugging Face 格式，避免 OOM（内存不足）问题。
新增 Llama 3.2 系列 模型，适用于快速下载和更高效推理。
可选模型范围广，包括 Meta、Mistral、Phi、Gemma 等不同性能的模型。
你选择了：

model_name = "unsloth/Llama-3.2-1B-Instruct"

这是一个 1B 参数的微调指令模型，适合中小型任务。

4. 加载模型
model, tokenizer = FastLanguageModel.from_pretrained( model_name = "unsloth/Llama-3.2-1B-Instruct", max_seq_length = max_seq_length, dtype=dtype, load_in_4bit=load_in_4bit, )

from_pretrained：从预训练模型库加载模型及其对应的分词器。
主要参数：
model_name：选择具体模型。
max_seq_length：限制输入序列的最大长度。
dtype：数据类型自动检测或手动指定。
load_in_4bit：开启 4bit 量化。
注：如加载带有访问限制的模型（如 Meta Llama 2 系列），需提供 Hugging Face 的访问令牌：

token = "hf_..." # 替换为你的 Hugging Face API 密钥

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
3.这段代码主要配置了 PEFT（Parameter-Efficient Fine-Tuning） 微调方法，以下是逐点解析以及对其相关参数的解释：

代码作用概述
PEFT 简介：
Parameter-Efficient Fine-Tuning 是一种轻量化微调技术，只对模型的小部分参数进行调整（如权重矩阵的低秩分解层），而不是微调整个模型。它能显著减少训练时间和显存需求。
在文言文翻译任务中，通过 LoRA（Low-Rank Adaptation） 等 PEFT 技术，可以在较小资源条件下实现微调。
关键参数解析
r

描述： LoRA 的秩参数，控制低秩矩阵的大小。
建议值：
小任务（简单微调）：8 或 16。
复杂任务（文言文翻译）：建议 16 或 32。
意义：
越大，模型能表达的额外特性越强，但显存占用也会增加。
target_modules

描述： 选择 LoRA 应用于哪些模块的参数。
默认配置： 针对 Transformer 的关键模块：
q_proj, k_proj, v_proj, o_proj（自注意力相关投影层）。
gate_proj, up_proj, down_proj（前馈网络的升维/降维层）。
建议：
默认配置通常效果良好。
lora_alpha

描述： 缩放因子，用于控制 LoRA 模块的输出影响力。
建议值： 16 是常见的默认值。
lora_dropout

描述： 在 LoRA 模块中引入随机丢弃（Dropout）来防止过拟合。
建议值： 任务简单或数据充足时设置为 0（优化性能）。
bias

描述： 是否对模型中的偏置（bias）参数应用微调。
选项：
"none"：不微调偏置，资源效率最高。
"all"：微调所有偏置。
"lora_only"：仅微调与 LoRA 模块相关的偏置。
建议： 文言文翻译建议使用 "none"。
use_gradient_checkpointing

描述： 激活梯度检查点（Gradient Checkpointing）以节省显存。
选项：
True：普通检查点功能。
"unsloth"：Unsloth 优化版，显存占用减少约 30%，可支持更长的上下文长度和更大的批量大小。
建议： 使用 "unsloth"，特别适合显存有限的环境。
random_state

描述： 随机种子，用于确保结果可复现。
建议： 设置固定值（如 3407）确保一致性。
use_rslora

描述： 启用 Rank Stabilized LoRA（RSLora），一种改进方法，用于在极小数据集上实现稳定的低秩适配。
建议： 默认关闭（False）即可。
loftq_config

描述： 如果需要结合 LoftQ（低秩量化）进行优化，可以传入对应配置。
建议： 当前任务可不启用，设置为 None。
分析与建议
配置合理性：

代码中的默认配置非常适合资源有限的环境，尤其是 Tesla T4 GPU。
文言文翻译可能需要更多表达能力，可以适当调高 r 值（如设置为 32）。
显存管理：

启用 "unsloth" 的梯度检查点能够显著节省显存，非常适合低显存设备。
改进方向：

如果训练数据有限（如少于 1 万条样本），可以考虑启用 use_rslora = True 以提升稳定性。
若显存空间较富裕，可关闭 lora_dropout = 0，设置为 0.1 左右以增强泛化能力。
from unsloth.chat_templates import get_chat_template
 
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)
 
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass
 
from datasets import load_dataset
dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
 4.这段代码的核心功能是加载一个 聊天模板 并将其应用于一个数据集的预处理。以下是逐步分析与解释：

代码结构与作用分析
引入聊天模板 (get_chat_template)
python
from unsloth.chat_templates import get_chat_template tokenizer = get_chat_template( tokenizer, chat_template = "llama-3.1", )
功能： 通过 get_chat_template 方法，加载一个聊天格式化模板，并将其绑定到模型的 tokenizer。
chat_template:
指定聊天模板类型（如 llama-3.1）。模板包含对话样式的预定义格式，比如：
用户输入
AI 的回复
特定的标记样式（如 User: 和 Assistant:）
结果： 模板会帮助模型理解对话的上下文结构，使其更适合对话生成任务。
2. 定义格式化函数 (formatting_prompts_func)
python
def formatting_prompts_func(examples): convos = examples["conversations"] texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos] return { "text" : texts, } pass 
功能： 处理数据集中对话样本，将其转换为适合模型微调的格式。
步骤解析：
提取对话： 从输入样本中提取对话内容（examples["conversations"]）。
应用模板：
使用 apply_chat_template 方法，将对话内容转换为预设模板格式。
tokenize = False: 不立即对结果进行分词处理。
add_generation_prompt = False: 不自动添加生成提示，方便后续灵活处理。
输出结果： 返回包含格式化文本的字典。
3. 加载数据集
python
from datasets import load_dataset dataset = load_dataset("mlabonne/FineTome-100k", split = "train") 
功能： 使用 Hugging Face 的 datasets 库加载一个名为 FineTome-100k 的对话数据集。
参数解析：
dataset: 数据集名称。
split = "train": 加载训练集部分。
数据集用途：
FineTome-100k 是一个包含 100,000 条对话样本的数据集，适合用于微调对话模型。
重要概念与作用
1. Chat Template (聊天模板)
作用： 在对话任务中，聊天模板能标准化模型的输入输出结构。例如：
plaintext
User: 你喜欢文言文吗？ Assistant: 是的，我非常喜欢文言文。 
模板帮助模型在对话中保持一致的语法和上下文标记。
2. 数据预处理 (formatting_prompts_func)
作用： 将原始对话数据转换为符合聊天模板的格式，为后续微调提供输入。
示例： 假设 conversations 的原始数据是：
json
{ "conversations": [ {"user": "屈原是谁？", "assistant": "屈原是中国战国时期的伟大诗人。"}, {"user": "你能背《离骚》吗？", "assistant": "《离骚》很长，但我可以背诵其中的片段。"} ] } 
应用模板后，转换为：
plaintext
User: 屈原是谁？ Assistant: 屈原是中国战国时期的伟大诗人。 User: 你能背《离骚》吗？ Assistant: 《离骚》很长，但我可以背诵其中的片段。 
3. 数据集加载
FineTome-100k 数据集：
可能包含各种领域的对话样本，适合用于通用聊天模型的微调。
如果你需要微调为文言文翻译模型，可能需要自行筛选、清洗或增强数据集。
下一步改进建议
检查数据集中是否包含文言文翻译对话样本：
如果没有，需要自行准备并合并特定样本到数据集中。
扩展格式化函数：
根据文言文翻译任务需求，修改 formatting_prompts_func，为数据集样本添加明确的任务提示。例如：
plaintext
User: 翻译文言文：《离骚》。原文："帝高阳之苗裔兮，朕皇考曰伯庸。" Assistant: 翻译：我是帝高阳的后代，我的父亲名叫伯庸。
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
 
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)
 5.这一段代码的目的是设置并初始化一个基于 SFTTrainer 的模型训练流程，下面我会逐步解释代码的各个部分：

1. 导入必要的库
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
SFTTrainer 来自 trl 库，它是一个用于训练语言模型的专用类。这个类简化了微调过程，尤其是在处理带有对话的任务时。
TrainingArguments 来自 transformers，它用于设置训练的参数，如批次大小、学习率等。
DataCollatorForSeq2Seq 用于为序列到序列任务（如翻译任务）准备数据，将数据批次化。
is_bfloat16_supported 用于检查是否支持 bfloat16 精度。
2. SFTTrainer 设置
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)
主要参数解释：
1. model 和 tokenizer
model：你的预训练语言模型（如 Llama 3.2）。
tokenizer：对应的分词器，用于将文本转化为模型能理解的输入。
2. 数据集设置
train_dataset：训练数据集，假设这个数据集已经按之前的代码处理成了标准格式。
dataset_text_field = "text"：指定数据集中的字段名，这里使用 text 字段作为输入。
dataset_num_proc = 2：使用 2 个进程加载数据集（并行化处理）。
packing = False：禁用 packing，意味着每个序列的长度是固定的，适用于较长的文本。
3. TrainingArguments 设置
这些参数控制训练的具体细节：

per_device_train_batch_size = 2：每个设备上的训练批次大小为 2。
gradient_accumulation_steps = 4：每 4 个步骤累积一次梯度，这样可以在相对较小的 GPU 内存下处理更大的批次。
warmup_steps = 5：预热步数，在训练初期使用较小的学习率，逐渐增加。
max_steps = 60：训练的总步数为 60。注意，这个参数和 num_train_epochs 是互斥的，可以选择设置一个。
learning_rate = 2e-4：初始学习率设置为 2e-4。
fp16 和 bf16：决定是否启用混合精度训练（fp16 或 bfloat16）。is_bfloat16_supported() 会检查是否支持 bfloat16，若不支持，则使用 fp16。
logging_steps = 1：每 1 步记录一次日志。
optim = "adamw_8bit"：优化器使用 8bit 精度的 AdamW 优化器，这样能减少内存消耗。
weight_decay = 0.01：使用权重衰减，防止过拟合。
lr_scheduler_type = "linear"：使用线性学习率调度，即随着训练的进行，学习率逐渐下降。
seed = 3407：设置随机种子，保证训练的可重复性。
output_dir = "outputs"：模型的输出路径，训练过程中会保存模型的检查点。
report_to = "none"：不使用像 WandB 这样的外部工具来报告训练过程。
总结
这段代码配置了一个完整的训练管道，使用了 SFTTrainer 来训练微调模型，支持混合精度训练，调整了各种训练参数（如批次大小、学习率、优化器等）。
你可以通过修改这些参数来调整训练过程，确保模型在特定任务上达到最好的表现。
如何根据任务调整
训练步数： 根据任务的大小，max_steps 或 num_train_epochs 可以调整。
批次大小和梯度累积： 如果遇到显存不足，可以减少 per_device_train_batch_size 或增加 gradient_accumulation_steps。
学习率调整： 你可以根据实验调整学习率，可能需要在训练过程中逐步调整以找到最佳值。
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
 6.这段代码的核心是通过 train_on_responses_only 方法，将训练过程聚焦在模型生成的 response 部分，而不是完整的对话数据。

代码功能分析
from unsloth.chat_templates import train_on_responses_only 
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
1. train_on_responses_only 方法
功能： 该方法从完整的对话中分离出用户提问（instruction_part）和模型生成的回答（response_part），并指定只对 response_part 进行微调。
优势： 这样可以避免模型重复学习用户输入的部分，专注优化模型生成回答的能力。
适用场景：
对话式微调。
当对训练数据中的 response 部分有更高质量要求时。
用户输入内容较长或包含冗余时。
2. 参数说明
instruction_part 和 response_part

instruction_part：标识用户提问部分的模板，这里定义为 <|start_header_id|>user<|end_header_id|>\n\n。
response_part：标识模型回答部分的模板，这里定义为 <|start_header_id|>assistant<|end_header_id|>\n\n。
作用： 匹配并提取对话数据中用户提问和模型回答的部分，分别标记成“指令”和“回答”。
trainer

这个 trainer 是之前 SFTTrainer 的实例，经过 train_on_responses_only 处理后，会针对数据集中的 response_part 进行重点训练。
结合任务的应用
1. 格式调整
对于翻译《史记》这样的任务，你的数据集中可能并没有显式的 instruction 和 response，而是由古文（classical）和白话文（modern）组成。这种情况下，你需要修改 instruction_part 和 response_part：

假设 classical 对应输入（用户提问部分），modern 对应输出（模型回答部分），可以设置为：
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_classical|>",
    response_part = "<|start_modern|>",
)
2. 为什么专注 response_part
在翻译任务中，模型生成的 白话文翻译（response） 是核心目标，而输入的古文（instruction）无需模型反复学习。因此，这样的训练方式能提升训练效率并降低不必要的参数更新。

扩展与调整
如果数据没有模板标记：

可以用类似 formatting_prompts_func 的方法预处理数据集，在每条数据前后加入自定义模板标签。
例如：
<|start_classical|>
离骚
<|start_modern|>
离骚是中国古代文学的代表作之一...
是否一定要专注 response_part：

如果希望模型同时优化对 instruction 的理解，可以跳过 train_on_responses_only，直接使用全对话数据训练。
总结
这段代码让模型聚焦于优化生成部分，适合对话式任务和翻译任务中需要高质量输出的场景。如果你的数据已经结构化，可以直接应用；否则需要通过预处理脚本，明确标记 instruction 和 response 的部分，再进行微调训练。

trainer_stats = trainer.train()
 7.这段代码是模型微调的核心部分，通过调用 trainer.train() 方法正式启动训练流程，并返回训练的统计信息 trainer_stats。

代码功能分析
trainer_stats = trainer.train()
功能：

trainer.train() 是 transformers 中训练器的核心方法，负责执行模型的完整训练过程。
训练过程中会基于配置的参数（如数据集、学习率、批量大小等）优化模型的参数。
返回值：

trainer.train() 返回的是训练的详细统计信息，通常包含以下内容：
训练损失（loss）： 每个步骤或每个 epoch 的损失值，用于衡量模型的训练效果。
训练步数（steps）： 已完成的优化步数。
时间统计： 训练所用的总时间和每个步骤的平均耗时。
最终模型： 包括微调后的权重和其他相关参数。
结合代码的关键点
依赖于之前的 SFTTrainer 配置

训练过程会使用之前在 SFTTrainer 中定义的所有参数，例如：
数据集（dataset）： train_dataset 指定的训练数据集。
优化器参数： 学习率（learning_rate）、权重衰减（weight_decay）等。
微调方法： 是否使用 LoRA 等参数高效微调方法。
最大训练步数（max_steps）： 此任务中被设置为 60。
训练的主要过程：

逐步迭代数据集，计算损失（如交叉熵）。
根据梯度更新模型参数。
如果配置了日志记录（logging_steps），会输出中间结果。
输出与后续操作
查看训练统计信息

训练完成后可以通过打印 trainer_stats 查看详细的训练结果，例如：
print(trainer_stats)
常见输出包括：
总训练步数（total_training_steps）。
每个日志步长的损失值（loss）。
训练耗时（training_time）。
保存模型

为了后续推理，可以将微调后的模型和 tokenizer 保存到磁盘：
trainer.save_model("outputs")
tokenizer.save_pretrained("outputs")
调试与调整

如果 loss 在训练结束后仍然较高，说明模型可能：
数据量不足。
学习率设置不当（可尝试降低）。
模型容量不足（可考虑切换到更大的模型）。
如果 loss 降得太低，可能是过拟合的信号，可以增加正则化或扩充数据集。
结合《史记》翻译任务的注意点
验证 train_dataset 的质量

确保数据集已正确标注，并通过 formatting_prompts_func 和 train_on_responses_only 函数预处理。
使用 dataset["text"][:5] 检查数据格式是否匹配。
关注训练时长

由于设置了 max_steps = 60，这是一轮快速调试用的训练。
在实际任务中，如果效果不理想，可以增加步数（max_steps）或直接设置 num_train_epochs。
后续测试

微调完成后，可以用训练好的模型对新古文进行测试，检查翻译质量：
test_input = "<|start_classical|>离骚<|end_classical|>"
result = trainer.model.generate(tokenizer.encode(test_input, return_tensors="pt"))
print(tokenizer.decode(result[0]))
总结
trainer.train() 是微调的核心步骤，输出的训练统计信息可以帮助你评估训练的效果。在《史记》翻译任务中，检查 loss 的趋势和生成测试结果，是评估模型性能的关键指标。
————————————————

                            版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。
                        
原文链接：https://blog.csdn.net/2301_80686939/article/details/144414221