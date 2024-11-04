from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM, DataCollatorWithPadding
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import evaluate
import torch
from pynvml import *
from transformers import EarlyStoppingCallback
import deepspeed
import os
import torch
import torch.nn.functional as F
import math


# 本地模型路径
model_local_path = '/mnt/workspace/me/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct/'
# 本地数据集路径
dataset_local_path = '/mnt/workspace/me/CodeAlpaca-20k/'
# 最长序列
max_length = 1024

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_local_path)
# 加载数据集
dataset = load_dataset(dataset_local_path)
train_val_dataset = dataset["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)


# def preprocess_function(examples):
#     model_inputs = tokenizer(
#         examples["prompt"],
#         max_length=max_length,
#         padding="longest",
#         truncation=True,
#         return_tensors="pt"
#     )
#
#     labels = tokenizer(
#         examples["completion"],
#         max_length=max_length,
#         padding="longest",
#         truncation=True,
#         return_tensors="pt"
#     )
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs
def preprocess_function(examples):
    # 组合 instruction 和 input 作为提示语
    prompts = []
    for inst, inp in zip(examples["instruction"], examples["input"]):
        # 如果 input 不为空，则组合 instruction 和 input
        if inp:
            prompt = f"Instruction: {inst}\nInput: {inp}"
        else:
            prompt = f"Instruction: {inst}"
        prompts.append(prompt)

    # 分词处理输入
    model_inputs = tokenizer(
        prompts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"  # 直接返回PyTorch张量
    )

    # 分词处理输出
    labels = tokenizer(
        examples["output"],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"  # 直接返回PyTorch张量
    )

    # 设置标签，将填充的 token 设为 -100，使其在计算损失时被忽略
    model_inputs["labels"] = labels["input_ids"]
    for i in range(len(model_inputs["labels"])):
        for j in range(len(model_inputs["labels"][i])):
            if labels["attention_mask"][i][j] == 0:
                model_inputs["labels"][i][j] = -100

    return model_inputs


# 应用预处理
train_dataset = train_val_dataset["train"].map(preprocess_function, batched=True)
val_dataset = train_val_dataset["test"].map(preprocess_function, batched=True)

# lora配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 创建早停回调
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.01
)

# DeepSpeed 训练参数
local_rank = int(os.getenv('LOCAL_RANK', '0'))

training_args = TrainingArguments(
    num_train_epochs=3,
    learning_rate=1e-5,
    lr_scheduler_type='linear',
    warmup_ratio=0.1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,

    # DeepSpeed specific
    deepspeed="ds_config.json",  # DeepSpeed配置文件路径
    local_rank=local_rank,  # 本地进程序号

    # 优化相关
    gradient_checkpointing=False,
    fp16=True,
    gradient_accumulation_steps=4,

    # 日志相关
    log_level='debug',
    logging_dir='./logs',
    logging_steps=500,

    # 评估相关
    evaluation_strategy="steps",
    eval_steps=500,
    metric_for_best_model="eval_loss",

    # checkpoint相关
    output_dir='./results',
    save_strategy="steps",
    save_steps=2_000,
    save_total_limit=2,
    load_best_model_at_end=True,

    # 其他
    remove_unused_columns=True,
    dataloader_drop_last=True,
    disable_tqdm=False,
)


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()





def compute_perplexity(logits, labels, ignore_index=-100):
    """
    计算困惑度
    Args:
        logits: shape (batch_size, sequence_length, vocab_size)
        labels: shape (batch_size, sequence_length)
        ignore_index: 忽略的标签值，通常是padding的index
    """
    # 将logits和labels移位对齐，去掉最后一个token的预测和第一个token的标签
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # 计算交叉熵损失
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    # 创建mask去除padding的影响
    mask = (shift_labels.view(-1) != ignore_index)
    loss = loss[mask]

    # 计算平均交叉熵
    mean_ce = loss.mean().item()

    # 计算困惑度 (perplexity = exp(cross_entropy))
    perplexity = math.exp(mean_ce)

    return perplexity


def compute_accuracy(predictions, labels, ignore_index=-100):
    """
    计算准确率
    Args:
        predictions: shape (batch_size, sequence_length)
        labels: shape (batch_size, sequence_length)
        ignore_index: 忽略的标签值，通常是padding的index
    """
    # 将预测和标签移位对齐
    shift_predictions = predictions[..., :-1].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # 展平预测和标签
    flat_predictions = shift_predictions.view(-1)
    flat_labels = shift_labels.view(-1)

    # 创建mask去除padding的影响
    mask = (flat_labels != ignore_index)
    masked_predictions = flat_predictions[mask]
    masked_labels = flat_labels[mask]

    # 计算准确率
    correct = (masked_predictions == masked_labels).sum().item()
    total = mask.sum().item()

    accuracy = correct / total if total > 0 else 0

    return accuracy


def compute_metrics(eval_pred):
    """
    计算评估指标
    Args:
        eval_pred: tuple (logits, labels)
            logits: shape (batch_size, sequence_length, vocab_size)
            labels: shape (batch_size, sequence_length)
    """
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)

    # 计算困惑度
    perplexity = compute_perplexity(logits, labels)

    # 计算准确率
    accuracy = compute_accuracy(predictions, labels)

    return {
        "perplexity": perplexity,
        "accuracy": accuracy
    }


# 创建DataCollator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 加载模型并应用 LoRA
model = AutoModelForCausalLM.from_pretrained(model_local_path)
model = get_peft_model(model, lora_config)

# 使用Trainer进行微调
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)

# 开始微调
result = trainer.train()

# 只在主进程上打印汇总信息和保存模型
if local_rank == 0:
    print_summary(result)
    trainer.save_model('models/Qwen-finetuned')
