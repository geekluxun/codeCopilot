import evaluate
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from pynvml import *
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM, DataCollatorWithPadding
from transformers import EarlyStoppingCallback

# 本地模型路径
model_local_path = 'models/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct/'
# 本地数据集路径
dataset_local_path = 'data/CodeAlpaca-20k/'
# 最长序列
max_length = 1024  # 可以根据实际情况调整这个值

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_local_path)
# 加载数据集
dataset = load_dataset(dataset_local_path)
train_val_dataset = dataset["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["prompt"],
        max_length=max_length,
        padding="longest",
        truncation=True,
        return_tensors="pt"  # 直接返回PyTorch张量
    )

    labels = tokenizer(
        examples["completion"],
        max_length=max_length,
        padding="longest",
        truncation=True,
        return_tensors="pt"
    )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


# 应用预处理
train_dataset = train_val_dataset["train"].map(preprocess_function, batched=True)
val_dataset = train_val_dataset["test"].map(preprocess_function, batched=True)

# lora配置
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,  # LoRA scaling factor
    target_modules=["q_proj", "v_proj"],  # 目标模块名称
    lora_dropout=0.1,  # LoRA dropout
    bias="none",  # 偏置设置
    task_type="CAUSAL_LM"  # 任务类型
)
# 创建早停回调
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,  # 3轮不提升就停止
    early_stopping_threshold=0.01  # 提升需超过0.01才算改善
)

# 训练参数
training_args = TrainingArguments(
    num_train_epochs=3,  # 训练轮数
    learning_rate=1e-5,  #
    lr_scheduler_type='linear',
    warmup_ratio=0.1,  # 10%步数用于预热
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,  # L2正则化
    # 优化
    gradient_checkpointing=False,  # 反向时候重新计算激活，减少内存
    fp16=True,  # 混合精度,减少内存
    gradient_accumulation_steps=2,  # 默认是每一step都会计算梯度，为了减少内存，可以多个step合并计算，减少内存占用
    # 日志相关
    log_level='debug',
    logging_dir='./logs',
    logging_steps=500,
    # 评估相关
    evaluation_strategy="steps",
    eval_steps=500,
    metric_for_best_model="eval_loss",  # 用损失评估
    # checkpoint相关
    output_dir='./results',
    save_strategy="steps",  # checkpoint保存策略
    save_steps=2_000,
    save_total_limit=2,
    load_best_model_at_end=True,
    # 其他
    remove_unused_columns=True,  # 是否删除未使用的列
    dataloader_drop_last=True,  # 默认是False，当最后一个batch的样本数少于batch_size时，是否丢弃该batch
    disable_tqdm=False,  # 是否禁用tqdm进度条

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


# 加载困惑度和准确率指标
perplexity_metric = evaluate.load("perplexity")
accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    # logits形状为 [batch_size, sequence_length, vocab_size]
    # labels和predictions的形状为 [batch_size, sequence_length]

    # 模型的原始输出（各个词打分），标签
    logits, labels = eval_pred
    # 各个词中的最高分就是预测值
    predictions = torch.argmax(logits, dim=-1)

    # 预测左移一位去掉最后一个（输入的最后一个不需要预测）
    shift_logits = logits[..., :-1, :].contiguous()
    # 标签右移1位去掉第一个
    shift_labels = labels[..., 1:].contiguous()
    # 预测左移一位去掉最后一
    shift_predictions = predictions[..., :-1].contiguous()

    # 计算困惑度 需要知道模型对每个词的确信度，概率分布来计算，所以是shift_logits（各个词的分数）
    perplexity_results = perplexity_metric.compute(predictions=shift_logits, references=shift_labels)

    # 计算准确率 只需要知道预测值对不对 view(-1)表示展平成一维且维度大小自动计算,所以是shift_predictions
    accuracy_results = accuracy_metric.compute(predictions=shift_predictions.view(-1), references=shift_labels.view(-1))

    return {
        "perplexity": perplexity_results["perplexity"],
        "accuracy": accuracy_results["accuracy"]
    }


# 创建DataCollator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 应用 LoRA
model = AutoModelForCausalLM.from_pretrained(model_local_path).to("cuda")
model = get_peft_model(model, lora_config)

# 使用Trainer进行微调
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 开始微调
result = trainer.train()
print_summary(result)

# 保存微调后的模型
trainer.save_model('model/Qwen1.5-0.5B-finetuned')
