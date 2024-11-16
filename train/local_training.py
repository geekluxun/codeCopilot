import math
import os

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from scipy.special import softmax
from torch.multiprocessing import freeze_support
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, DataCollatorWithPadding
from transformers import Trainer

# local modules
from eval.measure import compute_metrics
from utils.monitor_util import ResourceMonitor

# 使用数据集的百分比
use_datasets_percentage = 0.1
# 验证集的百分比
use_test_datasets_percentage = 0.01
# 序列长度
sequence_max_length = 512
# model_local_path = '/Users/luxun/workspace/ai/hf/models/Qwen1.5-0.5B'
model_local_path = '/Users/luxun/workspace/ai/mine/codeCopilot/utils/lora-merged'
dataset_local_path = '/Users/luxun/workspace/ai/hf/datasets/CodeAlpaca-20k'
logging_dir = "output/logs/tensorboard"
saved_model_path = 'output/models/CodeCopilot'
checkpoint_path = 'output/checkpoints/'
project_name = "CodeCopilot"

# 环境设置
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DIR"] = "wandb_log"
os.environ["WANDB_CACHE_DIR"] = "wandb_log"


# 配置设备
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# 可以预处理logits，部分逻辑可以compute_metrics函数中移到这里
def preprocess_logits_for_metrics(logits, labels):
    pass


def train_model():
    device = get_device()
    print(f"Using device: {device}")

    # 启动资源监控
    monitor = ResourceMonitor(interval=30)
    monitor.start()

    # 加载tokenizer和数据集
    tokenizer = AutoTokenizer.from_pretrained(model_local_path)
    dataset = load_dataset(dataset_local_path)
    # 仅使用部分数据进行测试，比如20%
    train_size = int(len(dataset["train"]) * use_datasets_percentage)
    dataset["train"] = dataset["train"].select(range(train_size))
    train_val_dataset = dataset["train"].train_test_split(test_size=use_test_datasets_percentage, shuffle=True, seed=42)

    def preprocess_data_function(examples):
        prompts = []
        for inst, inp in zip(examples["instruction"], examples["input"]):
            if inp:
                prompt = f"Instruction: {inst}\nInput: {inp}"
            else:
                prompt = f"Instruction: {inst}"
            prompts.append(prompt)

        model_inputs = tokenizer(
            prompts,
            max_length=sequence_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = tokenizer(
            examples["output"],
            max_length=sequence_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 将 labels 中填充值的部分设置为 -100
        labels["input_ids"][labels["attention_mask"] == 0] = -100
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    # 预处理数据集
    train_dataset = train_val_dataset["train"].map(preprocess_data_function, batched=True)
    val_dataset = train_val_dataset["demo"].map(preprocess_data_function, batched=True)
    print(f"train_dataset size: {len(train_dataset)}, val_dataset size: {len(val_dataset)}")
    # LoRA配置
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 为Mac优化的训练参数
    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,  # 降低批次大小以适应内存
        gradient_accumulation_steps=100,  # 增加梯度累积步数
        per_device_eval_batch_size=8,
        eval_accumulation_steps=1,  # 增加评估累积步数
        learning_rate=3e-5,
        evaluation_strategy="steps",
        eval_steps=1,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=3,
        logging_dir=logging_dir,
        log_level="debug",
        logging_steps=1,
        warmup_steps=100,
        report_to=["tensorboard", "wandb"],
        fp16=False,  # Mac上禁用混合精度
        dataloader_num_workers=0,  # 减少工作线程
        remove_unused_columns=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        # 性能优化
        ddp_find_unused_parameters=False,
        weight_decay=0.3,
        run_name=project_name
    )

    log_train_arg(training_args, len(train_dataset))

    wandb.init(
        project=project_name,
        mode="offline",
        config=training_args.to_dict(),
        allow_val_change=True,
        reinit=True
    )

    # 加载模型
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_local_path,
        torch_dtype=torch.float32,  # 使用float32而不是float16
        device_map='auto'
    )

    model = get_peft_model(model, lora_config).to(device)

    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    # 开始训练

    print("Starting training...")
    result = trainer.train(resume_from_checkpoint=checkpoint_path)
    # 打印训练结果
    print("\nTraining completed!")
    print(f"Total training time: {result.metrics['train_runtime']:.2f} seconds")
    print(f"Training samples/second: {result.metrics['train_samples_per_second']:.2f}")

    # 打印最终评估结果
    final_metrics = trainer.evaluate()
    print("\nFinal evaluation metrics:", final_metrics)
    for key, value in final_metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # 保存模型
    print("Saving model...")
    trainer.save_model(saved_model_path)
    print("Training completed!")


def log_train_arg(training_args: TrainingArguments, train_dataset_size: int):
    # 计算总训练步数
    total_train_steps = math.ceil(
        train_dataset_size
        * training_args.num_train_epochs
        / (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    )

    # 打印训练配置
    print(f"\nTraining Configuration:")
    print(f"Number of epochs: {training_args.num_train_epochs}")
    print(f"Batch size per device: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(
        f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Total training steps: {total_train_steps}")
    print(f"Warmup steps: {training_args.warmup_steps}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Weight decay: {training_args.weight_decay}")
    print(f"Evaluation every {training_args.eval_steps} steps")
    print(f"Logging every {training_args.logging_steps} steps")


def get_most_probable_sequence(logits, tokenizer, labels):
    """
    从logits中获取第一个样本的最可能序列，并转换成文本

    参数:
        logits: shape (batch_size, seq_len, vocab_size) 的模型输出
        tokenizer: 使用的分词器

    返回:
        predicted_text: 解码后的文本
        token_probs: 每个位置预测token的概率
    """
    import numpy as np
    # 1. 只取第一个样本的logits
    sample_logits = logits  # shape: (seq_len, vocab_size)

    # 2. 对每个位置取最大概率的token
    token_ids = np.argmax(sample_logits, axis=-1)  # shape: (seq_len,)

    # 3. 计算每个位置预测token的概率
    probs = softmax(sample_logits, axis=-1)  # shape: (seq_len, vocab_size)
    token_probs = np.max(probs, axis=-1)  # shape: (seq_len,)

    # 4. 将token ids转换为文本
    predicted_text = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    label_text = [tokenizer.decode(label[label != -100], skip_special_tokens=True) for label in labels]
    return predicted_text, token_probs, label_text


# 测试prodict
def predict_test(trainer, tokenizer, preprocess_data_function):
    val_dataset2 = load_dataset("/Users/luxun/workspace/ai/mine/codeCopilot/data2")
    dataset2 = val_dataset2["train"].map(preprocess_data_function, batched=True)
    predictions = trainer.predict(dataset2)
    print(predictions.predictions.shape, predictions.label_ids.shape)
    get_most_probable_sequence(predictions.predictions, tokenizer, predictions.label_ids)


if __name__ == '__main__':
    freeze_support()
    train_model()
