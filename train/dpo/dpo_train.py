# train_dpo.py
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
import os

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DIR"] = "tmp/wandb_log"
os.environ["WANDB_CACHE_DIR"] = "tmp/wandb_log"

def dpo_train():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct").to("mps")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    full_train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    train_dataset = full_train_dataset.select(range(100))
    # todo 为什么这里的如何不设置max_length，会消耗100G内存都不够？
    training_args = DPOConfig(output_dir="Qwen2-0.5B-DPO", logging_steps=10, report_to=["wandb"], max_length=256,
                              per_device_train_batch_size=8)
    trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
    trainer.train()
    trainer.save_model("Qwen2-0.5B-DPO")


if __name__ == "__main__":
    dpo_train()
