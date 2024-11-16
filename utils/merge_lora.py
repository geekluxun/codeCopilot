from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora(base_model_id, adapter_id, merged_model_path):
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    base_with_adapters_model = PeftModel.from_pretrained(base_model, adapter_id)
    merged_model = base_with_adapters_model.merge_and_unload()
    merged_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)


if __name__ == '__main__':
    base_model_id = "/Users/luxun/workspace/ai/hf/models/Qwen1.5-0.5B"
    adapter_id = "/Users/luxun/workspace/ai/mine/codeCopilot/train/models/Qwen-finetuned"
    merged_model_path = "lora-merged"
    merge_lora(base_model_id, adapter_id, merged_model_path)
