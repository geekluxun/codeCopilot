import os

from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置环境变量以禁用 tokenizers 的并行化
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def generate(model_path: str, prompts: list) -> list:
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 如果模型没有 pad_token_id，可以设置为 eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to("mps")  # 将模型移动到 MPS（Mac 上的 GPU）

    texts = []
    for prompt in prompts:
        # 对每个 prompt 单独编码，并生成输出

        model_inputs = tokenizer(
            prompt,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to("mps")

        # 生成输出
        model_outputs = model.generate(
            **model_inputs,
            max_length=512,
            min_length=10,
            num_beams=5,
            early_stopping=True,
            do_sample=False,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            repetition_penalty=1.0,
            length_penalty=1.0
        )
        # 解码并保存结果
        text = tokenizer.decode(model_outputs[0], skip_special_tokens=True)
        texts.append(text)

    return texts


if __name__ == '__main__':
    generate(model_path="/Users/luxun/workspace/ai/mine/codeCopilot/utils/lora-merged",
             prompts=["how are you !", "my name is"])
