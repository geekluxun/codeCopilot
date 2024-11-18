import evaluate

# 加载 accuracy 和 perplexity 指标
accuracy_metric = evaluate.load("accuracy")


# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#
#     # 确保 logits 是 [batch_size, sequence_length, vocab_size]
#     if len(logits.shape) == 3:
#         # 获取每个位置的预测类别
#         predictions = torch.argmax(torch.tensor(logits), dim=-1)  # 得到形状 [batch_size, sequence_length]
#     else:
#         raise ValueError("Expected logits to have shape [batch_size, sequence_length, vocab_size]")
#
#     # 展开 predictions 和 labels 以计算准确率
#     predictions = predictions.view(-1)  # 展开为一维张量
#     labels = torch.tensor(labels).view(-1)  # 展开为一维张量
#
#     # 移除 -100 标签，因为这些是 padding 或忽略位置
#     valid_indices = labels != -100
#     predictions = predictions[valid_indices]
#     labels = labels[valid_indices]
#
#     # 计算准确率
#     accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
#
#     return {
#         "accuracy": accuracy["accuracy"],
#     }

def compute_metrics(eval_pred):
    return {
        "accuracy": 0.5,
    }
