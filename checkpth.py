import torch

# 假设你的.pth文件路径是'path_to_your_file.pth'
model_path = './ResNet.pth'

# 加载模型权重
model_weights = torch.load(model_path)

# 打开一个文本文件用于写入
with open('model_weights.txt', 'w') as f:
    # 写入模型权重
    for name, weight in model_weights.items():
        # 将权重转换为numpy数组以便写入文件
        weight_numpy = weight.cpu().detach().numpy().flatten()
        # 写入权重的名称和尺寸
        f.write(f"Layer: {name} | Size: {weight.size()}\n")
        # 写入权重值
        f.write(f"Values: {weight_numpy}\n\n")

print("权重已经被打印并保存到 'model_weights.txt' 文件中。")
