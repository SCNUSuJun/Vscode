from writer import MyWriter
import os
from dataset import MyDataset
from torch.utils.data import DataLoader
from train import train
import torch

batch_size = 128
lr = 0.001
epochs = 300
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_dir = './model_dict'
log_dir = './log'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
writer = MyWriter(log_dir)

dataset_train = MyDataset('./processed/train/', './processed/label.csv')
train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True,
                          drop_last=True)

dataset_test = MyDataset('./processed/validate/', './processed/label.csv')
test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

train(train_loader, test_loader, writer, epochs, lr, device, model_dir)

#注释
""" 

当然，以下是对 `main.py` 文件中的每一行代码的解释：

1. `from writer import MyWriter`：从 `writer` 模块中导入 `MyWriter` 类，这个类可能用于记录训练过程。

2. `import os`：导入 `os` 模块，这个模块提供了许多与操作系统交互的函数。

3. `from dataset import MyDataset`：从 `dataset` 模块中导入 `MyDataset` 类，这个类可能用于处理数据集。

4. `from torch.utils.data import DataLoader`：从 PyTorch 库中导入 `DataLoader` 类，这个类用于批量加载数据。

5. `from train import train`：从 `train` 模块中导入 `train` 函数，这个函数可能用于训练模型。

6. `import torch`：导入 PyTorch 库，这是一个用于深度学习的库。

7-10. `batch_size = 128`，`lr = 0.001`，`epochs = 300`，`device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`：设置训练参数，包括批次大小、学习率、训练周期和设备。

11-12. `model_dir = './model_dict'`，`log_dir = './log'`：设置模型和日志的目录。

13-14. `os.makedirs(log_dir, exist_ok=True)`，`os.makedirs(model_dir, exist_ok=True)`：如果目录不存在，就创建它们。

15. `writer = MyWriter(log_dir)`：初始化 `MyWriter`，并将日志目录作为参数传入。

16-18. 创建训练数据集和数据加载器。`MyDataset` 接收数据和标签的路径作为输入，`DataLoader` 批量加载数据，并提供其他功能，如数据洗牌和多线程加载。

20-22. 创建测试数据集和数据加载器，方法与创建训练数据集和数据加载器相同。

23. `train(train_loader, test_loader, writer, epochs, lr, device, model_dir)`：调用 `train` 函数，开始训练模型。这个函数接收训练和测试的数据加载器、`MyWriter`、训练周期、学习率、设备和模型目录作为参数。


 """




'''

这段代码是一个深度学习训练流程的高级概述，使用了PyTorch框架。我们可以按照代码中的各个部分进行详细分析。

### 1. 导入必要的模块和类
- `MyWriter`: 自定义的日志记录类，可能用于记录训练过程中的信息。
- `os`: 用于处理文件和目录的标准库。
- `MyDataset`: 自定义的数据集类，用于加载和预处理数据。
- `DataLoader`: PyTorch中用于加载数据集的类，可以实现批处理、打乱数据、多线程等操作。
- `train`: 自定义的训练函数，负责模型的训练过程。
- `torch`: PyTorch框架的主模块。

### 2. 设置训练参数
- `batch_size = 128`: 每个批次的样本数。
- `lr = 0.001`: 学习率，影响模型在训练过程中的参数更新速度。
- `epochs = 300`: 训练的总轮数。
- `device`: 确定训练使用的设备，优先使用CUDA（GPU），如果不可用，则使用CPU。

### 3. 目录设置和日志记录
- 创建用于存储模型和日志的目录。
- 初始化 `MyWriter` 实例，用于日志记录。

### 4. 数据加载
- **训练集**:
  - 使用 `MyDataset` 加载训练数据。
  - 使用 `DataLoader` 创建一个数据加载器，设定批量大小、是否打乱数据、是否使用内存钉扎（提高数据加载效率）和是否在数据不足一个批量时丢弃数据。

- **测试集**:
  - 类似于训练集，但加载的是验证数据。

### 5. 调用训练函数
- 调用 `train` 函数，传入训练和测试的数据加载器、日志记录器、训练轮数、学习率、设备和模型保存目录。

### 代码运行流程
1. **初始化**: 设置训练参数和环境，准备日志记录工具。
2. **数据准备**: 加载训练和测试数据，准备数据加载器。
3. **训练过程**: 调用 `train` 函数，开始训练模型。在这个过程中，模型将在训练集上学习，并在测试集上验证其性能。
4. **日志记录和模型保存

**: 日志记录器 `writer` 可能会在 `train` 函数中被用来记录训练过程中的关键信息（如损失、准确率等）。模型可能会在每个epoch后或在达到特定性能标准时保存到 `model_dir` 指定的目录。

### 关键点和可能的注意事项
- **自定义类和函数**: `MyWriter`, `MyDataset`, 和 `train` 函数是自定义的，所以它们的具体实现和功能取决于其他代码部分。了解这些组件的具体实现对于理解整个训练过程至关重要。
- **数据集路径**: `MyDataset` 使用指定的路径加载数据和标签。数据的组织方式和预处理步骤也是理解这段代码的关键。
- **数据加载器的参数**: `DataLoader` 的参数如 `shuffle`、`pin_memory` 和 `drop_last` 对于数据加载的效率和训练过程有重要影响。
- **设备选择**: 代码自动选择使用GPU或CPU进行训练，这对于资源利用和训练速度是一个重要考虑。
- **训练函数的实现**: 最关键的部分是 `train` 函数，它控制了整个模型的训练过程，包括前向传播、损失计算、反向传播和参数更新。

您对代码的哪个部分还有疑问，或者需要更多的解释吗？我可以帮助您深入理解特定部分的实现和原理。

'''