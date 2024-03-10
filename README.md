### 项目介绍

ResNet网络的应用—抑郁症诊断

使用数据集：**AVEC2014**

预处理：

​	1.**采样**，AVEC2013每个视频取100帧，保留原始label

​	2.**人脸对齐裁剪**，使用**MTCNN**工具

### 文件介绍

```
preprocess.py	主要用于预处理视频信息，从中提取帧，并在视频帧中提取人脸
函数：generate_label_file()	将运来的label合并为一个csv文件
函数：get_img()	抽取视频帧，每个视频按间隔抽取100-105帧
函数：get_face()	使用MTCNN提取人脸，并分割图片

model.py	模型的网络结构
```

```
load_data.py	获取图片存放路径以及将标签与之对应
writer.py	创建Tensorboard记录器，保存训练过程损失
dataset.py	继承torch.utils.Dataset,负责将数据转化为torch.utils.data.DataLoader可以处理的迭代器
train.py	模型训练
validate.py	验证模型
test.py		测试模型的性能，并记录预测分数，保存在testInfo.csv,记录了每张图片的路径，label,预测分数
main.py		模型训练入口文件
```

```
img		提取的视频帧文件
log		Tensorboard日志文件
model_dict	训练好的模型参数文件
processed	存放预处理完成之后的人脸图片，label文件
AVEC2014	数据集存放位置
```

```
查看训练日志方法：
	安装tensorboard库之后，输入命令tensorboard --lofdir log_dir_path,打开命令执行后出现的网址即可
	log_dir_path是存放Tensorboard日志文件的文件夹路径
```

```
运行顺序:preprocess.py--->main.py--->test.py
```







-------

--------

# ==Model==

### Import 语句
```python
import torch.nn as nn
```
- 这行代码导入了 PyTorch 库的神经网络模块 `nn`，它提供了构建神经网络所需的各种功能和类。

### BasicBlock 类
```python
class BasicBlock(nn.Module):
    expansion = 1
```
- `BasicBlock` 类继承自 `nn.Module`，是构建 ResNet 的基础模块。
- `expansion` 是一个类属性，用于控制输出通道数的扩展。

```python
def __init__(self, in_channels, out_channels, stride=1):
    super().__init__()
```
- 这是 `BasicBlock` 类的初始化函数。
- `in_channels` 和 `out_channels` 分别表示输入和输出的通道数。
- `stride` 控制卷积操作的步长，默认为 1。

```python
self.residual_function = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
    nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(out_channels * BasicBlock.expansion)
)
```
- `self.residual_function` 定义了残差块中的卷积层序列。
- 包含两个 `nn.Conv2d` 卷积层，用于处理图像数据。
- `nn.BatchNorm2d` 用于批量归一化，有助于加快训练速度和减少过拟合。
- `nn.ReLU` 是激活函数，用于引入非线性。

```python
self.shortcut = nn.Sequential()
```
- 初始化一个空的快捷连接（用于直接连接输入和输出，当输入和输出形状不一致时使用）。

```python
if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
    self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels * BasicBlock.expansion)
    )
```
- 如果步长不是 1 或输入通道数与输出通道数不匹配，则重定义 `self.shortcut` 为包含卷积和批量归一化的序列。

```python
def forward(self, x):
    return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
```
- 定义了前向传播函数 `forward`。
- 将输入 `x` 通过残差函数和快捷连接处理后相加，并通过 ReLU 激活函数。

### BottleNeck 类
- `BottleNeck` 类的结构与 `BasicBlock` 相似，但包含更多层来构建更深的网络。

### ResNet 类
```python
class ResNet(nn.Module):

    def __init__(self, block,

num_blocks):
        super().__init__()
```
- `ResNet` 类也继承自 `nn.Module`，是整个残差网络的主体结构。
- `block` 参数用于指定网络使用的基础块类型（例如 `BasicBlock` 或 `BottleNeck`）。
- `num_blocks` 是一个列表，指定了在每个阶段中应重复基础块的次数。

```python
self.in_channels = 64
```
- 初始化输入通道数为 64。

```python
self.conv1 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True))
```
- `self.conv1` 定义了网络的第一个卷积层。
- `nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)` 表示一个卷积层，从 3 个输入通道转换到 64 个输出通道，核大小为 3，边缘填充为 1。

```python
self.conv2_x = self._make_layer(block, 64, num_blocks[0], 1)
self.conv3_x = self._make_layer(block, 128, num_blocks[1], 2)
self.conv4_x = self._make_layer(block, 256, num_blocks[2], 2)
self.conv5_x = self._make_layer(block, 512, num_blocks[3], 2)
```
- 这些行定义了四个不同阶段的卷积层组，每个阶段利用 `_make_layer` 方法构建，其中包含重复的基础块。

```python
self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
```
- 定义了一个自适应平均池化层，输出大小为 1x1，用于降低特征维度。

```python
self.fc = nn.Linear(512 * block.expansion, 1)
```
- 定义了一个全连接层（线性层），用于从特征映射到最终的输出。

### _make_layer 方法
```python
def _make_layer(self, block, out_channels, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
    return nn.Sequential(*layers)
```
- `_make_layer` 是一个辅助方法，用于创建包含多个基础块的层。
- `strides` 列表决定了每个块的步长。

### forward 方法
```python
def forward(self, x):
    output = self.conv1(x)
    output = self.conv2_x(output)
    output = self.conv3_x(output)
    output = self.conv4_x(output)
    output = self.conv5_x(output)
    output = self.avg_pool(output)
    output = output.view(output.size(0), -1)
    output = self.fc(output)

    return output
```
- `forward` 方法定义了数据在 ResNet 模型中的流动方式。
- 依次通过初始卷积层、四个阶段的卷积层组、平均池化层，最后通过全连接层得到最终输出。

### ResNet18 函数
```python
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
```
- `ResNet18` 是一个工厂函数，用于创建一个具有 18 层的 ResNet 模型（使用 `BasicBlock`）。

整个代码定义了一个具有残差连接的深度卷积神经网络，适合于图像识别和分类任务。通过残差连接，这种网络能够在增加深度的同时避免梯度消失和爆炸的问题，提高了训练效率。

---------

当然，我会对上述代码中的类进行更详细的解释。

### BasicBlock 类

1. `class BasicBlock(nn.Module)`:
   - `BasicBlock` 是一个构建块，用于构建较深的神经网络。
   - 继承自 `nn.Module`，这是 PyTorch 中所有神经网络模块的基类。

2. `expansion = 1`:
   - 这是一个类级别的属性，用于扩展输出通道数。在 `BasicBlock` 中，这个值设置为 1，意味着输出通道数不变。

3. `def __init__(self, in_channels, out_channels, stride=1)`:
   - 这是类的构造函数。它接收输入通道数 (`in_channels`)，输出通道数 (`out_channels`)，以及卷积层的步长 (`stride`)。

4. `self.residual_function`:
   - 这个属性是一个 `nn.Sequential` 对象，表示残差块中的一系列操作。
   - 包含两个卷积层 (`nn.Conv2d`)，两个批量归一化层 (`nn.BatchNorm2d`)，和一个激活函数 (`nn.ReLU`)。

5. `self.shortcut`:
   - 这是残差块的“快捷连接”，用于在必要时调整输入以匹配残差函数的输出。

6. `if stride != 1 or in_channels != BasicBlock.expansion * out_channels`:
   - 当步长不为 1 或输入和输出通道数不匹配时，需要调整快捷连接。

7. `def forward(self, x)`:
   - 这是模块的前向传播函数。它将输入数据 `x` 通过残差函数和快捷连接处理，然后将它们相加，并通过 ReLU 激活函数。

### BottleNeck 类

- 这个类的结构与 `BasicBlock` 类似，但它包含更多的层（三个卷积层），用于构建更复杂的模型。
- `expansion` 在此类中设置为 4，意味着最后一个卷积层的输出通道数是输入通道数的四倍。

### ResNet 类

1. `class ResNet(nn.Module)`:
   - 这是主要的神经网络模型，表示整个残差网络。

2. `def __init__(self, block, num_blocks)`:
   - 构造函数接收 `block` 参数（`BasicBlock` 或 `BottleNeck`）和 `num_blocks` 列表，后者指定每个阶段重复块的次数。

3. `self.in_channels = 64`:
   - 初始化输入通道数为 64。

4. `self.conv1`, `self.conv2_x`, `self.conv3_x`, `self.conv4_x`, `self.conv5_x`:
   - 这些是网络中的主要卷积层。第一个卷积层 (`conv1`) 用于初始特征提取。后面四个 (`conv2_x` 到 `conv5_x`) 按顺序构成网络的主体，每个都使用 `_make_layer` 函数来构建。

5. `self.avg_pool` 和 `self.fc`:
   - `avg_pool` 是平均池化层，用于减小特征映射的维度。
   - `fc` 是全连接层，用于最终的分类或回归任务。

6. `def _make_layer(self, block, out_channels, num_blocks, stride)`:
   - 这个辅助方法用于创建含有指定数量基础块的层。

7. `def forward(self, x)`:
   - 前向传播函数定义了数据通过网络的流程。

### ResNet18 函数

- `def ResNet18()`: 这是一个构造函数，用于创建一个具有 18 层（使用 `BasicBlock`）的 ResNet 模型。

总的来说，这段代码定义了一个复杂的深

度学习模型，使用残差连接来提高深层网络的训练效果。每个类都是模型的一个组成部分，共同工作以处理图像识别和分类任务。








通过初步阅读这些文件的内容，我们可以大致了解它们之间的关系和各自的功能。下面是对每个文件的功能和它们相互之间联系的分析：

### 1. dataset.py
- `dataset.py` 定义了一个名为 `MyDataset` 的类，这个类继承自 PyTorch 的 `Dataset` 类。它用于加载和预处理数据集。
- 这个类的 `__init__` 方法接收图像路径和标签路径，调用 `load` 函数从 `load_data.py` 文件中加载数据。
- 数据集被应用了一系列的变换（如调整大小、转换为张量、标准化等）。

### 2. writer.py
- `writer.py` 定义了 `MyWriter` 类，用于记录训练和验证过程中的各种指标，如 RMSE、MAE 和 LOSS。
- 它使用了 `tensorboardX` 的 `SummaryWriter`，这是用于 TensorBoard 可视化的工具。

### 3. validate.py
- `validate.py` 包含一个 `validate` 函数，用于评估模型在测试集上的性能。
- 它接收模型、测试数据加载器、设备和损失函数作为参数，并返回模型在测试集上的性能指标。

### 4. train.py
- `train.py` 包含一个 `train` 函数，用于训练模型。
- 它接收训练和测试数据加载器、`writer` 对象、训练配置（如迭代次数、学习率等）和设备。
- 在训练过程中，它调用 `validate` 函数来评估模型性能，并使用 `MyWriter` 记录性能指标。

### 5. test.py
- `test.py` 用于执行一些测试过程。
- 这个文件的具体内容没有展示，但通常包含代码来测试特定功能或模型性能。

### 6. preprocess.py
- `preprocess.py` 包含预处理相关的函数，如文件和目录的获取和标签文件的生成。

### 7. model.py
- `model.py` 定义了神经网络模型，包括 `BasicBlock` 和 `BottleNeck` 类，以及基于这些块构建的 `ResNet` 模型。
- 这些类被 `train.py` 和其他可能需要模型架构的文件调用。

### 8. load_data.py
- `load_data.py` 包含 `load` 函数，用于加载图像路径和标签。
- 它被 `dataset.py` 中的 `MyDataset` 类调用以加载和准备数据。

### 9. main.py


- `main.py` 是整个程序的入口点。它整合了其他所有模块，用于实际运行模型的训练和验证过程。
- 它导入并使用 `MyWriter` 类来初始化日志记录。
- 使用 `MyDataset` 类和 `DataLoader` 来加载和准备训练数据。
- 它还导入并调用 `train` 函数，传入所需的参数，如训练数据加载器、学习率、训练迭代次数等，以及模型保存路径和日志路径。

### 相互之间的联系和运行流程：
1. **数据加载与预处理**：
   - `load_data.py` 和 `preprocess.py` 负责数据的加载和预处理。`load_data.py` 通过 `load` 函数加载数据，`preprocess.py` 提供预处理相关的辅助功能。

2. **数据集定义**：
   - `dataset.py` 利用上述两个文件定义了一个 PyTorch 数据集，用于在训练和验证中使用。

3. **模型定义**：
   - `model.py` 包含网络模型的定义，这是训练和验证所需的核心部分。

4. **训练和验证**：
   - `train.py` 和 `validate.py` 分别负责模型的训练和验证。`train.py` 在训练过程中调用 `validate.py` 来评估模型。

5. **日志记录**：
   - `writer.py` 提供了记录训练和验证过程中各种性能指标的功能。

6. **主程序**：
   - `main.py` 将所有这些组件组合在一起，实际执行模型的训练和验证过程。

这样的结构提供了良好的模块化和代码组织，每个文件负责特定的任务，易于维护和扩展。通过 `main.py`，这些模块被统一调用，形成了一个完整的训练和验证流程。

如果您需要更具体的分析或对特定文件的更详细解释，请告知。








暂存：
""" 
这段代码是一个数据预处理脚本，用于处理视频数据集，并将其转换成适合深度学习模型训练的格式。它主要执行以下三个任务：生成标签文件、从视频中提取图像帧，以及使用 MTCNN 提取图像中的人脸。我们将对代码中的每个函数进行详细分析。

### 1. `get_files` 和 `get_dirs` 函数
这两个函数用于从指定路径中提取文件和目录列表。

- **get_files**: 返回指定路径下的所有文件。
- **get_dirs**: 返回指定路径下的所有目录（不包括根目录）。

### 2. `generate_label_file` 函数
此函数用于从多个CSV文件中提取标签，并将其合并成一个单独的CSV文件。

- **步骤**:
  - 遍历指定路径下的所有CSV文件。
  - 从每个文件中读取标签，并将文件名（去除特定后缀）和标签存储在一个列表中。
  - 列表被转换成DataFrame并保存为一个新的CSV文件。

### 3. `generate_img` 函数
这个函数用于从视频中提取固定数量的帧，并将它们保存为图像文件。

- **步骤**:
  - 遍历指定路径下的所有视频文件。
  - 对于每个视频，计算要提取的帧的间隔（以确保总共提取约100帧）。
  - 读取并保存指定间隔的帧作为JPEG图像。

### 4. `get_img` 函数
这个函数是`generate_img`的封装，用于处理不同类别（如 "Freeform" 和 "Northwind"）和不同数据集（如训练集、测试集和验证集）的视频。

### 5. `get_face` 函数
此函数使用 MTCNN 模型来检测并提取图像中的人脸。

- **步骤**:
  - 遍历指定路径下的所有图像文件。
  - 对每个图像使用 MTCNN 进行人脸检测。
  - 如果检测到人脸，提取人脸区域并保存为新的图像文件。

### 6. 脚本的主体
- 在主函数中，首先创建必要的目录。
- 然后按顺序调用 `generate_label_file`、`get_img` 和 `get_face` 函数，以执行整个数据预处理流程。

### 关键点和注意事项
- **数据处理流程**: 从原始视频到人脸图像的整个处理流程是自动化的，可以处理大量数据。
- **MTCNN 人脸检测**: MTCNN 是一种有效的人脸检测方法，用于提取图像中的人脸区域。
- **数据集组织**: 脚本按数据集类型（训练、测试、验证）和视频类型（"Freeform"、"Northwind"）组织数据。
- **性能考虑**: 从视频中提取图像帧和人脸检测可能是计算密集型的。使用 `tqdm` 可以帮助监控进度。

总的来说，这个脚本为机器学习或深度学习项目中的数据准备阶段提供了一个全面的工作流程。它涉及从原始视频数据中提取有用的信息（如图像帧和人脸），并将其转换成适合模型训练的格式。这个过程是许多计算机视觉项目的关键部分，特别是在处理涉及人脸识别或情感分析等领域的项目时。

### 实用性和扩展性
- 脚本中使用的函数和处理流程具有较高的通用性，可适用于不同的视频处理和图像分析任务。
- 对于具有不同需求的项目，这些函数可以轻松地进行调整或扩展。

### 性能和优化
- 在处理

大量数据或高分辨率视频时，特别是在使用 MTCNN 进行人脸检测时，脚本可能会消耗大量的计算资源。优化这些操作，例如通过调整帧提取频率或使用更高效的人脸检测算法，可以提高处理速度并减少资源消耗。
- 在多核处理器或高性能计算环境中，可以考虑并行处理或批处理来进一步提高效率。

### 总结
这个脚本提供了一个从原始视频数据到准备好的训练数据的完整路径，涵盖了数据抽取、预处理和准备的多个关键步骤。它是理解和实施视频数据处理流程的一个很好的示例，特别是对于涉及人脸识别和情感分析的项目。
 """


