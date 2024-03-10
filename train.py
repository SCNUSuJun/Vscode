# from model import ResNet18
# import torch
# import torch.nn as nn
# from validate import validate
# from tqdm import tqdm


# def train(train_loader, test_loader, writer, epochs, lr, device, model_dict):
#     best_l = 1000
#     model = ResNet18().to(device)
#     optimizer_e = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.MSELoss()
#     for epoch in range(epochs):
#         model.train()
#         train_rmes, train_mae, train_loss = 0., 0., 0.
#         step = 0
#         loader = tqdm(train_loader)
#         for img, label,_ in loader:
#             img, label = img.to(device), label.to(device).to(torch.float32)
#             optimizer_e.zero_grad()
#             score = model(img)
#             loss = criterion(score, label)
#             train_loss += loss.item()
#             rmse = torch.sqrt(torch.pow(torch.abs(score - label), 2).mean()).item()
#             train_rmes += rmse
#             mae = torch.abs(score - label).mean().item()
#             train_mae += mae
#             loss.backward()
#             optimizer_e.step()
#             step += 1
#             loader.set_description("Epoch:{} Step:{} RMSE:{:.2f} MAE:{:.2f}".format(epoch, step, rmse, mae))
#         train_rmes /= step
#         train_mae /= step
#         train_loss /= step
#         model.eval()
#         val_rmes, val_mae, val_loss = validate(model, test_loader, device, criterion)
#         writer.log_train(train_rmes, train_mae, train_loss, val_rmes, val_mae, val_loss, epoch)
#         if val_loss < best_l:
#             torch.save({'ResNet': model.state_dict()}, '{}/ResNet.pth'.format(model_dict))
#             print('Save model!,Loss Improve:{:.2f}'.format(best_l - val_loss))
#             best_l = val_loss
#         print('Train RMSE:{:.2f} MAE:{:.2f} \t Val RMSE:{:.2f} MAE:{:.2f}'.format(train_rmes, train_mae, val_rmes,
#                                                                                    val_mae))
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# from model import ResNet18
# import torch
# import torch.nn as nn
# from validate import validate
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from writer import MyWriter

# print("Starting script...")  # 打印调试信息

# def train(train_loader, test_loader, writer, epochs, lr, device):
#     best_l = 1000
#     model = ResNet18().to(device)
#     print("Model created and moved to device.")  # 打印模型创建信息

#     # 使用Adam优化器，学习率设置为0.001
#     optimizer_e = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.MSELoss()
#     # 设置训练轮数为20
#     for epoch in range(20):
#         model.train()
#         train_rmes, train_mae, train_loss = 0., 0., 0.
#         step = 0
#         loader = tqdm(train_loader)
#         for img, label in loader:
#             img, label = img.to(device), label.to(device).to(torch.float32)
#             optimizer_e.zero_grad()
#             score = model(img)
#             label = label.view(-1, 1)  # 添加这行代码
#             loss = criterion(score, label)
#             train_loss += loss.item()
#             rmse = torch.sqrt(torch.pow(torch.abs(score - label), 2).mean()).item()
#             train_rmes += rmse
#             mae = torch.abs(score - label).mean().item()
#             train_mae += mae
#             loss.backward()
#             optimizer_e.step()
#             step += 1
#             loader.set_description("Epoch:{} Step:{} RMSE:{:.2f} MAE:{:.2f}".format(epoch, step, rmse, mae))
#         train_rmes /= step
#         train_mae /= step
#         train_loss /= step
#         model.eval()
#         val_rmes, val_mae, val_loss = validate(model, test_loader, device, criterion)
#         writer.log_train(train_rmes, train_mae, train_loss, val_rmes, val_mae, val_loss, epoch)
#         if val_loss < best_l:
#             torch.save({'ResNet': model.state_dict()}, './ResNet.pth')
#             print('Save model!,Loss Improve:{:.2f}'.format(best_l - val_loss))
#             best_l = val_loss
#         print('Train RMSE:{:.2f} MAE:{:.2f} \t Val RMSE:{:.2f} MAE:{:.2f}'.format(train_rmes, train_mae, val_rmes,
#                                                                                    val_mae))

# print("Script finished.")  # 打印脚本完成信息

# # 创建数据加载器
# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
# train_dataset = datasets.ImageFolder(root='G:/keep/train', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_dataset = datasets.ImageFolder(root='G:/keep/test', transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # 创建日志记录器
# writer = MyWriter('./logs')

# # 设置训练轮数
# epochs = 20

# # 设置学习率
# lr = 0.001

# # 设置设备
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # 调用train函数
# train(train_loader, test_loader, writer, epochs, lr, device)
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
from model import ResNet18
import torch
import torch.nn as nn
from validate import validate
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from writer import MyWriter

print("Starting script...")  # 打印调试信息

def train(train_loader, test_loader, writer, epochs, lr, device):
    best_l = 1000
    model = ResNet18().to(device)
    print("Model created and moved to device.")  # 打印模型创建信息

    # 使用Adam优化器，学习率设置为0.001
    optimizer_e = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    # 设置训练轮数为20
    for epoch in range(20):
        model.train()
        train_rmes, train_mae, train_loss = 0., 0., 0.
        step = 0
        loader = tqdm(train_loader)
        for img, label in loader:
            img, label = img.to(device), label.to(device).to(torch.float32)
            print(f"Image shape: {img.shape}, Label shape: {label.shape}")  # 打印图像和标签的形状
            optimizer_e.zero_grad()
            score = model(img)
            print(f"Score shape: {score.shape}")  # 打印模型输出的形状
            label = label.view(-1, 1)  # 添加这行代码
            loss = criterion(score, label)
            train_loss += loss.item()
            rmse = torch.sqrt(torch.pow(torch.abs(score - label), 2).mean()).item()
            train_rmes += rmse
            mae = torch.abs(score - label).mean().item()
            train_mae += mae
            loss.backward()
            optimizer_e.step()
            step += 1
            loader.set_description("Epoch:{} Step:{} RMSE:{:.2f} MAE:{:.2f}".format(epoch, step, rmse, mae))
        train_rmes /= step
        train_mae /= step
        train_loss /= step
        model.eval()
        val_rmes, val_mae, val_loss = validate(model, test_loader, device, criterion)
        writer.log_train(train_rmes, train_mae, train_loss, val_rmes, val_mae, val_loss, epoch)
        if val_loss < best_l:
            torch.save({'ResNet': model.state_dict()}, './ResNet.pth')
            print('Save model!,Loss Improve:{:.2f}'.format(best_l - val_loss))
            best_l = val_loss
        print('Train RMSE:{:.2f} MAE:{:.2f} \t Val RMSE:{:.2f} MAE:{:.2f}'.format(train_rmes, train_mae, val_rmes,
                                                                                   val_mae))

print("Script finished.")  # 打印脚本完成信息

# 创建数据加载器
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.ImageFolder(root='G:/keep/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = datasets.ImageFolder(root='G:/keep/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 创建日志记录器
writer = MyWriter('./logs')

# 设置训练轮数
epochs = 20

# 设置学习率
lr = 0.001

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 调用train函数
train(train_loader, test_loader, writer, epochs, lr, device)