import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_cfc_PRDFA import Cfc
from data2 import trainX1, trainX2, trainY
import time
import numpy as np
import random
# from spatial_data import trainX1, trainX2, trainY
# from validate_data import trainX1, trainX2, trainY

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # ensure deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)  # set PYTHONHASHSEED environment variable for reproducibility

fix_seed(2020)

CFC = {
    "epochs": 100,
    "clipnorm": 0,
    "hidden_size": 64,
    "base_lr": 0.004,
    "decay_lr": 0.95,
    "backbone_activation": "silu",
    "backbone_units": 128,
    "backbone_layers": 2,
    "backbone_dr": 0.3,
    "weight_decay": 0.0001,
    "tau": 10,
    "batch_size": 256,
    "optim": "adam",
    "init": 0.84,
    "in_seq": 8,
    "in_f": 16,
    "use_mixed": False,
    "minimal": False,
    "no_gate": False,
    "period_len": [8],
}
CFC_MIXED = {
    "epochs": 100,
    "clipnorm": 0,
    "hidden_size": 64,
    "base_lr": 0.004,
    "decay_lr": 0.95,
    "backbone_activation": "silu",
    "backbone_units": 32,
    "backbone_layers": 2,
    "backbone_dr": 0.3,
    "weight_decay": 0.0001,
    "tau": 10,
    "batch_size": 512,
    "optim": "adam",
    "init": 0.84,
    "in_seq": 8,
    "in_f": 16,
    "use_mixed": True,
    "minimal": False,
    "no_gate": False,
    "period_len": [8],
}
# 5 0.97
CFC_NOGATE = {
    "epochs": 100,
    "clipnorm": 0,
    "hidden_size": 64,
    "base_lr": 0.004,
    "decay_lr": 0.97,
    "backbone_activation": "silu",
    "backbone_units": 64,
    "backbone_layers": 2,
    "backbone_dr": 0.3,
    "weight_decay": 0.0001,
    "tau": 10,
    "batch_size": 512,
    "optim": "adam",
    "init": 0.84,
    "in_seq": 8,
    "in_f": 16,
    "use_mixed": False,
    "minimal": False,
    "no_gate": True,
    "period_len": [8],
}
CFC_MINIMAL = {
    "epochs": 100,
    "clipnorm": 0,
    "hidden_size": 64,
    "base_lr": 0.004,
    "decay_lr": 0.97,
    "backbone_activation": "silu",
    "backbone_units": 128,
    "backbone_layers": 2,
    "backbone_dr": 0.1,
    "weight_decay": 0.0001,
    "tau": 10,
    "batch_size": 512,
    "optim": "adam",
    "init": 0.84,
    "in_seq": 8,
    "in_f": 16,
    "use_mixed": False,
    "minimal": True,
    "no_gate": False,
    "period_len": [8],
}

model_zoo = {"cfc": CFC, "minimal": CFC_MINIMAL, "no_gate": CFC_NOGATE, "mixed": CFC_MIXED}
model_name = "minimal"


class Config:
    seq_len = 8  # 输入序列长度
    pred_len = 2  # 预测长度
    enc_in = 1  # 输入特征的维度
    aux_features = 15
    period_len = 2  # 周期长度
    embed_dim = 64
    decomposition_kernel_size = 25
    batch_size = 512  # 批次大小
    learning_rate = model_zoo[model_name]["base_lr"]  # 学习率
    num_epochs = 1  # 训练轮数
    validation_split = 0.2  # 验证集比例
    verbose = 1  # 控制输出的详细程度，1: 每个epoch输出

# 假设 trainX 和 trainY 是 numpy 数组，转换为 PyTorch 张量
trainX1_tensor = torch.tensor(trainX1, dtype=torch.float32)
trainX2_tensor = torch.tensor(trainX2, dtype=torch.float32)
trainY_tensor = torch.tensor(trainY, dtype=torch.float32)

# 检查 trainX1 的形状
if trainX1_tensor.shape[1] != Config.seq_len or trainX1_tensor.shape[2] != Config.enc_in:
    raise ValueError(
        f"trainX1 的形状不符合要求，实际形状为 {trainX1_tensor.shape}，应为 (samples, {Config.seq_len}, {Config.enc_in})")

# 检查 trainX2 的形状
if trainX2_tensor.shape[1] != Config.seq_len or trainX2_tensor.shape[2] != Config.aux_features:
    raise ValueError(
        f"trainX2 的形状不符合要求，实际形状为 {trainX2_tensor.shape}，应为 (samples, {Config.seq_len}, {Config.aux_features})")

# 检查目标变量 trainY 的形状
if trainY_tensor.shape[1] != Config.pred_len:
    raise ValueError(f"trainY 的形状不符合要求，实际形状为 {trainY_tensor.shape}，应为 (samples, {Config.pred_len}, 1)")

# 去掉 trainY 的最后一维，调整形状为 (samples, pred_len)
trainY_tensor = trainY_tensor.squeeze(-1)

# 划分训练集和验证集
train_size = int((1 - Config.validation_split) * len(trainX1_tensor))
val_size = len(trainX1_tensor) - train_size

# 分割数据
trainX1_train, valX1 = trainX1_tensor[:train_size], trainX1_tensor[train_size:]
trainX2_train, valX2 = trainX2_tensor[:train_size], trainX2_tensor[train_size:]
trainY_train, valY = trainY_tensor[:train_size], trainY_tensor[train_size:]

# 创建训练集和验证集的 DataLoader
train_dataset = TensorDataset(trainX1_train, trainX2_train, trainY_train)
val_dataset = TensorDataset(valX1, valX2, valY)

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)

save_path = "output"

# 计算参数
def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total_param = sum(p.numel() for p in model.parameters())
        bytes_per_param = 4
        total_bytes = total_param * bytes_per_param
        total_megabytes = total_bytes / (1024 * 1024)
        return total_param, total_megabytes

# 实例化模型并迁移到 GPU
# model = Model(Config).to(device)
model = Cfc(
    in_features=16,
    hidden_size=model_zoo[model_name]["hidden_size"],
    out_feature=1,
    hparams=model_zoo[model_name],
    return_sequences=True,
    use_mixed=model_zoo[model_name]["use_mixed"],
).to(device)

# 计算模型的所有参数量以及占用的内存（以MB为单位）
total_param, total_megabytes = count_parameters(model, only_trainable=False)
print(f"Total parameters: {total_param}")
print(f"Total megabytes: {total_megabytes} MB")

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失

optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
# scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda epoch: model_zoo[model_name]["decay_lr"] ** epoch
)
# 用于保存训练和验证损失
train_losses = []
val_losses = []

# 训练模型
for epoch in range(Config.num_epochs):
    start_time = time.time()  # 记录当前时间

    model.train()
    running_train_loss = 0.0

    for i, (batch_x1, batch_x2, batch_y) in enumerate(train_loader):
        batch_x1, batch_x2, batch_y = batch_x1.to(device), batch_x2.to(device), batch_y.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(batch_x1, batch_x2)
        loss = criterion(outputs, batch_y.unsqueeze(-1))  # 计算损失
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 验证集损失
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch_x1, batch_x2, batch_y in val_loader:
            batch_x1, batch_x2, batch_y = batch_x1.to(device), batch_x2.to(device), batch_y.to(device)
            outputs = model(batch_x1, batch_x2)
            loss = criterion(outputs, batch_y.unsqueeze(-1))
            running_val_loss += loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    scheduler.step()
    # 计算当前 epoch 的运行时间
    epoch_time = time.time() - start_time

    if Config.verbose >= 1:
        print(f"Epoch [{epoch + 1}/{Config.num_epochs}], "
              f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
              f"Epoch Time: {epoch_time:.2f} seconds,"
              f"lr: {optimizer.param_groups[0]['lr']:.6f}")

# 保存模型`
model_path = os.path.join(save_path, 'without_social.pth')
torch.save(model.state_dict(), model_path)

# 绘制训练损失和验证损失
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
