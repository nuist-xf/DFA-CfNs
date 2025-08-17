import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import random
import os
# from trend import Model
from torch_cfc_DFA import Cfc

from data2 import testX1, testX2, testY, scaler1, scaler2
# from spatial_data import testX1, testX2, testY, scaler1, scaler2
# from validate_data import testX1, testX2, testY, scaler1, scaler2

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

# 使用cuda:0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CFC = {
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
    "minimal": False,
    "no_gate": False,
    "period_len": [2, 4],
}
CFC_MIXED = {
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
    "use_mixed": True,
    "minimal": False,
    "no_gate": False,
    "period_len": [2, 4],
}
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
    "period_len": [2, 4],
}
CFC_MINIMAL = {
    "epochs": 100,
    "clipnorm": 0,
    "hidden_size": 64,
    "base_lr": 0.004,
    "decay_lr": 0.97,
    "backbone_activation": "silu",
    "backbone_units": 64,
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
    "period_len": [2, 4],
}

model_zoo = {"cfc": CFC, "minimal": CFC_MINIMAL, "no_gate": CFC_NOGATE, "mixed": CFC_MIXED}
model_name = "no_gate"

class Config:
    seq_len = 8   # 输入序列长度
    pred_len = 2  # 预测长度
    enc_in = 1  # 输入特征的维度
    aux_features = 15
    period_len = 2  # 周期长度
    embed_dim = 64
    decomposition_kernel_size = 25
    batch_size = 512  # 批次大小
    learning_rate = 0.001  # 学习率
    num_epochs = 30  # 训练轮数
    verbose = 1  # 控制输出的详细程度，1: 每个epoch输出

# 加载模型
# model = Model(Config).to(device)  # 将模型迁移到指定的GPU
model = Cfc(
    in_features=16,
    hidden_size=model_zoo[model_name]["hidden_size"],
    out_feature=1,
    hparams=model_zoo[model_name],
    return_sequences=True,
    use_mixed=model_zoo[model_name]["use_mixed"],
).to(device)
model.load_state_dict(torch.load('output/without_social.pth', map_location=device), strict=False)
model.eval()

# 将测试数据转换为张量并构造数据集和加载器
testX1_tensor = torch.tensor(testX1, dtype=torch.float32)
testX2_tensor = torch.tensor(testX2, dtype=torch.float32)
testY_tensor = torch.tensor(testY, dtype=torch.float32)

test_dataset = TensorDataset(testX1_tensor, testX2_tensor, testY_tensor)
test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, drop_last=False)

# 校验输入数据形状
if testX1_tensor.shape[1] != Config.seq_len or testX1_tensor.shape[2] != Config.enc_in:
    raise ValueError(f"testX1 的形状不符合要求，实际形状为 {testX1_tensor.shape}")
if testX2_tensor.shape[1] != Config.seq_len or testX2_tensor.shape[2] != Config.aux_features:
    raise ValueError(f"testX2 的形状不符合要求，实际形状为 {testX2_tensor.shape}")

# 初始化结果容器
all_outputs = []
all_labels = []

# 按批次进行预测
criterion = nn.MSELoss()
test_loss = 0

with torch.no_grad():
    for batch_X1, batch_X2, batch_Y in test_loader:
        # 将数据移动到设备
        batch_X1 = batch_X1.to(device)
        batch_X2 = batch_X2.to(device)
        batch_Y = batch_Y.to(device)

        # 模型预测
        outputs = model(batch_X1, batch_X2)
        loss = criterion(outputs, batch_Y)
        test_loss += loss.item() * batch_X1.size(0)  # 累计每个批次的损失

        all_outputs.append(outputs.cpu().numpy())
        all_labels.append(batch_Y.cpu().numpy())

test_loss /= len(test_dataset)
print("Test Loss:", test_loss)

# 整合结果
all_outputs = np.concatenate(all_outputs, axis=0).squeeze(-1)
all_labels = np.concatenate(all_labels, axis=0).squeeze(-1)
print(all_outputs.shape)
print(all_labels.shape)

# 将预测和标签数据反归一化
all_outputs = scaler2.inverse_transform(all_outputs)
all_labels = scaler2.inverse_transform(all_labels)

# 去除零标签样本计算 MAPE
non_zero_mask = all_labels != 0
label_non_zero = all_labels[non_zero_mask]
predict_non_zero = all_outputs[non_zero_mask]

MAPE = np.mean(np.abs((label_non_zero - predict_non_zero) / label_non_zero))
R2 = r2_score(all_labels, all_outputs)
MAE = mean_absolute_error(all_labels, all_outputs)
RMSE = np.sqrt(mean_squared_error(all_labels, all_outputs))

print('R2:', R2)
print('MAE:', MAE)
print('RMSE:', RMSE)
print('MAPE:', MAPE)

# 保存结果，为每个时间步创建单独的列
results = pd.DataFrame({
   'Real_Value_1': all_labels[:, 0].flatten(),
   #'Real_Value_2': all_labels[:, 1].flatten(),
   'Prediction_1': all_outputs[:, 0].flatten(),
   #'Prediction_2': all_outputs[:, 1].flatten()
})
results.to_csv('Results/without_social.csv', index=False)

# 可视化结果
plt.plot(all_labels[:, 0], label='Real Value 1', linewidth=0.5)
plt.plot(all_outputs[:, 0], label='Prediction 1', linewidth=0.5)
plt.plot(all_labels[:, 0], label='Real Value 2', linewidth=0.5)
plt.plot(all_outputs[:, 0], label='Prediction 2', linewidth=0.5)
plt.legend()
plt.grid(True)
#plt.savefig("results.png", dpi=300)
plt.show()