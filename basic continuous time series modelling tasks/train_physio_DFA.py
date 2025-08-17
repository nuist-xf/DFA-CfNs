import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import time
from duv_physionet import get_physio
# from torch_cfc import Cfc
from torch_cfc_DFA import Cfc
import random

# AUROC: 手动计算AUC
def AUROC(preds, labels):
    """
    手动计算 AUROC
    :param preds: 模型的预测值，二分类时应为 [batch_size, 1] 或多分类时为 [batch_size, num_classes]
    :param labels: 真实标签，二分类时应为 [batch_size,] 或多分类时为 [batch_size,]
    :return: 返回 AUROC 分数
    """
    preds = torch.sigmoid(preds).squeeze()  # 对于二分类问题
    labels = labels.float()  # 标签需要转换为 float 类型
    auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().detach().numpy())
    return auc


# Accuracy: 手动计算准确率
def Accuracy(preds, labels):
    """
    手动计算准确率
    :param preds: 模型的预测值，二分类时应为 [batch_size, 1] 或多分类时为 [batch_size, num_classes]
    :param labels: 真实标签，二分类时应为 [batch_size,] 或多分类时为 [batch_size,]
    :return: 返回准确率
    """
    preds = torch.sigmoid(preds).squeeze()  # Sigmoid 输出
    preds = (preds > 0.5).float()  # 将预测值转为 0 或 1
    correct = (preds == labels).float()  # 比较预测值和标签，返回 1 或 0
    accuracy = correct.sum() / len(correct)  # 计算准确率
    return accuracy


class PhysionetLearner(nn.Module):
    def __init__(self, model, hparams):
        super(PhysionetLearner, self).__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(
            weight=torch.Tensor((1.0, hparams["class_weight"]))
        )
        self._hparams = hparams

    def forward(self, x, t, mask):
        return self.model(x, t, mask)


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    running_auroc = 0.0

    for batch in train_loader:
        x, tt, mask, y = batch
        x, tt, mask, y = x.to(device), tt.to(device), mask.to(device), y.to(device)

        optimizer.zero_grad()

        # Prepare batch and forward pass
        y_hat = model(x, tt, mask)
        y_hat = y_hat.view(-1, y_hat.size(-1))
        y = y.view(-1)

        # Compute loss
        loss = model.loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()


        # Calculate accuracy and AUC
        preds = torch.argmax(y_hat.detach(), dim=1)
        accuracy = Accuracy(preds, y)
        softmax = torch.nn.functional.softmax(y_hat, dim=1)[:, 1]
        auc = AUROC(softmax, y)

        running_loss += loss.item()
        running_accuracy += accuracy.item()
        running_auroc += auc

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = running_accuracy / len(train_loader)
    epoch_auroc = running_auroc / len(train_loader)

    return epoch_loss, epoch_accuracy, epoch_auroc


def validate(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    val_auroc = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x, tt, mask, y = batch
            x, tt, mask, y = x.to(device), tt.to(device), mask.to(device), y.to(device)

            # Prepare batch and forward pass
            y_hat = model(x, tt, mask)
            y_hat = y_hat.view(-1, y_hat.size(-1))
            y = y.view(-1)

            # Compute loss
            loss = model.loss_fn(y_hat, y)

            # Calculate accuracy and AUC
            preds = torch.argmax(y_hat, dim=1)
            accuracy = Accuracy(preds, y)
            softmax = torch.nn.functional.softmax(y_hat, dim=1)[:, 1]
            auc = AUROC(softmax, y)
            val_loss += loss.item()
            val_accuracy += accuracy.item()
            val_auroc += auc

    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)
    val_auroc /= len(val_loader)

    return val_loss, val_accuracy, val_auroc


def eval(hparams):
    model = Cfc(
        in_features=41 * 2,
        hidden_size=hparams["hidden_size"],
        out_feature=2,
        hparams=hparams,
    )
    learner = PhysionetLearner(model, hparams)

    class FakeArg:
        batch_size = 32
        classif = True
        n = 8000
        extrap = False
        sample_tp = None
        cut_tp = None

    fake_arg = FakeArg()
    fake_arg.batch_size = hparams["batch_size"]
    device = "cpu"
    data_obj = get_physio(fake_arg, device)
    train_loader = data_obj["train_dataloader"]
    val_loader = data_obj["test_dataloader"]
    # test_loader = data_obj["test_dataloader"]
    test_loader = val_loader
    optimizer = optim.AdamW(
        learner.parameters(),
        lr=hparams["base_lr"],
        weight_decay=hparams["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: hparams["decay_lr"] ** epoch
    )
    best_roc_auc = 0.0
    for epoch in range(hparams["epochs"]):
        train_loss, train_accuracy, train_auroc = train_one_epoch(
            learner, train_loader, optimizer, device
        )

        val_loss, val_accuracy, val_auroc = validate(learner, val_loader, device)
        print(f"Epoch {epoch + 1}/{hparams['epochs']}, Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Train AUROC: {train_auroc:.4f}"
              f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Test AUROC: {val_auroc:.4f}")


        if val_auroc > best_roc_auc:
            best_roc_auc = val_auroc
            torch.save(learner.state_dict(), "best_model.pth")
        scheduler.step()
    print('----------------------------------------------')
    print(f"Best Validation AUROC: {best_roc_auc:.4f}")

    # Test the model
    learner.load_state_dict(torch.load("best_model.pth"))
    test_loss, test_accuracy, test_auroc = validate(learner, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, AUROC: {test_auroc:.4f}")
    return best_roc_auc

# Hyperparameters and model configurations
BEST_DEFAULT = {
    "epochs": 20,
    "class_weight": 11.69,
    "clipnorm": 0,
    "hidden_size": 256,
    "base_lr": 0.002,
    "decay_lr": 0.87,
    "backbone_activation": "silu",
    "backbone_units": 64,
    "backbone_dr": 0.2,
    "backbone_layers": 1,
    "weight_decay": 4e-06,
    "optim": "adamw",
    "init": 0.5,
    "batch_size": 128,
    "period_len": [5, 15, 45],
    "in_seq": 135,
    "in_f": 41,
    "minimal": False,
}

# 0.8397588133811951
BEST_MIXED = {
    "epochs": 65,
    "class_weight": 5.91,
    "clipnorm": 0,
    "hidden_size": 128,
    "base_lr": 0.001,
    "decay_lr": 0.9,
    "backbone_activation": "lecun",
    "backbone_units": 64,
    "backbone_dr": 0.3,
    "backbone_layers": 2,
    "weight_decay": 4e-06,
    "optim": "adamw",
    "init": 0.6,
    "batch_size": 128,
    "use_mixed": True,
    "no_gate": False,
    "minimal": False,
    "use_ltc": False,
}

# 0.8395 $\pm$ 0.0033
BEST_NO_GATE = {
    "epochs": 20,
    "class_weight": 7.73,
    "clipnorm": 0,
    "hidden_size": 64,
    "base_lr": 0.003,
    "decay_lr": 0.73,
    "backbone_activation": "relu",
    "backbone_units": 192,
    "backbone_dr": 0.0,
    "backbone_layers": 2,
    "weight_decay": 5e-05,
    "optim": "adamw",
    "init": 0.55,
    "batch_size": 128,
    "use_mixed": False,
    "no_gate": True,
    "minimal": False,
    "use_ltc": False,
}
# test AUC 0.6431 $\pm$ 0.0180
BEST_MINIMAL = {
    "epochs": 116,
    "class_weight": 18.25,
    "clipnorm": 0,
    "hidden_size": 64,
    "base_lr": 0.003,
    "decay_lr": 0.85,
    "backbone_activation": "tanh",
    "backbone_units": 64,
    "backbone_dr": 0.1,
    "backbone_layers": 3,
    "weight_decay": 5e-05,
    "optim": "adamw",
    "init": 0.53,
    "batch_size": 128,
    "use_mixed": False,
    "no_gate": False,
    "minimal": True,
    "use_ltc": False,
}
# 0.6577
BEST_LTC = {
    "optimizer": "adam",
    "base_lr": 0.05,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "forget_bias": 2.4,
    "epochs": 80,
    "class_weight": 8,
    "clipnorm": 0,
    "hidden_size": 64,
    "backbone_units": 64,
    "backbone_dr": 0.1,
    "backbone_layers": 3,
    "weight_decay": 0,
    "optim": "adamw",
    "init": 0.53,
    "batch_size": 64,
    "use_mixed": False,
    "no_gate": False,
    "minimal": False,
    "use_ltc": True,
}

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # ensure deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)  # set PYTHONHASHSEED environment variable for reproducibility


def score(model_con):
    acc = [eval(model_con) for _ in range(5)]
    print(f"All acc: {np.mean(acc):0.5f} ± {np.std(acc):0.5f}")

if __name__ == "__main__":
    # fix_seed(10)
    fix_seed(8)
    # fix_seed(0)
    # fix_seed(7)


    # default=True,
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_mixed", action="store_true")
    parser.add_argument("--no_gate", action="store_true")
    parser.add_argument("--minimal", default=True, action="store_true")
    parser.add_argument("--use_ltc", action="store_true")
    args = parser.parse_args()

    if args.minimal:
        score(BEST_MINIMAL)
    elif args.no_gate:
        score(BEST_NO_GATE)
    elif args.use_ltc:
        score(BEST_LTC)
    elif args.use_mixed:
        score(BEST_MIXED)
    else:
        score(BEST_DEFAULT)
