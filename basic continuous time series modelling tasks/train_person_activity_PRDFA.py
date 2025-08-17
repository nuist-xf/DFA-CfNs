import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from duv_person_activity import get_person_dataset
from torch_cfc_PRDFA import Cfc
import time
import warnings

warnings.filterwarnings("ignore")


class PersonActivityLearner(nn.Module):
    def __init__(self, model, hparams):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self._hparams = hparams
        self.train_losses = []  # 用于记录每次训练的loss值
        self.val_losses = []  # 用于记录每次验证的loss值
        self.val_accuracies = []  # 用于记录每次验证的accuracy值
        self.test_losses = []  # 用于记录测试集的loss值
        self.test_accuracies = []  # 用于记录测试集的accuracy值

    def _prepare_batch(self, batch):
        _, t, x, mask, y = batch

        t_elapsed = t[:, 1:] - t[:, :-1]
        t_fill = torch.zeros(t.size(0), 1, device=x.device)
        t = torch.cat((t_fill, t_elapsed), dim=1)

        t = t * self._hparams["tau"]
        return x, t, mask, y

    def forward(self, x, t, mask):
        return self.model(x, t, mask=mask)

    def training_step(self, batch):
        x, t, mask, y = self._prepare_batch(batch)

        y_hat = self.forward(x, t, mask=mask)

        enable_signal = torch.sum(y, -1) > 0.0
        y_hat = y_hat[enable_signal]
        y = y[enable_signal]

        y = torch.argmax(y.detach(), dim=-1)
        loss = self.loss_fn(y_hat, y)

        preds = torch.argmax(y_hat.detach(), dim=-1)
        acc = (preds == y).float().mean()

        self.train_losses.append(loss.item())  # 将每次训练的loss值添加到列表中
        return loss, acc

    def validation_step(self, batch):
        x, t, mask, y = self._prepare_batch(batch)

        y_hat = self.forward(x, t, mask=mask)

        enable_signal = torch.sum(y, -1) > 0.0
        y_hat = y_hat[enable_signal]
        y = y[enable_signal]

        y = torch.argmax(y, dim=-1)

        loss = self.loss_fn(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)

        acc = (preds == y).float().mean()

        self.val_losses.append(loss.item())  # 记录验证集的loss
        self.val_accuracies.append(acc.item())  # 记录验证集的accuracy
        return loss, acc

    def test_step(self, batch):
        x, t, mask, y = self._prepare_batch(batch)

        y_hat = self.forward(x, t, mask=mask)

        enable_signal = torch.sum(y, -1) > 0.0
        y_hat = y_hat[enable_signal]
        y = y[enable_signal]

        y = torch.argmax(y, dim=-1)

        loss = self.loss_fn(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)

        acc = (preds == y).float().mean()

        self.test_losses.append(loss.item())  # 记录测试集的loss
        self.test_accuracies.append(acc.item())  # 记录测试集的accuracy
        return loss, acc


def eval(hparams):
    print('eval!')

    # Initialize model
    model = Cfc(
        in_features=12 * 2,
        hidden_size=hparams["hidden_size"],
        out_feature=11,
        hparams=hparams,
        return_sequences=True,
        use_mixed=hparams["use_mixed"],
    )

    learner = PersonActivityLearner(model, hparams)

    # Data loading
    fake_arg = FakeArg()
    fake_arg.batch_size = hparams["batch_size"]
    data_obj = get_person_dataset(fake_arg)

    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    print('数据构建完成，Data Loader 加载完毕！')

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(
        learner.parameters(),
        lr=hparams["base_lr"],
        weight_decay=hparams["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: hparams["decay_lr"] ** epoch
    )

    # Training loop
    for epoch in range(hparams["epochs"]):
        learner.train()
        train_loss = 0.0
        train_acc = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, acc = learner.training_step(batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc.item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # print(f"Epoch {epoch+1}/{hparams['epochs']}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # Validation loop
        learner.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for batch in test_loader:
                loss, acc = learner.validation_step(batch)
                val_loss += loss.item()
                val_acc += acc.item()

        val_loss /= len(test_loader)
        val_acc /= len(test_loader)

        print(f"Epoch {epoch+1}/{hparams['epochs']}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f},"
              f" Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        scheduler.step()

    # Testing loop
    learner.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for batch in test_loader:
            loss, acc = learner.test_step(batch)
            test_loss += loss.item()
            test_acc += acc.item()

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save results to file
    with open('data.txt', 'a') as f:
        for loss in learner.train_losses:
            f.write(str(loss) + '\n')
        f.write(str(test_loss) + '\n')

    return test_acc


class FakeArg:
    batch_size = 32
    classif = True
    extrap = False
    sample_tp = None
    cut_tp = None
    n = 10000


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # ensure deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)  # set PYTHONHASHSEED environment variable for reproducibility


CFC = {
    "epochs": 48,
    "clipnorm": 0,
    "hidden_size": 128,
    "base_lr": 0.004,
    "decay_lr": 0.97,
    "backbone_activation": "silu",
    "backbone_units": 128,
    "backbone_layers": 1,
    "backbone_dr": 0.2,
    "weight_decay": 0.0001,
    "tau": 10,
    "batch_size": 64,
    "optim": "adam",
    "init": 0.84,
    "in_seq": 50,
    "in_f": 12,
    "use_mixed": False,
    "period_len": [5, 10],
}

CFC_MIXED = {
    "epochs": 100,
    "clipnorm": 0,
    "hidden_size": 256,
    "base_lr": 0.004,
    "decay_lr": 0.99,
    "backbone_activation": "gelu",
    "backbone_units": 128,
    "backbone_layers": 2,
    "backbone_dr": 0.5,
    "weight_decay": 4e-05,
    "tau": 10,
    "batch_size": 64,
    "optim": "adamw",
    "init": 1.35,
    "in_seq": 50,
    "in_f": 12,
    "use_mixed": True,
    "no_gate": False,
    "minimal": False,
    "period_len": [5, 10],
}
CFC_NOGATE = {
    "epochs": 100,
    "clipnorm": 0,
    "hidden_size": 128,
    "base_lr": 0.005,
    "decay_lr": 0.97,
    "backbone_activation": "silu",
    "backbone_units": 192,
    "backbone_layers": 2,
    "backbone_dr": 0.2,
    "weight_decay": 0.0002,
    "tau": 0.5,
    "batch_size": 64,
    "optim": "adamw",
    "init": 0.78,
    "in_seq": 50,
    "in_f": 12,
    "use_mixed": False,
    "no_gate": True,
    "minimal": False,
    "period_len": [5, 10],
}
CFC_MINIMAL = {
    "epochs": 150,
    "clipnorm": 0,
    "hidden_size": 128,
    "base_lr": 0.004,
    "decay_lr": 0.97,
    "backbone_activation": "gelu",
    "backbone_units": 256,
    "backbone_layers": 3,
    "backbone_dr": 0.4,
    "weight_decay": 3e-05,
    "tau": 0.1,
    "batch_size": 64,
    "optim": "adamw",
    "init": 0.67,
    "in_seq": 50,
    "in_f": 12,
    "use_mixed": False,
    "no_gate": False,
    "minimal": True,
    "period_len": [5, 10],
}
model_zoo = {"cfc": CFC, "minimal": CFC_MINIMAL, "no_gate": CFC_NOGATE, "mixed": CFC_MIXED}
def score(model_con):
    acc = [eval(model_con) for _ in range(5)]
    print(f"All acc: {np.mean(acc):0.5f} ± {np.std(acc):0.5f}")

if __name__ == "__main__":
    # fix_seed(0)
    # fix_seed(5)
    # fix_seed(1)
    fix_seed(2)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="minimal")
    args = parser.parse_args()

    if args.model not in model_zoo.keys():
        raise ValueError(f"Unknown model '{args.model}', available: {list(model_zoo.keys())}")

    score(model_zoo[args.model])
