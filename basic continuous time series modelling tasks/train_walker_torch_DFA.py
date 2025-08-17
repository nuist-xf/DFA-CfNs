import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# from irregular_sampled_datasets import Walker2dImitationData
from torch_cfc_DFA import Cfc
import argparse
import os
import urllib.request
import zipfile
import warnings
warnings.filterwarnings("ignore")

class Walker2dImitationData:
    def __init__(self, seq_len):
        self.seq_len = seq_len
        os.makedirs("data", exist_ok=True)
        data_path = "data/walker/rollout_000.npy"
        zip_path = "walker.zip"
        if not os.path.isfile(data_path):
            url = "https://www.mit.edu/~mlechner/walker.zip"
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, zip_path)
            print("Download completed. Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("data/")
            print("Extraction completed.")
            os.remove(zip_path)  # 删除下载的压缩文件
        all_files = sorted(
            [
                os.path.join("data/walker", d)
                for d in os.listdir("data/walker")
                if d.endswith(".npy")
            ]
        )

        self.rng = np.random.RandomState(891374)
        np.random.RandomState(125487).shuffle(all_files)
        # 15% test set, 10% validation set, the rest is for training
        test_n = int(0.15 * len(all_files))
        valid_n = int((0.15 + 0.1) * len(all_files))
        test_files = all_files[:test_n]
        valid_files = all_files[test_n:valid_n]
        train_files = all_files[valid_n:]

        train_x, train_t, train_y = self._load_files(train_files)
        valid_x, valid_t, valid_y = self._load_files(valid_files)
        test_x, test_t, test_y = self._load_files(test_files)

        train_x, train_t, train_y = self.perturb_sequences(train_x, train_t, train_y)
        valid_x, valid_t, valid_y = self.perturb_sequences(valid_x, valid_t, valid_y)
        test_x, test_t, test_y = self.perturb_sequences(test_x, test_t, test_y)

        self.train_x, self.train_times, self.train_y = self.align_sequences(
            train_x, train_t, train_y
        )
        self.valid_x, self.valid_times, self.valid_y = self.align_sequences(
            valid_x, valid_t, valid_y
        )
        self.test_x, self.test_times, self.test_y = self.align_sequences(
            test_x, test_t, test_y
        )
        self.input_size = self.train_x.shape[-1]

        # print("train_times: ", str(self.train_times.shape))
        # print("train_x: ", str(self.train_x.shape))
        # print("train_y: ", str(self.train_y.shape))

    def align_sequences(self, set_x, set_t, set_y):

        times = []
        x = []
        y = []
        for i in range(len(set_y)):

            seq_x = set_x[i]
            seq_t = set_t[i]
            seq_y = set_y[i]

            for t in range(0, seq_y.shape[0] - self.seq_len, self.seq_len // 4):
                x.append(seq_x[t : t + self.seq_len])
                times.append(seq_t[t : t + self.seq_len])
                y.append(seq_y[t : t + self.seq_len])

        return (
            np.stack(x, axis=0),
            np.expand_dims(np.stack(times, axis=0), axis=-1),
            np.stack(y, axis=0),
        )

    def perturb_sequences(self, set_x, set_t, set_y):

        x = []
        times = []
        y = []
        for i in range(len(set_y)):

            seq_x = set_x[i]
            seq_y = set_y[i]

            new_x, new_times = [], []
            new_y = []

            skip = 0
            for t in range(seq_y.shape[0]):
                skip += 1
                if self.rng.rand() < 0.9:
                    new_x.append(seq_x[t])
                    new_times.append(skip)
                    new_y.append(seq_y[t])
                    skip = 0

            x.append(np.stack(new_x, axis=0))
            times.append(np.stack(new_times, axis=0))
            y.append(np.stack(new_y, axis=0))

        return x, times, y

    def _load_files(self, files):
        all_x = []
        all_t = []
        all_y = []
        for f in files:

            arr = np.load(f)
            x_state = arr[:-1, :].astype(np.float32)
            y = arr[1:, :].astype(np.float32)

            x_times = np.ones(x_state.shape[0])
            all_x.append(x_state)
            all_t.append(x_times)
            all_y.append(y)

            # print("Loaded file '{}' of length {:d}".format(f, x_state.shape[0]))
        return all_x, all_t, all_y


device = "cuda" if torch.cuda.is_available() else "cpu"
# Fix seed for reproducibility
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# Data preparation function
def prepare_data(data, batch_size):
    train_dataset = TensorDataset(torch.tensor(data.train_x, dtype=torch.float32),
                                  torch.tensor(data.train_times, dtype=torch.float32),
                                  torch.tensor(data.train_y, dtype=torch.float32))
    valid_dataset = TensorDataset(torch.tensor(data.valid_x, dtype=torch.float32),
                                  torch.tensor(data.valid_times, dtype=torch.float32),
                                  torch.tensor(data.valid_y, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(data.test_x, dtype=torch.float32),
                                 torch.tensor(data.test_times, dtype=torch.float32),
                                 torch.tensor(data.test_y, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


# Training function
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x, t, y in dataloader:
        x, t, y = x.to(device), t.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x, t)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# Evaluation function
def eval_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, t, y in dataloader:
            x, t, y = x.to(device), t.to(device), y.to(device)
            outputs = model(x, t)
            loss = criterion(outputs, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# Main evaluation function
def eval(config):
    data = Walker2dImitationData(seq_len=64)

    model = Cfc(
        in_features=data.input_size,
        hidden_size=config["size"],
        out_feature=data.input_size,
        hparams=config,
        return_sequences=True,
        use_mixed=config["use_mixed"],
    ).to(device)
    print(1)
    train_loader, valid_loader, test_loader = prepare_data(data, config["batch_size"])
    print(2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["base_lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["decay_lr"])

    best_loss = float('inf')
    best_weights = None

    for epoch in range(config["epochs"]):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        valid_loss = eval_epoch(model, valid_loader, criterion)
        print(f"Epoch {epoch + 1}/{config['epochs']}, Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_weights = model.state_dict()

        scheduler.step()

    model.load_state_dict(best_weights)
    test_loss = eval_epoch(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}")
    return test_loss


# Scoring function
def score(config):
    acc = [eval(config) for _ in range(5)]
    print(f"MSE: {np.mean(acc):0.5f} ± {np.std(acc):0.5f}")

# 0.64038 +- 0.00574
BEST_DEFAULT = {
    "clipnorm": 1,
    "optimizer": "adam",
    "batch_size": 256,
    "size": 64,
    "epochs": 100,
    "base_lr": 0.01,
    "decay_lr": 0.97,
    "backbone_activation": "silu",
    "backbone_dr": 0.1,
    "forget_bias": 1.6,
    "backbone_units": 256,
    "backbone_layers": 2,
    "weight_decay": 1e-06,
    "use_mixed": False,
    "no_gate": False,
    "minimal": False,
    "return_sequences": True,
}

# MSE: 0.61654 +- 0.00634
BEST_MIXED = {
    "clipnorm": 10,
    "optimizer": "adam",
    "batch_size": 128,
    "size": 128,
    "epochs": 100,
    "base_lr": 0.01,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "backbone_dr": 0.25,
    "forget_bias": 2.1,
    "backbone_units": 128,
    "backbone_layers": 2,
    "weight_decay": 6e-06,
    "use_mixed": True,
    "no_gate": False,
    "minimal": False,
    "return_sequences": True,
}

# 0.65040 $\pm$ 0.00814
BEST_NO_GATE = {
    "clipnorm": 1,
    "optimizer": "adam",
    "batch_size": 128,
    "size": 128,
    "epochs": 100,
    "base_lr": 0.01,
    "decay_lr": 0.97,
    "backbone_activation": "lecun",
    "backbone_dr": 0.1,
    "forget_bias": 2.8,
    "backbone_units": 128,
    "backbone_layers": 2,
    "weight_decay": 3e-05,
    "use_mixed": False,
    "no_gate": True,
    "minimal": False,
    "return_sequences": True,

}
# 0.94844 $\pm$ 0.00988
BEST_MINIMAL = {
    "clipnorm": 10,
    "optimizer": "adam",
    "batch_size": 128,
    "size": 128,
    "epochs": 100,
    "base_lr": 0.01,
    "decay_lr": 0.97,
    "backbone_activation": "silu",
    "backbone_dr": 0.1,
    "forget_bias": 5.0,
    "backbone_units": 128,
    "backbone_layers": 2,
    "weight_decay": 1e-06,
    "use_mixed": False,
    "no_gate": False,
    "minimal": True,
    "return_sequences": True,

}

# 0.66225 $\pm$ 0.01330
BEST_LTC = {
    "clipnorm": 10,
    "optimizer": "adam",
    "batch_size": 128,
    "size": 128,
    "epochs": 50,
    "base_lr": 0.05,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "backbone_dr": 0.0,
    "forget_bias": 2.4,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 1e-05,
    "use_mixed": False,
    "no_gate": False,
    "minimal": False,
    "use_ltc": True,
}
# Main script
if __name__ == "__main__":
    fix_seed(0)
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
