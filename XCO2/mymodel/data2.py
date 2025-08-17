import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def reshape_for_model(dataX, seq_len):
    return dataX.reshape(dataX.shape[0], seq_len, dataX.shape[2])

# Create dataset for model training (Time series data formatting)
#def create_dataset(datasetX, datasetY, seq_len, pred_len):
#    dataX, dataY = [], []
#    for i in range(0, len(datasetX) - seq_len - pred_len, pred_len):
#        a = datasetX[i:(i + seq_len), :]
#        dataX.append(a)
#        dataY.append(datasetY[i + seq_len:i + seq_len + pred_len])
#    return np.array(dataX, dtype=np.float32), np.array(dataY, dtype=np.float32)

def create_dataset(datasetX, datasetY, seq_len, pred_len):
    dataX, dataY = [], []
    n_samples = len(datasetX)

    # 确保序列长度和预测长度不会超出数据集的范围
    for i in range(0, n_samples - seq_len - pred_len + 1):
        # 取出 seq_len 长度的历史数据作为输入
        a = datasetX[i:(i + seq_len), :]
        dataX.append(a)

        # 取出 pred_len 长度的未来数据作为目标
        dataY.append(datasetY[i + seq_len:i + seq_len + pred_len])

    return np.array(dataX, dtype=np.float32), np.array(dataY, dtype=np.float32)

def load_and_preprocess_data(train_folder, test_folder, seq_len, pred_len):
    # Load and preprocess training data
    train_filtered_data_list = []
    for file_name in os.listdir(train_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(train_folder, file_name)
            df = pd.read_csv(file_path, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            train_filtered_data_list.append(df)

    train_dataset = pd.concat(train_filtered_data_list, axis=0, ignore_index=True)

    # 提取目标变量和辅助变量
    target_column = 'xco2'
    train_data_Y = train_dataset[target_column].values.reshape(-1, 1)  # 提取目标变量
    train_data_X = train_dataset.values  # 保留所有特征（包括目标变量）

    # 数据归一化
    scaler1 = MinMaxScaler(feature_range=(0, 1))
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    train_X_scaled = scaler1.fit_transform(train_data_X)
    train_Y_scaled = scaler2.fit_transform(train_data_Y)

    # 创建时间窗口数据
    trainX, trainY = create_dataset(train_X_scaled, train_Y_scaled, seq_len, pred_len)
    trainX = reshape_for_model(trainX, seq_len)  # 调整为 (batch_size, seq_len, num_features)

    # 确保正确的目标变量索引
    target_index = list(train_dataset.columns).index(target_column)

    # 分割目标变量和辅助变量
    trainX1 = trainX[:, :, [target_index]]  # 提取目标变量列
    trainX2 = np.delete(trainX, target_index, axis=2)  # 删除目标列，保留辅助变量

    # Load and preprocess testing data
    test_filtered_data_list = []
    for file_name in os.listdir(test_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(test_folder, file_name)
            df = pd.read_csv(file_path, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            test_filtered_data_list.append(df)

    test_dataset = pd.concat(test_filtered_data_list, axis=0, ignore_index=True)

    # 提取目标变量和特征
    test_data_X = test_dataset.values  # 所有特征（包括目标变量）
    test_data_Y = test_dataset['xco2'].values.reshape(-1, 1)  # 目标变量（xco2）

    test_X = scaler1.transform(test_data_X)  # Use the same scaler as for training data
    test_Y = scaler2.transform(test_data_Y)

    testX, testY = create_dataset(test_X, test_Y, seq_len, pred_len)
    testX = reshape_for_model(testX, seq_len)

    # 确保正确分割目标变量和辅助变量
    target_index = list(test_dataset.columns).index('xco2')  # 动态查找目标列索引
    testX1 = testX[:, :, [target_index]]  # 提取目标变量列
    testX2 = np.delete(testX, target_index, axis=2)  # 删除目标列，保留辅助变量

    return trainX1, trainX2, trainY, testX1, testX2, testY, scaler1, scaler2

# Parameters
seq_len = 8
pred_len = 2

train_folder = '../数据/train'  # Replace with the actual path
test_folder = '../数据/test'  # Replace with the actual path

# Load data
trainX1, trainX2, trainY, testX1, testX2, testY, scaler1, scaler2 = load_and_preprocess_data(train_folder, test_folder, seq_len, pred_len)

# Print the shapes of the datasets
print("trainX1 shape:", trainX1.shape)  # 目标变量 xco2 的时间序列
print("trainX2 shape:", trainX2.shape)  # 辅助变量的时间序列
print("trainY shape:", trainY.shape)    # 目标变量
print("testX1 shape:", testX1.shape)    # 测试集中的目标变量时间序列
print("testX2 shape:", testX2.shape)    # Exp测试集中的辅助变量
print("testY shape:", testY.shape)      # 测试集中的目标变量标签

# （样本数, seq， 特征）
