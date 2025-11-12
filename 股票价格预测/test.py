import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from model import LSTM  # 导入LSTM类

# 设置字体为SimHei，用于显示中文
plt.rcParams['font.family'] = 'SimHei'

# 读取CSV文件
file_path = 'TSLA.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path)

# 确保日期列是 datetime 类型
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# 选择多特征：'Close', 'Open', 'High', 'Low', 'Volume'
features = data[['Close', 'Open', 'High', 'Low', 'Volume']].values

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features)

# 准备训练和测试数据
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step)]
        X.append(a)
        y.append(data[i + time_step, 0])  # 预测收盘价
    return np.array(X), np.array(y)

# 创建测试数据集
time_step = 60  # 时间步长
X_test, y_test = create_dataset(test_data, time_step)

# 转换为PyTorch张量
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float().view(-1, 1)

# 加载模型
model = LSTM()
model.load_state_dict(torch.load('lstm_model.pth'))
model.eval()

# 测试模型
with torch.no_grad():
    test_outputs = model(X_test)
    # test_outputs 是预测的收盘价，将其重新归一化为原始价格
    test_outputs = scaler.inverse_transform(np.concatenate((test_outputs.numpy(), np.zeros((test_outputs.shape[0], 4))), axis=1))[:, 0]  # 反归一化收盘价
    y_test_inverse = scaler.inverse_transform(np.concatenate((y_test.numpy(), np.zeros((y_test.shape[0], 4))), axis=1))[:, 0]

# 可视化结果
plt.figure(figsize=(14, 7))
plt.plot(data.index[-len(y_test):], y_test_inverse, label='真实价格', color='blue')
plt.plot(data.index[-len(test_outputs):], test_outputs, label='预测价格', color='red')
plt.title('股票价格预测')
plt.xlabel('日期')
plt.ylabel('价格')
plt.legend()
plt.show()
