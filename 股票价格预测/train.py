import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from model import LSTM


# 设置随机种子
torch.manual_seed(42)

# 读取CSV文件
file_path = 'TSLA.csv'  # 替换为你的CSV文件路径
data = pd.read_csv('TSLA.csv')

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

# 创建数据集
time_step = 60  # 时间步长
X_train, y_train = create_dataset(train_data, time_step)

# 转换为PyTorch张量
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float().view(-1, 1)

# 初始化模型、损失函数和优化器
model = LSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'lstm_model.pth')
print("模型已保存为 'lstm_model.pth'")
