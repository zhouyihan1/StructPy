import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


# 读取训练和测试数据

train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

# 特征和标签
X_train = train_df[['Load_F', 'q']].values
y_train = train_df[['Displacement']].values

X_test = test_df[['Load_F', 'q']].values
y_test = test_df[['Displacement']].values


#  标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

# 转 PyTorch 张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

#  创建 DataLoader（mini-batch）
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  #每次 epoch 都重新洗牌，使得 mini-batch 的组合是随机的

# 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 32)  # 两个输入特征
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

net = Net()

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

#  训练模型
epochs = 1000
for epoch in range(epochs):
    net.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = net(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

#  测试预测

net.eval()
with torch.no_grad():
    preds_scaled = net(X_test_tensor).numpy()
    preds = scaler_y.inverse_transform(preds_scaled)
    y_test_orig = scaler_y.inverse_transform(y_test_scaled)

# 9输出测试前10个样本预测结果

print("\n测试集前10个样本的实际值和预测值：")
for i in range(10):
    print(f"实际位移: {y_test_orig[i][0]:.6e} m, 预测位移: {preds[i][0]:.6e} m")
