import torch
import torch.nn as nn
import joblib  #Python 序列化库，用于保存和加载模型或对象（如标准化器）
import numpy as np

# 定义网络结构（要和训练时一样）
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 加载模型
net = Net()
net.load_state_dict(torch.load("model.pth"))  #读取保存的模型参数（weights + biases）
net.eval()

# 加载标准化器
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# 输入新的 F 和 q
F = 2000  # 举例
q = 100

# 预处理
X_new = np.array([[F, q]])
X_new_scaled = scaler_X.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

# 预测
with torch.no_grad():
    pred_scaled = net(X_new_tensor).numpy()
    pred = scaler_y.inverse_transform(pred_scaled)

# 参数
h = 4
l = 6
e = 2.1e5
i = 1.4e4
ei = e * i * 1e-6
FB = 0.5 * q * l + 0.5 * F * h * l
displacement = (
            0.5*l*h*2/3*FB*l +
            2/3*l*q*l**2/8*0.5*h +
            0.5*l*h*F*h*2/3 +
            h*0.25*F*h*0.5*0.5*h
        ) / ei

# 计算误差
abs_error = abs(pred[0][0] - displacement)
rel_error = abs_error / abs(displacement) * 100  # 百分比

# 输出
print(f'输入F: {F} , 输入q: {q}')
print(f"预测位移: {pred[0][0]:.6e} m")
print(f"实际位移: {displacement:.6e} m")
print(f"绝对误差: {abs_error:.6e} m")
print(f"相对误差: {rel_error:.2f} %")
