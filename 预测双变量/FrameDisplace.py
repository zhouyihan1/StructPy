import sys
import torch
import torch.nn as nn
import joblib
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
import sklearn


matplotlib.rcParams['axes.unicode_minus'] = False

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体

# -------------------- 神经网络定义 --------------------
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

# -------------------- 主窗口 --------------------
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("钢架结构位移预测")
        self.setGeometry(100, 100, 1000, 700)  # 放大窗口
        self.init_ui()
        self.load_model()

        # 默认参数绘制初始图
        default_q, default_l, default_h, default_F = 100, 6, 4, 2000
        default_FB = 0.5 * default_q * default_l + 0.5 * default_F * default_h * default_l
        self.draw_original(default_q, default_l, default_h, default_F, default_FB)
        self.input_F.setText(str(default_F))
        self.input_q.setText(str(default_q))


    # 初始化界面
    def init_ui(self):
        layout = QVBoxLayout()

         # 输入区：只占顶部小部分
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("F(水平力/kN):"))
        self.input_F = QLineEdit()
        input_layout.addWidget(self.input_F)

        input_layout.addWidget(QLabel("q(均布荷载/kN/m):"))
        self.input_q = QLineEdit()
        input_layout.addWidget(self.input_q)

        self.btn_predict = QPushButton("预测A点位移")
        self.btn_predict.clicked.connect(self.predict_displacement)
        input_layout.addWidget(self.btn_predict)

        layout.addLayout(input_layout, stretch=1)  # stretch=1 表示输入区占比例小

        # 显示预测结果
        self.label_result = QLabel("预测位移: ")
        layout.addWidget(self.label_result, stretch=1)

        # 绘图 canvas：占主区域
        self.figure = Figure(figsize=(10, 9))  # 调大图像尺寸
        self.canvas_original = FigureCanvas(self.figure)
        self.canvas_original.axes = self.figure.add_subplot(111)
        layout.addWidget(self.canvas_original, stretch=10)  # stretch=10 表示占大部分空间

        self.setLayout(layout)
    # 加载模型和标准化器
    def load_model(self):
        self.net = Net()
        self.net.load_state_dict(torch.load("model.pth"))
        self.net.eval()
        self.scaler_X = joblib.load("scaler_X.pkl")
        self.scaler_y = joblib.load("scaler_y.pkl")

    # 预测功能
    def predict_displacement(self):
        try:
            F = float(self.input_F.text())
            q = float(self.input_q.text())
        except:
            QMessageBox.warning(self, "输入错误", "请输入有效数字！")
            return

        # 预处理输入
        X_new = np.array([[F, q]])
        X_new_scaled = self.scaler_X.transform(X_new)
        X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

        # 模型预测
        with torch.no_grad():
            pred_scaled = self.net(X_new_tensor).numpy()
            pred = self.scaler_y.inverse_transform(pred_scaled)

        # 理论计算位移（用于参考）
        h, l = 4, 6
        e, i = 2.1e5, 1.4e4
        ei = e * i * 1e-6
        FB = 0.5 * q * l + 0.5 * F * h * l
        displacement = (
            0.5*l*h*2/3*FB*l +
            2/3*l*q*l**2/8*0.5*h +
            0.5*l*h*F*h*2/3 +
            h*0.25*F*h*0.5*0.5*h
        ) / ei

        # 误差计算
        abs_error = abs(pred[0][0] - displacement)
        rel_error = abs_error / abs(displacement) * 100

        self.label_result.setText(
            f"预测位移: {pred[0][0]:.6e} m | 实际位移: {displacement:.6e} m | 相对误差: {rel_error:.2f}%"
        )

        # 绘制原图
        self.draw_original(q, l, h, F, FB)

    # 绘制钢架结构原图
    def draw_original(self, q, l, h, F, FB):
        ax = self.canvas_original.axes
        ax.clear()
        
        ax.grid(True, linestyle='--', alpha=0.5)
        
        A, B, C, D = (0,0), (l,0), (0,-h), (0,-h/2)
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k-', linewidth=2)
        ax.plot([A[0], C[0]], [A[1], C[1]], 'k-', linewidth=2)
        ax.plot(C[0], C[1], 'o', markersize=8, color='black')

        # AB段均布力箭头
        num_arrows = 9
        arrow_spacing = l / num_arrows
        arrow_length = 0.5
        for i in range(num_arrows+1):
            x_pos = A[0] + i*arrow_spacing
            ax.arrow(x_pos, A[1]+0.5, 0, -arrow_length*0.7,
                     head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=1)
        ax.plot([A[0], B[0]], [A[1]+0.5, B[1]+0.5], 'k--', linewidth=1.5)
        mid_x, mid_y = (A[0]+B[0])/2, A[1]+0.5+0.2
        ax.text(mid_x, mid_y, f'q={q}', fontsize=14, ha='center')

        # D点水平力F
        arrow_length = 1.0
        ax.arrow(D[0]-arrow_length, D[1], arrow_length*0.9, 0,
                 head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=1.5)
        ax.text(D[0]-arrow_length-0.3, D[1]+0.2, f'F_B={F}', fontsize=12)

        # 点标注
        ax.text(A[0]-0.3, A[1]-0.3, 'A', fontsize=12, color='#1e40af', fontweight='bold')
        ax.text(B[0]-0.3, B[1], 'B', fontsize=12, color='#1e40af')
        ax.text(C[0]-0.3, C[1], 'C', fontsize=12, color='#1e40af')
        ax.text(D[0]+0.1, D[1], 'D', fontsize=12, color='#1e40af')

        ax.set_xlabel('X 坐标 (m)')
        ax.set_ylabel('Y 坐标 (m)')
        ax.set_title('钢架结构示意图')
        ax.set_xlim(-l*0.2, l*1.2)
        ax.set_ylim(-h*1.2, 1.1)
        ax.set_aspect('equal')

        self.canvas_original.draw()

# -------------------- 运行 --------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
