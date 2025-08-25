import sys
import torch
import torch.nn as nn
import joblib
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QTabWidget
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
        self.setGeometry(100, 100, 1000, 700)
        self.init_ui()
        self.load_model()

        # 默认参数
        default_q, default_l, default_h, default_F = 100, 6, 4, 2000
        default_FB = 0.5 * default_q * default_l + 0.5 * default_F * default_h * default_l

        # 默认参数绘制初始图
        self.draw_original(default_q, default_l, default_h, default_F, default_FB)
        self.input_F.setText(str(default_F))
        self.input_q.setText(str(default_q))
        
        # 初始时也绘制位移图
        self.draw_displacement_plot()

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
        layout.addLayout(input_layout, stretch=1)
        # 显示预测结果
        self.label_result = QLabel("预测位移: ")
        layout.addWidget(self.label_result, stretch=1)
        
        # 创建选项卡控件
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, stretch=10)

        # 选项卡1：原始结构图
        self.tab_original = QWidget()
        self.figure_original = Figure(figsize=(10, 9))
        self.canvas_original = FigureCanvas(self.figure_original)
        self.canvas_original.axes = self.figure_original.add_subplot(111)
        tab_layout_original = QVBoxLayout()
        tab_layout_original.addWidget(self.canvas_original)
        self.tab_original.setLayout(tab_layout_original)
        self.tabs.addTab(self.tab_original, "结构图")

        # 选项卡2：位移图
        self.tab_displacement = QWidget()
        self.figure_displacement = Figure(figsize=(10, 9))
        self.canvas_displacement = FigureCanvas(self.figure_displacement)
        self.canvas_displacement.axes = self.figure_displacement.add_subplot(111)
        tab_layout_displacement = QVBoxLayout()
        tab_layout_displacement.addWidget(self.canvas_displacement)
        self.tab_displacement.setLayout(tab_layout_displacement)
        self.tabs.addTab(self.tab_displacement, "位移图")

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
        except ValueError:
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
            0.5 * l * h * 2 / 3 * FB * l +
            2 / 3 * l * q * l ** 2 / 8 * 0.5 * h +
            0.5 * l * h * F * h * 2 / 3 +
            h * 0.25 * F * h * 0.5 * 0.5 * h
        ) / ei

        # 误差计算
        abs_error = abs(pred[0][0] - displacement)
        rel_error = abs_error / abs(displacement) * 100
        self.label_result.setText(
            f"预测位移: {pred[0][0]:.6e} m | 实际位移: {displacement:.6e} m | 相对误差: {rel_error:.2f}%"
        )
        
        # 绘制原图和位移图
        self.draw_original(q, l, h, F, FB)
        self.draw_displacement_plot()  # 不再传递参数
        # 切换到位移图选项卡
        self.tabs.setCurrentIndex(1)

    # 绘制钢架结构原图
    def draw_original(self, q, l, h, F, FB):
        ax = self.canvas_original.axes
        ax.clear()
        ax.grid(True, linestyle='--', alpha=0.5)
        A, B, C, D = (0, 0), (l, 0), (0, -h), (0, -h / 2)
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k-', linewidth=2)
        ax.plot([A[0], C[0]], [A[1], C[1]], 'k-', linewidth=2)
        ax.plot(C[0], C[1], 'o', markersize=8, color='black')
        # AB段均布力箭头
        num_arrows = 9
        arrow_spacing = l / num_arrows
        arrow_length = 0.5
        for i in range(num_arrows + 1):
            x_pos = A[0] + i * arrow_spacing
            ax.arrow(x_pos, A[1] + 0.5, 0, -arrow_length * 0.7,
                     head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=1)
        ax.plot([A[0], B[0]], [A[1] + 0.5, B[1] + 0.5], 'k--', linewidth=1.5)
        self.draw_hanging_rod(ax, B)
        mid_x, mid_y = (A[0] + B[0]) / 2, A[1] + 0.5 + 0.2
        ax.text(mid_x, mid_y, f'q={q}', fontsize=14, ha='center')
        # D点水平力F
        arrow_length = 1.0
        ax.arrow(D[0] - arrow_length, D[1], arrow_length * 0.9, 0,
                 head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=1.5)
        ax.text(D[0] - arrow_length - 0.3, D[1] + 0.2, f'F_B={F}', fontsize=12)
        # 点标注
        ax.text(A[0] - 0.3, A[1] - 0.3, 'A', fontsize=12, color='#1e40af', fontweight='bold')
        ax.text(B[0] - 0.3, B[1], 'B', fontsize=12, color='#1e40af')
        ax.text(C[0] - 0.3, C[1], 'C', fontsize=12, color='#1e40af')
        ax.text(D[0] + 0.1, D[1], 'D', fontsize=12, color='#1e40af')
        ax.set_xlabel('X 坐标 (m)')
        ax.set_ylabel('Y 坐标 (m)')
        ax.set_title('钢架结构示意图')
        ax.set_xlim(-l * 0.2, l * 1.2)
        ax.set_ylim(-h * 1.2, 1.1)
        ax.set_aspect('equal')
        self.canvas_original.draw()

    # 绘制悬挂小杆
    def draw_hanging_rod(self, ax, point_coord, rod_length=0.25, marker_radius_offset=0.02):
        px, py = point_coord
        ax.plot([px, px], [py - marker_radius_offset, py - rod_length], 'k-', linewidth=1.8)
        ax.plot(px, py - marker_radius_offset, 'ko', markersize=6, markerfacecolor='w')
        ax.plot(px, py - rod_length, 'ko', markersize=6, markerfacecolor='w')

    # 绘制位移图 (固定不变)
    def draw_displacement_plot(self):
        ax = self.canvas_displacement.axes
        ax.clear()
        
        # 固定参数，不随输入变化
        l, h = 6, 4

        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)

        # 定义原始点坐标
        A_orig = (0, 0)
        B_orig = (l, 0)
        C_orig = (0, -h)

        # 定义固定不变的变形后的点坐标
        A_deformed = (l/5, 0)
        B_deformed = (l, 0)
        C_deformed = (0, -h)

        # 绘制原始结构（虚线）
        ax.plot([A_orig[0], B_orig[0]], [A_orig[1], B_orig[1]], 'k--', linewidth=1.5, alpha=0.5, label='原始结构')
        ax.plot([A_orig[0], C_orig[0]], [A_orig[1], C_orig[1]], 'k--', linewidth=1.5, alpha=0.5)

        # 根据A点位移拟合变形后的结构
        # AB段拟合二次函数
        x_ab = np.linspace(l/5, l, 100)
        midpoint_x = (l/5 + l) / 2
        midpoint_y = -l/5 * 0.3
        coef_ab = np.polyfit([l/5, midpoint_x, l], [0, midpoint_y, 0], 2)
        y_ab = np.polyval(coef_ab, x_ab)

        # AC段拟合二次函数
        x_ac = np.linspace(0, l/5, 100)
        midpoint_y = -h / 2
        midpoint_x = l/5 * 0.5
        coef_ac = np.polyfit([0, midpoint_x, l/5], [-h, midpoint_y, 0], 2)
        y_ac = np.polyval(coef_ac, x_ac)

        # 绘制变形后的结构
        ax.plot(x_ab, y_ab, 'r-', linewidth=2, label='变形结构AB')
        ax.plot(x_ac, y_ac, 'r-', linewidth=2, label='变形结构AC')

        # 标记点
        ax.plot(C_orig[0], C_orig[1], 'o', markersize=8, markerfacecolor='#000000', markeredgecolor='#000000')
        self.draw_hanging_rod(ax, B_orig)
        ax.text(A_deformed[0] - 0.3, A_deformed[1] - 0.3, 'A', fontsize=12, color='#1e40af', fontweight='bold')
        ax.text(B_deformed[0] - 0.3, B_deformed[1], 'B', fontsize=12, color='#1e40af')
        ax.text(C_deformed[0] - 0.3, C_deformed[1], 'C', fontsize=12, color='#1e40af')

        # 标注位移值
        ax.arrow(0, 0, l/5 * 0.9, 0, head_width=0.1, head_length=l/5 * 0.1,
                 fc='blue', ec='blue', linewidth=1.5)

        # 设置图形属性
        ax.set_xlabel('X 坐标 (m)', fontsize=10)
        ax.set_ylabel('Y 坐标 (m)', fontsize=10)
        ax.set_title('位移图', fontsize=14, fontweight='bold')
        ax.legend(loc='best')

        # 设置坐标轴的范围，确保图形比例合适
        x_margin = l * 0.2
        y_margin = h * 0.2
        ax.set_xlim(-x_margin, l + x_margin)
        ax.set_ylim(-h - y_margin, 0.5)
        self.canvas_displacement.draw()

# -------------------- 运行 --------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())