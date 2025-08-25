import sys
import numpy as np
import matplotlib
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, 
                           QFormLayout, QLineEdit, QPushButton, QLabel, QTabWidget, QMessageBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 添加高DPI支持，这有助于在不同屏幕上获得更好的显示效果
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

# 设置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.updateGeometry()
        # 设置更大的边距，减少空白
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)

class AdvancedDisplacementCalculationSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("单位力位移计算教学辅助程序")
        self.resize(800, 650)
        self.centralwidget = QWidget(self)
        self.setCentralWidget(self.centralwidget)

        # 设置字体
        font = QtGui.QFont()
        font.setPointSize(9)
        QApplication.setFont(font)

        # EI默认值
        self.e_val = 2.1e5
        self.i_val = 1.4e4

        # 主布局
        main_layout = QVBoxLayout(self.centralwidget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- 参数输入区域 ---
        input_frame = QtWidgets.QFrame(self.centralwidget)
        input_frame.setObjectName("inputFrame")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(10, 10, 10, 10)
        input_layout.setSpacing(15)

        # 添加标题
        title_label = QLabel("参数输入")
        title_label.setObjectName("titleLabel")
        title_font = QtGui.QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(QtCore.Qt.AlignCenter)

        # 使用QFormLayout
        form_layout = QFormLayout()
        form_layout.setSpacing(8)
        form_layout.setLabelAlignment(QtCore.Qt.AlignRight)
        
        # 创建输入框
        self.lineEdit_h = QLineEdit("4")  # 高度h
        self.lineEdit_l = QLineEdit("6")  # 长度l
        self.lineEdit_q = QLineEdit("2")  # 均布荷载q
        self.lineEdit_f = QLineEdit("1")  # 水平力F
        
        # EI说明标签
        ei_info_label = QLabel("EI = 2.94 × 10⁹ kN·m²")
        ei_info_label.setStyleSheet("color: #0078d7; font-weight: bold;")
        
        # 将输入框添加到布局
        form_layout.addRow(QLabel("<b>h</b> (高度/m):"), self.lineEdit_h)
        form_layout.addRow(QLabel("<b>l</b> (长度/m):"), self.lineEdit_l)
        form_layout.addRow(QLabel("<b>q</b> (均布荷载/kN/m):"), self.lineEdit_q)
        form_layout.addRow(QLabel("<b>F</b> (水平力/kN):"), self.lineEdit_f)
        form_layout.addRow(QLabel("<b>材料刚度:</b>"), ei_info_label)

        # 创建按钮
        self.btn_calculate = QPushButton("计算")
        self.btn_calculate.setObjectName("calculateButton")
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.btn_calculate)
        button_layout.addStretch()

        # 将表单和按钮添加到输入区域的布局中
        input_layout.addWidget(title_label)
        input_layout.addLayout(form_layout)
        input_layout.addLayout(button_layout)
        input_layout.setStretch(1, 5)

        main_layout.addWidget(input_frame)

        # --- 标签页区域 ---
        main_layout.setStretch(1, 4)  # 让图表区域占更多空间
        self.tab_widget = QTabWidget(self.centralwidget)
        self.tab_widget.setObjectName("tabWidget")
        
        # 创建原图标签页
        self.tab_original = QWidget()
        tab_original_layout = QVBoxLayout(self.tab_original)
        self.canvas_original = MplCanvas(self.tab_original)
        tab_original_layout.addWidget(self.canvas_original)
        self.tab_widget.addTab(self.tab_original, "原图")
        
        # 创建M1图标签页
        self.tab_m1 = QWidget()
        tab_m1_layout = QVBoxLayout(self.tab_m1)
        self.canvas_m1 = MplCanvas(self.tab_m1)
        tab_m1_layout.addWidget(self.canvas_m1)
        self.tab_widget.addTab(self.tab_m1, "M1图")
        
        # 创建Mp图标签页
        self.tab_mp = QWidget()
        tab_mp_layout = QVBoxLayout(self.tab_mp)
        self.canvas_mp = MplCanvas(self.tab_mp)
        tab_mp_layout.addWidget(self.canvas_mp)
        self.tab_widget.addTab(self.tab_mp, "Mp图")
        
        # 创建位移图标签页
        self.tab_displacement = QWidget()
        tab_displacement_layout = QVBoxLayout(self.tab_displacement)
        self.canvas_displacement = MplCanvas(self.tab_displacement)
        tab_displacement_layout.addWidget(self.canvas_displacement)
        self.tab_widget.addTab(self.tab_displacement, "位移图")
        
        # 创建A点位移计算标签页
        self.tab_a_displacement = QWidget()
        tab_a_displacement_layout = QVBoxLayout(self.tab_a_displacement)
        
        # 结果显示区域
        self.result_label = QLabel("A点位移计算结果将显示在这里")
        self.result_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_label.setObjectName("resultLabel")
        self.result_label.setWordWrap(True)
        
        tab_a_displacement_layout.addWidget(self.result_label)
        self.tab_widget.addTab(self.tab_a_displacement, "A点位移计算")
        
        main_layout.addWidget(self.tab_widget)

        # 绑定按钮
        self.btn_calculate.clicked.connect(self.on_calculate)

        # --- 样式表 ---
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            #inputFrame {
                background-color: #ffffff;
                border-radius: 8px;
                border: 1px solid #cccccc;
            }
            #titleLabel {
                color: #333;
                padding-right: 10px;
            }
            QLabel {
                font-size: 9pt;
                color: #333333;
            }
            QLineEdit {
                padding: 5px 6px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                font-size: 9pt;
                min-width: 150px;
            }
            QLineEdit:focus {
                border: 2px solid #0078d7;
                padding: 4px 5px;
            }
            QPushButton#calculateButton {
                background-color: #0078d7;
                color: white;
                border: none;
                padding: 6px 15px;
                border-radius: 4px;
                font-size: 9pt;
                font-weight: bold;
            }
            QPushButton#calculateButton:hover {
                background-color: #005a9e;
            }
            QPushButton#calculateButton:pressed {
                background-color: #00396e;
            }
            #resultLabel {
                font-size: 11pt;
                color: #0078d7;
                padding: 10px;
            }
            QTabBar::tab {
                padding: 6px 12px;
                font-size: 9pt;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 5px;
                background-color: #ffffff;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                border-bottom: 1px solid #ffffff;
            }
            QTabBar::tab:hover:!selected {
                background-color: #f0f0f0;
            }
        """)

        # 初始化时运行一次计算和绘图
        self.on_calculate()

    def on_calculate(self):
        try:
            h_val = float(self.lineEdit_h.text() or "4")
            l_val = float(self.lineEdit_l.text() or "6")
            q_val = float(self.lineEdit_q.text() or "2")
            f_val = float(self.lineEdit_f.text() or "1")
            # 使用类中预设的e_val和i_val值
            e_val = self.e_val
            i_val = self.i_val
            
            # 计算FB
            fb_val = 0.5 * q_val * l_val + 0.5 * f_val * h_val * l_val
            
            # 更新所有图表和计算
            self.draw_original(q_val, l_val, h_val, f_val, fb_val)
            self.draw_m1_plot(l_val, h_val)
            self.draw_mp_plot(l_val, h_val, q_val, f_val, fb_val)
            self.calculate_displacement(l_val, h_val, q_val, f_val, fb_val, e_val, i_val)
            
        except ValueError as e:
            QMessageBox.warning(self, "输入错误", f"请输入有效的数值: {str(e)}")
    
    def draw_original(self, q, l, h, F, FB):
        """绘制钢架结构原图"""
        ax = self.canvas_original.axes
        ax.clear()
        
        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 定义点坐标
        A = (0, 0)
        B = (l, 0)
        C = (0, -h)
        D = (0, -h/2)
        
        # 绘制基础结构线条 AB, AC
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k-', linewidth=2, alpha=0.8)
        ax.plot([A[0], C[0]], [A[1], C[1]], 'k-', linewidth=2, alpha=0.8)
        
        # 标记点
        ax.plot(C[0], C[1], 'o', markersize=8, markerfacecolor='#000000', markeredgecolor='#000000')

        self.draw_hanging_rod(ax, B)
        
        # 添加点的标签
        ax.text(A[0]-0.3, A[1]-0.3, 'A', fontsize=12, color='#1e40af', fontweight='bold')
        ax.text(B[0]-0.3, B[1], 'B', fontsize=12, color='#1e40af')
        ax.text(C[0]-0.3, C[1], 'C', fontsize=12, color='#1e40af')
        ax.text(D[0]+0.1, D[1], 'D', fontsize=12, color='#1e40af')
        
        # 绘制D点处的水平力F
        arrow_length = 1.0
        ax.arrow(D[0]-arrow_length, D[1], arrow_length*0.9, 0, 
                head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=1.5)
        ax.text(D[0]-arrow_length-0.3, D[1]+0.2, f'$F_B={F}$', fontsize=12)
        
        # 绘制AB段上的均布力
        num_arrows = 9
        arrow_spacing = l / num_arrows
        arrow_length = 0.5
        
        # 绘制箭头
        for i in range(0, num_arrows + 1):
            x_pos = A[0] + i * arrow_spacing
            ax.arrow(x_pos, A[1]+0.5, 0, -arrow_length*0.7, 
                    head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=1)
        
        # 绘制箭头上方的虚线
        ax.plot([A[0], B[0]], [A[1]+0.5, B[1]+0.5], 'k--', linewidth=1.5, alpha=0.8)
        
        # 在虚线中央添加标注'q'
        mid_x = (A[0] + B[0]) / 2
        mid_y = A[1] + 0.5 + 0.2
        ax.text(mid_x, mid_y, f'$q={q}$', fontsize=14, ha='center')
        
        # 设置图形属性
        ax.set_xlabel('X 坐标 (m)', fontsize=10)
        ax.set_ylabel('Y 坐标 (m)', fontsize=10)
        ax.set_title('钢架结构示意图', fontsize=14, fontweight='bold')
        
        # 设置坐标轴的范围，确保图形比例合适
        x_margin = l * 0.2
        y_margin = h * 0.2
        ax.set_xlim(-x_margin, l + x_margin)
        ax.set_ylim(-h - y_margin, 0.5 + y_margin)
        
        # 使坐标轴等比例
        ax.set_aspect('equal')
        
        self.canvas_original.draw()

    def draw_m1_plot(self, l, h):
        """绘制M1图"""
        ax = self.canvas_m1.axes
        ax.clear()
        
        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 定义点坐标
        A = (0, 0)
        B = (l, 0)
        C = (0, -h)
        
        # 绘制基础结构线条 AB, AC
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k-', linewidth=2, alpha=0.8)
        ax.plot([A[0], C[0]], [A[1], C[1]], 'k-', linewidth=2, alpha=0.8)
        
        # 标记点
        ax.plot(C[0], C[1], 'o', markersize=8, markerfacecolor='#000000', markeredgecolor='#000000')

        self.draw_hanging_rod(ax, B)
        
        # 添加点的标签
        ax.text(A[0]-0.3, A[1]-0.3, 'A', fontsize=12, color='#1e40af', fontweight='bold')
        ax.text(B[0]-0.3, B[1], 'B', fontsize=12, color='#1e40af')
        ax.text(C[0]-0.3, C[1], 'C', fontsize=12, color='#1e40af')
        
        # 绘制Y1曲线: (0,-h/3) 到 (l,0)
        x1 = np.linspace(0, l, 100)
        y1 = -h/3 * (1 - x1/l)
        ax.plot(x1, y1, 'r-', linewidth=1.2, label='Y1曲线')
        
        # 绘制Y2曲线: (l/3,0) 到 (0,-h)
        x2 = np.linspace(l/3, 0, 100)
        y2 = 0 - (h * (1 - x2/(l/3)))
        ax.plot(x2, y2, 'b-', linewidth=1.2, label='Y2曲线')
        
        # 添加点的值
        ax.text(0, -h/3, h, fontsize=12, color='red')
        ax.text(l/3, 0, h, fontsize=12, color='blue')
        
        # 设置图形属性
        ax.set_xlabel('X 坐标 (m)', fontsize=10)
        ax.set_ylabel('Y 坐标 (m)', fontsize=10)
        ax.set_title('M1图', fontsize=14, fontweight='bold')
        ax.legend()
        
        # 设置坐标轴的范围，确保图形比例合适
        x_margin = l * 0.2
        y_margin = h * 0.2
        ax.set_xlim(-x_margin, l + x_margin)
        ax.set_ylim(-h - y_margin, y_margin)
        
        # 使坐标轴等比例
        ax.set_aspect('equal')
        
        self.canvas_m1.draw()

    def draw_mp_plot(self, l, h, q, F, FB):
        """绘制Mp图"""
        ax = self.canvas_mp.axes
        ax.clear()
        
        # 设置网格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 定义点坐标
        A = (0, 0)
        B = (l, 0)
        C = (0, -h)
        
        # 绘制基础结构线条 AB, AC (黑色粗线)
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k-', linewidth=3)
        ax.plot([A[0], C[0]], [A[1], C[1]], 'k-', linewidth=3)
        
        # 标记点 - 只保留C点并将其改为实心黑色
        ax.plot(C[0], C[1], 'o', markersize=8, markerfacecolor='#000000', markeredgecolor='#000000')
        
        # 添加点的标签
        ax.text(A[0]-0.3, A[1]-0.5, 'A', fontsize=12)
        ax.text(B[0]+0.1, B[1]+0.1, 'B', fontsize=12)
        ax.text(C[0]-0.5, C[1], 'C', fontsize=12)
        
        # 绘制B点的悬挂小杆
        self.draw_hanging_rod(ax, B)
        
        # 简化绘制，按照需求直接使用三点拟合
        
        # AB上的Mp曲线 - 使用(0,-h/4), (0.5*l,h/3), (l,0)三点拟合
        point1_ab = (0, -h/4)
        point2_ab = (0.5*l, -h/3)
        point3_ab = (l, 0)
        
        # 拟合二次函数
        x_points = [point1_ab[0], point2_ab[0], point3_ab[0]]
        y_points = [point1_ab[1], point2_ab[1], point3_ab[1]]
        coef_ab = np.polyfit(x_points, y_points, 2)
        
        # 生成曲线点
        x_ab = np.linspace(0, l, 100)
        y_ab = np.polyval(coef_ab, x_ab)
        
        # 绘制AB段Mp曲线
        ax.plot(x_ab, y_ab, 'r-', linewidth=1.2, label='Mp图线')
        
        # AC上的Mp曲线 - 使用(h/4,0), (h/3,-h/2), (0,-h)三点
        point1_ac = (h/4, 0)
        point2_ac = (h/4, -h/2)
        point3_ac = (0, -h)
        
        # 绘制AC段Mp曲线 - 三点连线
        x_ac = [point1_ac[0], point2_ac[0], point3_ac[0]]
        y_ac = [point1_ac[1], point2_ac[1], point3_ac[1]]
        ax.plot(x_ac, y_ac, 'r-', linewidth=1.2)


        # 添加点的值
        ax.text(point1_ac[0], point1_ac[1]+0.1, F*h, fontsize=12, color='red')
        ax.text(point1_ab[0]-0.5, point1_ab[1], F*h, fontsize=12, color='red')
        ax.text(point2_ab[0], point2_ab[1]-0.3, F*h/2+q*l*l/8, fontsize=12, color='red')

        
        # 设置图形属性
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('Mp', fontsize=10)
        ax.set_title('Mp图', fontsize=14, fontweight='bold')
        
        # 调整坐标轴范围
        ax.set_ylim(-h - 0.2*h, 0.5)
        ax.set_xlim(-0.2*l, l + 0.2*l)
        
        # 添加图例
        ax.legend(loc='lower right')
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        self.canvas_mp.draw()
    
    def draw_hanging_rod(self, ax, point_coord, rod_length=0.25, marker_radius_offset=0.02):
        """绘制悬挂小杆"""
        px, py = point_coord
        # 绘制小杆的竖线
        ax.plot([px, px], [py - marker_radius_offset, py - rod_length], 'k-', linewidth=1.8)
        # 绘制上方的空心圆
        ax.plot(px, py - marker_radius_offset, 'ko', markersize=6, markerfacecolor='w')
        # 绘制下方的空心圆
        ax.plot(px, py - rod_length, 'ko', markersize=6, markerfacecolor='w')
        
    def calculate_displacement(self, l, h, q, F, FB, e, i):
        """计算A点位移"""
        # 将单位转换: E(MPa) * I(mm^4) -> N·mm^2，然后转为kN·m^2
        ei = e * i * 1e-6
        
        # 计算A点横向位移
        displacement = (0.5*l*h*2/3*FB*l + 2/3*l*q*l**2/8*0.5*h + 0.5*l*h*F*h*2/3 + h*0.25*F*h*0.5*0.5*h) / ei
        
        # 显示结果
        self.result_label.setText(f"A点横向位移计算结果:\n\n"
                                f"FB = 0.5*q*l + 0.5*F*h*l = {FB:.2f} kN\n\n"
                                f"位移计算表达式:\n"
                                f"(0.5*l*h*2/3*FB*l + 2/3*l*q*l**2/8*0.5*h + 0.5*l*h*F*h*2/3 + h*0.25*F*h*0.5*0.5*h) / EI\n\n"
                                f"A点横向位移: {displacement:.6f} m")
        
        # 绘制位移图
        self.draw_displacement_plot(l, h, displacement)
    
    def draw_displacement_plot(self, l, h, displacement):
        """绘制位移图"""
        ax = self.canvas_displacement.axes
        ax.clear()
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 定义原始点坐标
        A_orig = (0, 0)
        B_orig = (l, 0)
        C_orig = (0, -h)
        
        # 定义变形后的点坐标
        A_deformed = (displacement, 0)
        B_deformed = (l, 0)
        C_deformed = (0, -h)
        
        # 绘制原始结构（虚线）
        ax.plot([A_orig[0], B_orig[0]], [A_orig[1], B_orig[1]], 'k--', linewidth=1.5, alpha=0.5, label='原始结构')
        ax.plot([A_orig[0], C_orig[0]], [A_orig[1], C_orig[1]], 'k--', linewidth=1.5, alpha=0.5)
        
        # 根据A点位移拟合变形后的结构
        # AB段拟合二次函数
        x_ab = np.linspace(displacement, l, 100)
        # 假设B点固定，A点右移，中点向下凹陷
        midpoint_x = (displacement + l) / 2
        midpoint_y = -displacement * 0.3  # 使中点下凹，形成下突效果
        coef_ab = np.polyfit([displacement, midpoint_x, l], [0, midpoint_y, 0], 2)
        y_ab = np.polyval(coef_ab, x_ab)
        
        # AC段拟合二次函数
        x_ac = np.linspace(0, displacement, 100)
        # 假设C点固定，A点右移，中点向右偏移
        midpoint_y = -h / 2
        midpoint_x = displacement * 0.5  # 假设中点右偏为位移的50%
        coef_ac = np.polyfit([0, midpoint_x, displacement], [-h, midpoint_y, 0], 2)
        y_ac = np.polyval(coef_ac, x_ac)
        
        # 绘制变形后的结构
        ax.plot(x_ab, y_ab, 'r-', linewidth=2, label='变形结构AB')
        ax.plot(x_ac, y_ac, 'r-', linewidth=2, label='变形结构AC')
        
        # 标记点
        ax.plot(C_orig[0], C_orig[1], 'o', markersize=8, markerfacecolor='#000000', markeredgecolor='#000000')

        # 绘制B点的悬挂小杆
        self.draw_hanging_rod(ax, B_orig)
        # 添加点的标签
        ax.text(A_deformed[0]-0.3, A_deformed[1]-0.3, 'A', fontsize=12, color='#1e40af', fontweight='bold')
        ax.text(B_deformed[0]-0.3, B_deformed[1], 'B', fontsize=12, color='#1e40af')
        ax.text(C_deformed[0]-0.3, C_deformed[1], 'C', fontsize=12, color='#1e40af')
        
        # 标注位移值
        ax.text(displacement/2, 0.2, f'A点位移={displacement:.6f}m', color='red', fontsize=10)
        
        # 画一个箭头表示位移方向
        ax.arrow(0, 0, displacement*0.9, 0, head_width=0.1, head_length=displacement*0.1, 
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
        ax.set_ylim(-h - y_margin, 0.5)  # 调整上界限，减少上方空白
        
        self.canvas_displacement.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdvancedDisplacementCalculationSystem()
    window.show()
    sys.exit(app.exec_())