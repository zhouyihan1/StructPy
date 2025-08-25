import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QLineEdit, QPushButton, QTabWidget, QGroupBox, 
                            QFormLayout, QSplitter, QFrame, QGridLayout)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QPalette, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def safe_eval(text):
    """安全解析数字表达式"""
    s = text.strip()
    try:
        return float(s)
    except Exception:
        pass
    s = s.replace('^', '**')
    if not re.fullmatch(r"[0-9eE\+\-\*/\.\(\)\s]+", s):
        raise ValueError(f"无法解析表达式: {text}")
    if re.search(r"[A-DF-Za-df-z_]", s):
        raise ValueError(f"无法解析表达式: {text}")
    try:
        return float(eval(s, {"__builtins__": {}}, {}))
    except Exception:
        raise ValueError(f"无法解析表达式: {text}")

class MplCanvas(FigureCanvas):
    """Matplotlib画布类 """
    def __init__(self, parent=None, width=12, height=9, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08)
        self.axes = self.fig.add_subplot(111)
        
        # 设置图形样式
        self.fig.patch.set_facecolor('#ffffff')
        self.axes.set_facecolor('#fafafa')
        
        super(MplCanvas, self).__init__(self.fig)
        self.setStyleSheet("background-color: white; border: 1px solid #e0e0e0; border-radius: 8px;")

class StructuralAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("结构热效应教学辅助程序")
        self.setGeometry(50, 50, 1600, 1000)
        
        # 设置应用样式 
        self.setup_style()
        
        # 初始化参数
        self.l = 4.0
        self.t = 10.0
        self.a = 12e-6
        self.EI = 1.3e7
        
        # 派生参数
        self.i = self.EI / self.l
        self.h = self.l / 10.0
        
        # 创建中央部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建主布局
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(20)
        
        # 创建左侧控制面板
        self.create_control_panel()
        
        # 创建右侧图表区域
        self.create_chart_area()
        
        # 初始化图表
        self.update_all_plots()
        
        # 创建状态栏
        self.statusBar().showMessage("系统就绪 - 请输入参数并查看分析结果")
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #f8f9fa;
                color: #495057;
                border-top: 1px solid #dee2e6;
                font-size: 12px;
                padding: 5px;
            }
        """)
    
    def setup_style(self):
        """设置应用程序样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QWidget {
                font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
            }
        """)
    
    def create_control_panel(self):
        """创建左侧控制面板"""
        control_widget = QWidget()
        control_widget.setFixedWidth(350)
        control_widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e9ecef;
            }
        """)
        
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(25, 25, 25, 25)
        control_layout.setSpacing(25)
        
        # 标题
        title_label = QLabel("参数设置")
        title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                padding: 15px;
                background-color: #ecf0f1;
                border-radius: 8px;
                margin-bottom: 10px;
            }
        """)
        control_layout.addWidget(title_label)
        
        # 参数输入组
        param_group = QGroupBox("结构参数")
        param_group.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        param_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 10px;
                margin-top: 15px;
                padding-top: 20px;
                background-color: #fdfdfd;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 5px 15px;
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                color: #34495e;
            }
        """)
        
        param_layout = QGridLayout()
        param_layout.setVerticalSpacing(20)
        param_layout.setHorizontalSpacing(15)
        
        # 输入框样式
        input_style = """
            QLineEdit {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                padding: 12px 15px;
                background-color: white;
                font-size: 14px;
                font-weight: bold;
            }
            QLineEdit:focus {
                border: 2px solid #3498db;
                background-color: #f8f9ff;
            }
        """
        
        # 标签样式 
        label_style = """
            QLabel {
                font-weight: bold;
                font-size: 14px;
                color: #2c3e50;
                padding: 5px;
            }
        """
        
        # l参数
        l_label = QLabel("l (长度):")
        l_label.setStyleSheet(label_style)
        self.l_edit = QLineEdit(str(self.l))
        self.l_edit.setStyleSheet(input_style)
        param_layout.addWidget(l_label, 0, 0)
        param_layout.addWidget(self.l_edit, 0, 1)
        
        # t参数
        t_label = QLabel("t (温差):")
        t_label.setStyleSheet(label_style)
        self.t_edit = QLineEdit(str(self.t))
        self.t_edit.setStyleSheet(input_style)
        param_layout.addWidget(t_label, 1, 0)
        param_layout.addWidget(self.t_edit, 1, 1)
        
        # a参数
        a_label = QLabel("a (线膨胀系数):")
        a_label.setStyleSheet(label_style)
        self.a_edit = QLineEdit("12*10^-6")
        self.a_edit.setStyleSheet(input_style)
        param_layout.addWidget(a_label, 2, 0)
        param_layout.addWidget(self.a_edit, 2, 1)
        
        # EI参数
        EI_label = QLabel("EI (弯曲刚度):")
        EI_label.setStyleSheet(label_style)
        self.EI_edit = QLineEdit("1.3*10^7")
        self.EI_edit.setStyleSheet(input_style)
        param_layout.addWidget(EI_label, 3, 0)
        param_layout.addWidget(self.EI_edit, 3, 1)
        
        param_group.setLayout(param_layout)
        control_layout.addWidget(param_group)
        
        # 派生参数显示组
        derived_group = QGroupBox("计算结果")
        derived_group.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        derived_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #27ae60;
                border-radius: 10px;
                margin-top: 15px;
                padding-top: 20px;
                background-color: #f8fff8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 5px 15px;
                background-color: white;
                border: 1px solid #27ae60;
                border-radius: 5px;
                color: #27ae60;
            }
        """)
        
        derived_layout = QVBoxLayout()
        derived_layout.setSpacing(15)
        
        # i值显示
        self.i_label = QLabel()
        self.i_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #27ae60;
                padding: 10px;
                background-color: white;
                border: 1px solid #27ae60;
                border-radius: 8px;
            }
        """)
        derived_layout.addWidget(self.i_label)
        
        # h值显示
        self.h_label = QLabel()
        self.h_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #27ae60;
                padding: 10px;
                background-color: white;
                border: 1px solid #27ae60;
                border-radius: 8px;
            }
        """)
        derived_layout.addWidget(self.h_label)
        
        derived_group.setLayout(derived_layout)
        control_layout.addWidget(derived_group)
        
        # 更新按钮
        update_btn = QPushButton("更新参数并重新计算")
        update_btn.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        update_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px 20px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5dade2, stop:1 #3498db);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2980b9, stop:1 #1c6ea4);
            }
        """)
        update_btn.clicked.connect(self.update_parameters)
        control_layout.addWidget(update_btn)
        
        # 重置按钮
        reset_btn = QPushButton("重置为默认值")
        reset_btn.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        reset_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #95a5a6, stop:1 #7f8c8d);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px 20px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #bdc3c7, stop:1 #95a5a6);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #7f8c8d, stop:1 #6c7b7d);
            }
        """)
        reset_btn.clicked.connect(self.reset_parameters)
        control_layout.addWidget(reset_btn)
        
        # 添加弹性空间
        control_layout.addStretch()
        
        # 说明文本
        info_label = QLabel("""
        <div style='text-align: center; line-height: 1.6;'>
        <h3 style='color: #2c3e50; margin-bottom: 10px;'>使用说明</h3>
        <p style='color: #7f8c8d; margin: 5px 0;'>• 输入结构参数 l, t, a, EI</p>
        <p style='color: #7f8c8d; margin: 5px 0;'>• 支持表达式如 12*10^-6</p>
        <p style='color: #7f8c8d; margin: 5px 0;'>• 点击更新按钮应用更改</p>
        <p style='color: #7f8c8d; margin: 5px 0;'>• 查看各个选项卡中的分析结果</p>
        <p style='color: #e74c3c; margin: 10px 0; font-weight: bold;'>
        ⚡ 基于位移法的热应力分析
        </p>
        </div>
        """)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
            }
        """)
        control_layout.addWidget(info_label)
        
        self.main_layout.addWidget(control_widget)
    
    def create_chart_area(self):
        """创建右侧图表区域 """
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(15)
        
        # 创建选项卡部件
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane { 
                border: 2px solid #bdc3c7;
                background: white;
                border-radius: 10px;
                margin-top: 5px;
            }
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ecf0f1, stop:1 #d5dbdb);
                border: 1px solid #bdc3c7;
                padding: 12px 25px;
                margin-right: 3px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
                font-size: 13px;
                color: #2c3e50;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border-bottom: 2px solid #3498db;
            }
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
            }
        """)
        
        # 创建各个选项卡
        self.create_original_tab()
        self.create_m1_tab()
        self.create_mt0_tab()
        self.create_mt2_tab()
        self.create_result_tab()
        
        chart_layout.addWidget(self.tab_widget)
        self.main_layout.addWidget(chart_widget)
    
    def create_original_tab(self):
        """创建原始结构图选项卡"""
        original_tab = QWidget()
        layout = QVBoxLayout(original_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 创建画布 
        self.original_canvas = MplCanvas(self, width=14, height=10)
        layout.addWidget(self.original_canvas)
        
        self.tab_widget.addTab(original_tab, "原始结构图")
    
    def create_m1_tab(self):
        """创建M1图选项卡"""
        m1_tab = QWidget()
        layout = QVBoxLayout(m1_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        
        self.m1_canvas = MplCanvas(self, width=14, height=10)
        layout.addWidget(self.m1_canvas)
        
        self.tab_widget.addTab(m1_tab, "M1图")
    
    def create_mt0_tab(self):
        """创建Mt0图选项卡"""
        mt0_tab = QWidget()
        layout = QVBoxLayout(mt0_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        
        self.mt0_canvas = MplCanvas(self, width=14, height=10)
        layout.addWidget(self.mt0_canvas)
        
        self.tab_widget.addTab(mt0_tab, "Mt0图")
    
    def create_mt2_tab(self):
        """创建Mt2图选项卡"""
        mt2_tab = QWidget()
        layout = QVBoxLayout(mt2_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        
        self.mt2_canvas = MplCanvas(self, width=14, height=10)
        layout.addWidget(self.mt2_canvas)
        
        self.tab_widget.addTab(mt2_tab, "Mt2图")
    
    def create_result_tab(self):
        """创建结果图选项卡"""
        result_tab = QWidget()
        layout = QVBoxLayout(result_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        
        self.result_canvas = MplCanvas(self, width=14, height=10)
        layout.addWidget(self.result_canvas)
        
        self.tab_widget.addTab(result_tab, "结果图")
    
    def update_derived_labels(self):
        """更新派生参数显示"""
        self.i = self.EI / self.l
        self.h = self.l / 10.0
        self.i_label.setText(f"i = EI / l = {self.i:.3e}")
        self.h_label.setText(f"h = l / 10 = {self.h:.3f}")
    
    def update_parameters(self):
        """更新参数值"""
        try:
            self.l = safe_eval(self.l_edit.text())
            self.t = safe_eval(self.t_edit.text())
            self.a = safe_eval(self.a_edit.text())
            self.EI = safe_eval(self.EI_edit.text())
            
            # 参数验证
            if self.l <= 0 or self.EI <= 0:
                self.statusBar().showMessage("❌ 错误：长度和弯曲刚度必须为正数", 3000)
                return
            
            # 更新派生参数
            self.update_derived_labels()
            
            # 更新所有图表
            self.update_all_plots()
            self.statusBar().showMessage(f"✅ 参数已更新: l={self.l}, t={self.t}, a={self.a:.2e}, EI={self.EI:.2e}", 3000)
        except Exception as e:
            self.statusBar().showMessage(f"❌ 错误：{e}", 3000)
    
    def reset_parameters(self):
        """重置参数"""
        self.l_edit.setText("4")
        self.t_edit.setText("10")
        self.a_edit.setText("12*10^-6")
        self.EI_edit.setText("1.3*10^7")
        self.update_parameters()
        self.statusBar().showMessage("✅ 已重置为默认参数", 3000)
    
    def update_all_plots(self):
        """更新所有图表"""
        self.draw_original_structure()
        self.draw_m1_diagram()
        self.draw_mt0_diagram()
        self.draw_mt2_diagram()
        self.draw_result_diagram()
    
    def set_axis_limits(self, ax):
        """设置坐标轴范围 """
        # 计算合适的坐标轴范围
        x_min = -1.5
        x_max = self.l + 1.5
        y_min = -self.l - 1.5
        y_max = 1.5
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # 设置网格
        ax.grid(True, linestyle='--', alpha=0.3, color='#bdc3c7')
        
        # 设置背景色
        ax.set_facecolor('#fafafa')
        
        # 美化坐标轴
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_color('#bdc3c7')
        
        # 设置坐标轴标签
        ax.set_xlabel('X 坐标', fontsize=12, fontweight='bold', color='#2c3e50')
        ax.set_ylabel('Y 坐标', fontsize=12, fontweight='bold', color='#2c3e50')
        
        # 设置刻度标签样式
        ax.tick_params(axis='both', which='major', labelsize=10, colors='#2c3e50')
    
    # ================= 绘图方法 =================
    
    def draw_original_structure(self):
        """绘制原始结构图"""
        ax = self.original_canvas.axes
        ax.clear()
        l = self.l

        # 定义点坐标
        points = {'B': (0, 0), 'A': (0, -l), 'C': (l, 0)}
        
        # 绘制连接线 
        connections = [('A', 'B'), ('B', 'C')]
        for start, end in connections:
            ax.plot([points[start][0], points[end][0]],
                   [points[start][1], points[end][1]],
                   'k-', linewidth=3.5, alpha=0.8)  

        # 绘制节点标签 
        label_style = {'fontsize': 16, 'fontweight': 'bold', 'color': '#2c3e50', 
                       'bbox': dict(facecolor='white', alpha=0.9, edgecolor='#bdc3c7', 
                                   pad=4, boxstyle="round,pad=0.3")}
        
        ax.text(points['A'][0]-0.2, points['A'][1]+0.1, 'A', ha='center', **label_style)
        ax.text(points['B'][0]-0.2, points['B'][1]+0.1, 'B', ha='center', **label_style)
        ax.text(points['C'][0], points['C'][1]+0.2, 'C', ha='right', **label_style)

        # A点支撑 
        ground_width = 0.8
        ax.plot([points['A'][0]-ground_width/3, points['A'][0]+ground_width/3],
               [points['A'][1], points['A'][1]], 'k-', linewidth=3)
        
        # 绘制地面下方的填充区域
        hatch_width = ground_width/2
        hatch_height = 0.3
        
        # 绘制斜线纹理
        num_lines = 8
        for i in range(num_lines+1):
            x_start = points['A'][0] - hatch_width/2 + i * (hatch_width / num_lines)
            x_end = x_start + (hatch_width / num_lines)
            ax.plot([x_start, x_end], [points['A'][1] - hatch_height, points['A'][1]], 'k-', linewidth=1, alpha=0.7)

        # C点支撑
        ax.plot([points['C'][0], points['C'][0]],
               [points['C'][1]-ground_width/4, points['C'][1]+ground_width/4], 'k-', linewidth=3)
        
        for i in range(num_lines):
            y_start = points['C'][1] + hatch_width/2 - i * (hatch_width / num_lines)
            y_end = y_start - (hatch_width / num_lines)
            x_start = points['C'][0] + hatch_height
            x_end = points['C'][0]
            ax.plot([x_start, x_end], [y_start, y_end], 'k-', linewidth=1, alpha=0.7)

        # 标注信息 
        ax.text(l/2, 0.2, 't℃', fontsize=14, ha='center', fontweight='bold', color='#2c3e50')
        ax.text(-0.2, -l/2, 't℃', fontsize=14, ha='center', fontweight='bold', color='#2c3e50')
        ax.text(l/3, -l/3, '2t℃', fontsize=14, ha='center', fontweight='bold', color='#2c3e50')
        ax.text(2*l/3, -0.2, '2EI', fontsize=14, ha='center', fontweight='bold', color='#2c3e50')
        ax.text(0.2, -2*l/3, 'EI', fontsize=14, ha='center', fontweight='bold', color='#2c3e50')

        # 设置标题 
        ax.set_title(f'原始结构图 (l={self.l}, t={self.t}, a={self.a:.2e}, EI={self.EI:.2e})', 
                    fontsize=18, fontweight='bold', color='#2c3e50', pad=20)
        
        self.set_axis_limits(ax)
        self.original_canvas.draw()

    def draw_m1_diagram(self):
        """绘制M1图"""
        ax = self.m1_canvas.axes
        ax.clear()
        l = self.l

        # 定义点坐标
        points = {'B': (0, 0), 'A': (0, -l), 'C': (l, 0)}
        
        # 绘制连接线 
        connections = [('A', 'B'), ('B', 'C')]
        for start, end in connections:
            ax.plot([points[start][0], points[end][0]],
                   [points[start][1], points[end][1]],
                   'k-', linewidth=3.5, alpha=0.8)  
            
        # 绘制节点标签 
        label_style = {'fontsize': 16, 'fontweight': 'bold', 'color': '#2c3e50', 
                       'bbox': dict(facecolor='white', alpha=0.9, edgecolor='#bdc3c7', 
                                   pad=4, boxstyle="round,pad=0.3")}
        
        ax.text(points['A'][0]-0.2, points['A'][1]+0.1, 'A', ha='center', **label_style)
        ax.text(points['B'][0]-0.2, points['B'][1]+0.1, 'B', ha='center', **label_style)
        ax.text(points['C'][0], points['C'][1]+0.2, 'C', ha='right', **label_style)

        # 简化支撑绘制
        ground_width = 0.8
        ax.plot([points['A'][0]-ground_width/3, points['A'][0]+ground_width/3],
               [points['A'][1], points['A'][1]], 'k-', linewidth=3)
        ax.plot([points['C'][0], points['C'][0]],
               [points['C'][1]-ground_width/4, points['C'][1]+ground_width/4], 'k-', linewidth=3)

        # M1图特有标记点
        point1 = (0, l/4)
        point2 = (l, -l/8)
        point3 = (l/8, 0)
        point4 = (-l/16, -l)

        m1_line_style = {'color': '#e74c3c', 'linewidth': 3.5, 'alpha': 0.9}
        
        connections2 = [(point1, point2), (point3, point4)]
        for start, end in connections2:
            ax.plot([start[0], end[0]], [start[1], end[1]], **m1_line_style)

        ax.plot([points['B'][0], point1[0]], [points['B'][1], point1[1]], **m1_line_style)
        ax.plot([points['B'][0], point3[0]], [points['B'][1], point3[1]], **m1_line_style)
        ax.plot([points['C'][0], point2[0]], [points['C'][1], point2[1]], **m1_line_style)
        ax.plot([points['A'][0], point4[0]], [points['A'][1], point4[1]], **m1_line_style)


        m1_text_style = {'fontsize': 14, 'fontweight': 'bold', 'color': '#e74c3c',
                         'bbox': dict(facecolor='white', alpha=0.9, edgecolor='#e74c3c', 
                                     pad=4, boxstyle="round,pad=0.3")}
        
        ax.text(point1[0]+0.2, point1[1], '8i', ha='center', **m1_text_style)
        ax.text(point2[0], point2[1]-0.2, '4i', ha='center', **m1_text_style)
        ax.text(point3[0]+0.2, point3[1], '4i', ha='center', **m1_text_style)
        ax.text(point4[0]-0.2, point4[1], '2i', ha='center', **m1_text_style)

        # B点圆弧
        center = points['B']
        radius = 0.5
        start_angle = 75
        end_angle = 200
        start_rad = np.deg2rad(start_angle)
        end_rad = np.deg2rad(end_angle)
        start_point = (center[0] + radius * np.cos(start_rad), center[1] + radius * np.sin(start_rad))
        end_point = (center[0] + radius * np.cos(end_rad), center[1] + radius * np.sin(end_rad))
        
        arc = patches.FancyArrowPatch(
            end_point, start_point, arrowstyle='<-',
            connectionstyle='arc3, rad=-0.5',
            color='#3498db', linestyle='--', linewidth=3, mutation_scale=25
        )
        ax.add_patch(arc)

        ax.set_title(f'M1图 (i = {self.i:.2e})', 
                    fontsize=18, fontweight='bold', color='#2c3e50', pad=20)
        
        self.set_axis_limits(ax)
        self.m1_canvas.draw()

    def draw_mt0_diagram(self):
        """绘制Mt0图"""
        ax = self.mt0_canvas.axes
        ax.clear()
        l = self.l

        # 定义点坐标
        points = {'B': (0, 0), 'A': (0, -l), 'C': (l, 0)}
        
        # 绘制连接线 
        connections = [('A', 'B'), ('B', 'C')]
        for start, end in connections:
            ax.plot([points[start][0], points[end][0]],
                   [points[start][1], points[end][1]],
                   'k-', linewidth=3.5, alpha=0.8) 
            
        # 绘制节点标签 
        label_style = {'fontsize': 16, 'fontweight': 'bold', 'color': '#2c3e50', 
                       'bbox': dict(facecolor='white', alpha=0.9, edgecolor='#bdc3c7', 
                                   pad=4, boxstyle="round,pad=0.3")}
        
        ax.text(points['A'][0]-0.2, points['A'][1]+0.1, 'A', ha='center', **label_style)
        ax.text(points['B'][0]-0.2, points['B'][1]+0.1, 'B', ha='center', **label_style)
        ax.text(points['C'][0], points['C'][1]+0.2, 'C', ha='right', **label_style)

        # 支撑结构
        ground_width = 0.8
        ax.plot([points['A'][0]-ground_width/3, points['A'][0]+ground_width/3],
               [points['A'][1], points['A'][1]], 'k-', linewidth=3)
        ax.plot([points['C'][0], points['C'][0]],
               [points['C'][1]-ground_width/4, points['C'][1]+ground_width/4], 'k-', linewidth=3)

        # Mt0图特有标记点
        point1 = (0, l/4)
        point2 = (l, -l/4)
        point3 = (-l/8, 0)
        point4 = (l/8, -l)

        mt0_line_style = {'color': '#9b59b6', 'linewidth': 3.5, 'alpha': 0.9}
        
        connections2 = [(point1, point2), (point3, point4)]
        for start, end in connections2:
            ax.plot([start[0], end[0]], [start[1], end[1]], **mt0_line_style)

        ax.plot([points['B'][0], point1[0]], [points['B'][1], point1[1]], **mt0_line_style)
        ax.plot([points['B'][0], point3[0]], [points['B'][1], point3[1]], **mt0_line_style)
        ax.plot([points['C'][0], point2[0]], [points['C'][1], point2[1]], **mt0_line_style)
        ax.plot([points['A'][0], point4[0]], [points['A'][1], point4[1]], **mt0_line_style)

      
        mt0_text_style = {'fontsize': 14, 'fontweight': 'bold', 'color': '#9b59b6',
                          'bbox': dict(facecolor='white', alpha=0.9, edgecolor='#9b59b6', 
                                      pad=4, boxstyle="round,pad=0.3")}
        
        ax.text(point1[0]+0.2, point1[1], '12iat*1.5', ha='center', **mt0_text_style)
        ax.text(point2[0], point2[1]-0.2, '12iat*1.5', ha='center', **mt0_text_style)
        ax.text(point3[0]-0.2, point3[1], '6iat*1.5', ha='center', **mt0_text_style)
        ax.text(point4[0]+0.2, point4[1], '6iat*1.5', ha='center', **mt0_text_style)

        # B点圆弧 
        center = points['B']
        radius = 0.5
        start_angle = 75
        end_angle = 200
        start_rad = np.deg2rad(start_angle)
        end_rad = np.deg2rad(end_angle)
        start_point = (center[0] + radius * np.cos(start_rad), center[1] + radius * np.sin(start_rad))
        end_point = (center[0] + radius * np.cos(end_rad), center[1] + radius * np.sin(end_rad))
        
        arc = patches.FancyArrowPatch(
            end_point, start_point, arrowstyle='<-',
            connectionstyle='arc3, rad=-0.5',
            color='#3498db', linestyle='--', linewidth=3, mutation_scale=25
        )
        ax.add_patch(arc)

        ax.set_title(f'Mt0图 (i={self.i:.2e}, a={self.a:.2e}, t={self.t})', 
                    fontsize=18, fontweight='bold', color='#2c3e50', pad=20)
        
        self.set_axis_limits(ax)
        self.mt0_canvas.draw()

    def draw_mt2_diagram(self):
        """绘制Mt2图"""
        ax = self.mt2_canvas.axes
        ax.clear()
        l = self.l

        # 定义点坐标
        points = {'B': (0, 0), 'A': (0, -l), 'C': (l, 0)}
        
        # 绘制连接线 
        connections = [('A', 'B'), ('B', 'C')]
        for start, end in connections:
            ax.plot([points[start][0], points[end][0]],
                   [points[start][1], points[end][1]],
                   'k-', linewidth=3.5, alpha=0.8) 

        # 绘制节点标签 
        label_style = {'fontsize': 16, 'fontweight': 'bold', 'color': '#2c3e50', 
                       'bbox': dict(facecolor='white', alpha=0.9, edgecolor='#bdc3c7', 
                                   pad=4, boxstyle="round,pad=0.3")}
        
        ax.text(points['A'][0]-0.2, points['A'][1]+0.1, 'A', ha='center', **label_style)
        ax.text(points['B'][0]-0.2, points['B'][1]+0.1, 'B', ha='center', **label_style)
        ax.text(points['C'][0], points['C'][1]+0.2, 'C', ha='right', **label_style)

        # 支撑结构
        ground_width = 0.8
        ax.plot([points['A'][0]-ground_width/3, points['A'][0]+ground_width/3],
               [points['A'][1], points['A'][1]], 'k-', linewidth=3)
        ax.plot([points['C'][0], points['C'][0]],
               [points['C'][1]-ground_width/4, points['C'][1]+ground_width/4], 'k-', linewidth=3)

        # Mt2图特有标记点
        point1 = (0, l/4)
        point2 = (l, l/4)
        point3 = (-l/8, 0)
        point4 = (-l/8, -l)

      
        mt2_line_style = {'color': '#2ecc71', 'linewidth': 3.5, 'alpha': 0.9}
        
        connections2 = [(point1, point2), (point3, point4)]
        for start, end in connections2:
            ax.plot([start[0], end[0]], [start[1], end[1]], **mt2_line_style)

        ax.plot([points['B'][0], point1[0]], [points['B'][1], point1[1]], **mt2_line_style)
        ax.plot([points['B'][0], point3[0]], [points['B'][1], point3[1]], **mt2_line_style)
        ax.plot([points['C'][0], point2[0]], [points['C'][1], point2[1]], **mt2_line_style)
        ax.plot([points['A'][0], point4[0]], [points['A'][1], point4[1]], **mt2_line_style)

       
        mt2_text_style = {'fontsize': 14, 'fontweight': 'bold', 'color': '#2ecc71',
                          'bbox': dict(facecolor='white', alpha=0.9, edgecolor='#2ecc71', 
                                      pad=4, boxstyle="round,pad=0.3")}
        
        ax.text(point1[0]-0.2, point1[1]+0.1, '2aEIt/1.25h', ha='center', **mt2_text_style)
        ax.text(point2[0]+0.2, point2[1]+0.1, '2aEIt/1.25h', ha='center', **mt2_text_style)
        ax.text(point3[0]-0.2, point3[1], 'aEIt/h', ha='center', **mt2_text_style)
        ax.text(point4[0]-0.2, point4[1], 'aEIt/h', ha='center', **mt2_text_style)

        # B点圆弧 
        center = points['B']
        radius = 0.5
        start_angle = 75
        end_angle = 200
        start_rad = np.deg2rad(start_angle)
        end_rad = np.deg2rad(end_angle)
        start_point = (center[0] + radius * np.cos(start_rad), center[1] + radius * np.sin(start_rad))
        end_point = (center[0] + radius * np.cos(end_rad), center[1] + radius * np.sin(end_rad))
        
        arc = patches.FancyArrowPatch(
            end_point, start_point, arrowstyle='<-',
            connectionstyle='arc3, rad=-0.5',
            color='#3498db', linestyle='--', linewidth=3, mutation_scale=25
        )
        ax.add_patch(arc)

        ax.set_title(f'Mt2图 (h={self.h:.3f}, a={self.a:.2e}, EI={self.EI:.2e}, t={self.t})', 
                    fontsize=18, fontweight='bold', color='#2c3e50', pad=20)
        
        self.set_axis_limits(ax)
        self.mt2_canvas.draw()

    def draw_result_diagram(self):
        """绘制结果图"""
        ax = self.result_canvas.axes
        ax.clear()
        l = self.l

        # 定义点坐标
        points = {'B': (0, 0), 'A': (0, -l), 'C': (l, 0)}
        
        # 绘制连接线 
        connections = [('A', 'B'), ('B', 'C')]
        for start, end in connections:
            ax.plot([points[start][0], points[end][0]],
                   [points[start][1], points[end][1]],
                   'k-', linewidth=3.5, alpha=0.8)  

        # 绘制节点标签
        label_style = {'fontsize': 16, 'fontweight': 'bold', 'color': '#2c3e50', 
                       'bbox': dict(facecolor='white', alpha=0.9, edgecolor='#bdc3c7', 
                                   pad=4, boxstyle="round,pad=0.3")}
        
        ax.text(points['A'][0]-0.2, points['A'][1]+0.1, 'A', ha='center', **label_style)
        ax.text(points['B'][0]-0.2, points['B'][1]+0.1, 'B', ha='center', **label_style)
        ax.text(points['C'][0], points['C'][1]+0.2, 'C', ha='right', **label_style)

        # 支撑结构
        ground_width = 0.8
        ax.plot([points['A'][0]-ground_width/3, points['A'][0]+ground_width/3],
               [points['A'][1], points['A'][1]], 'k-', linewidth=3)
        ax.plot([points['C'][0], points['C'][0]],
               [points['C'][1]-ground_width/4, points['C'][1]+ground_width/4], 'k-', linewidth=3)

        # 结果图特有标记点
        point1 = (0, l/4)
        point2 = (l, -l/12)
        point3 = (-l/4, 0)
        point4 = (l/12, -l)

       
        result_line_style = {'color': '#f39c12', 'linewidth': 3.5, 'alpha': 0.9}
        
        connections2 = [(point1, point2), (point3, point4)]
        for start, end in connections2:
            ax.plot([start[0], end[0]], [start[1], end[1]], **result_line_style)

        ax.plot([points['B'][0], point1[0]], [points['B'][1], point1[1]], **result_line_style)
        ax.plot([points['B'][0], point3[0]], [points['B'][1], point3[1]], **result_line_style)
        ax.plot([points['C'][0], point2[0]], [points['C'][1], point2[1]], **result_line_style)
        ax.plot([points['A'][0], point4[0]], [points['A'][1], point4[1]], **result_line_style)

       
        result_text_style = {'fontsize': 14, 'fontweight': 'bold', 'color': '#f39c12',
                             'bbox': dict(facecolor='white', alpha=0.9, edgecolor='#f39c12', 
                                         pad=4, boxstyle="round,pad=0.3")}
        
        b=format(3*self.EI*self.a*self.t/(20*self.h)*10**-2,'.2f')
        a=format(12*self.EI*self.a*self.t/(5*self.h)*10**-2,'.2f')

        ax.text(point1[0]-0.2, point1[1]+0.1, a, ha='center', **result_text_style)
        ax.text(point2[0]+0.2, point2[1]+0.1, b, ha='center', **result_text_style)
        ax.text(point3[0]-0.2, point3[1], a, ha='center', **result_text_style)
        ax.text(point4[0]-0.2, point4[1], b, ha='center', **result_text_style)

        ax.set_title(f'结果图(12EIat/5h={a},3EIat/20h={b})', 
                    fontsize=18, fontweight='bold', color='#2c3e50', pad=20)
        
        self.set_axis_limits(ax)
        self.result_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用程序图标和名称
    app.setApplicationName("结构热效应教学辅助程序")
    app.setApplicationVersion("2.0")
    
    window = StructuralAnalysisApp()
    window.show()

    sys.exit(app.exec_())
