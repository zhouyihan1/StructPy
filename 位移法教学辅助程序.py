import sys
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

class MplCanvas(FigureCanvas):
    """Matplotlib画布类"""
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
        self.setWindowTitle("位移法教学辅助程序 ")
        self.setGeometry(50, 50, 1600, 1000)
        
        # 设置应用样式
        self.setup_style()
        
        # 初始化参数
        self.a = 1
        self.F = 14
        self.l = 4
        self.q = 2
        self.EI = 2e5
        
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
        
        # a参数
        a_label = QLabel("a 值:")
        a_label.setStyleSheet(label_style)
        self.a_edit = QLineEdit(str(self.a))
        self.a_edit.setStyleSheet(input_style)
        param_layout.addWidget(a_label, 0, 0)
        param_layout.addWidget(self.a_edit, 0, 1)
        
        # F参数
        f_label = QLabel("F 值:")
        f_label.setStyleSheet(label_style)
        self.F_edit = QLineEdit(str(self.F))
        self.F_edit.setStyleSheet(input_style)
        param_layout.addWidget(f_label, 1, 0)
        param_layout.addWidget(self.F_edit, 1, 1)
        
        # l参数
        l_label = QLabel("l 值:")
        l_label.setStyleSheet(label_style)
        self.l_edit = QLineEdit(str(self.l))
        self.l_edit.setStyleSheet(input_style)
        param_layout.addWidget(l_label, 2, 0)
        param_layout.addWidget(self.l_edit, 2, 1)
        
        # q参数
        q_label = QLabel("q 值:")
        q_label.setStyleSheet(label_style)
        self.q_edit = QLineEdit(str(self.q))
        self.q_edit.setStyleSheet(input_style)
        param_layout.addWidget(q_label, 3, 0)
        param_layout.addWidget(self.q_edit, 3, 1)
        
        param_group.setLayout(param_layout)
        control_layout.addWidget(param_group)
        
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
        
        # 添加弹性空间
        control_layout.addStretch()
        
        # 说明文本
        info_label = QLabel("""
        <div style='text-align: center; line-height: 1.6;'>
        <h3 style='color: #2c3e50; margin-bottom: 10px;'>使用说明</h3>
        <p style='color: #7f8c8d; margin: 5px 0;'>• 输入结构参数 a, F, l, q</p>
        <p style='color: #7f8c8d; margin: 5px 0;'>• 点击更新按钮应用更改</p>
        <p style='color: #7f8c8d; margin: 5px 0;'>• 查看各个选项卡中的分析结果</p>
        <p style='color: #e74c3c; margin: 10px 0; font-weight: bold;'>
        ⚡ 弯矩图已优化：最大值自动缩放为 l/3
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
        """创建右侧图表区域"""
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
        self.create_mp_tab()
        
        self.create_delta1_tab()
        self.create_delta2_tab()
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
    
    def create_mp_tab(self):
        """创建MP图选项卡"""
        mp_tab = QWidget()
        layout = QVBoxLayout(mp_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 创建画布
        self.mp_canvas = MplCanvas(self, width=14, height=10)
        layout.addWidget(self.mp_canvas)
        
        self.tab_widget.addTab(mp_tab, "MP图")
    

    
    def create_delta1_tab(self):
        """创建Δ1图选项卡"""
        delta1_tab = QWidget()
        layout = QVBoxLayout(delta1_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 创建画布
        self.delta1_canvas = MplCanvas(self, width=14, height=10)
        layout.addWidget(self.delta1_canvas)
        
        self.tab_widget.addTab(delta1_tab, "Δ1图")
    
    def create_delta2_tab(self):
        """创建Δ2图选项卡"""
        delta2_tab = QWidget()
        layout = QVBoxLayout(delta2_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 创建画布
        self.delta2_canvas = MplCanvas(self, width=14, height=10)
        layout.addWidget(self.delta2_canvas)
        
        self.tab_widget.addTab(delta2_tab, "Δ2图")
        
    def create_result_tab(self):
        """创建结果图选项卡"""
        result_tab = QWidget()
        layout = QVBoxLayout(result_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 创建画布
        self.result_canvas = MplCanvas(self, width=14, height=10)
        layout.addWidget(self.result_canvas)
        
        self.tab_widget.addTab(result_tab, "结果图")
    
    def update_parameters(self):
        """更新参数值"""
        try:
            self.a = int(self.a_edit.text())
            self.F = int(self.F_edit.text())
            self.l = int(self.l_edit.text())
            self.q = int(self.q_edit.text())
            
            # 参数验证
            if self.a <= 0 or self.F <= 0 or self.l <= 0 or self.q <= 0:
                self.statusBar().showMessage("❌ 错误：所有参数必须为正数", 3000)
                return
            
            # 更新所有图表
            self.update_all_plots()
            self.statusBar().showMessage(f"✅ 参数已更新: a={self.a}, F={self.F}, l={self.l}, q={self.q}", 3000)
        except ValueError:
            self.statusBar().showMessage("❌ 错误：请输入有效的数值", 3000)
    
    def update_all_plots(self):
        """更新所有图表"""
        self.draw_original_structure()
        self.draw_mp_diagram()
        self.draw_result_diagram()
        self.draw_delta1_diagram()
        self.draw_delta2_diagram()
    
    def get_points(self):
        """获取结构的基本点坐标"""
        return {
            'A': (0, -self.l),
            'B': (self.l, -self.l),
            'C': (-self.a, 0),
            'D': (0, 0),
            'E': (self.l, 0),
        }
    
    def draw_basic_structure(self, ax):
        """绘制基本结构框架"""
        points = self.get_points()
        connections = [
            ('C', 'D'), ('D', 'A'), ('D', 'E'), ('E', 'B'),
        ]
        
        # 绘制连接线
        for start, end in connections:
            ax.plot([points[start][0], points[end][0]],
                     [points[start][1], points[end][1]],
                     'k-', linewidth=3.5, alpha=0.8)
        
        # 绘制节点
        node_style = {'marker': 'o', 'markersize': 12, 'markerfacecolor': '#3498db', 
                      'markeredgecolor': '#2c3e50', 'markeredgewidth': 2}

        # E处铰
        ax.plot(points['E'][0], points['E'][1], 'ko', markersize=12, 
                markerfacecolor='white', markeredgewidth=2, markeredgecolor='#2c3e50')
        
        # 绘制节点标签
        label_style = {'fontsize': 16, 'fontweight': 'bold', 'color': '#2c3e50', 
                       'bbox': dict(facecolor='white', alpha=0.9, edgecolor='#bdc3c7', 
                                   pad=4, boxstyle="round,pad=0.3")}
        
        ax.text(points['A'][0]+0.15, points['A'][1]+0.15, 'A', ha='center', **label_style)
        ax.text(points['B'][0]-0.15, points['B'][1]+0.15, 'B', ha='center', **label_style)
        ax.text(points['C'][0]-0.1, points['C'][1]-0.25, 'C', ha='center', **label_style)
        ax.text(points['D'][0], points['D'][1]+0.25, 'D', ha='center', **label_style)
        ax.text(points['E'][0], points['E'][1]+0.25, 'E', ha='center', **label_style)
        
        # 绘制支撑
        self.draw_support(ax, points['A'][0], points['A'][1])
        self.draw_support(ax, points['B'][0], points['B'][1])
        
        ax.set_aspect('equal', adjustable='box')
        
        # 设置标题
        ax.set_title(f'结构分析图 (a={self.a}, F={self.F}, l={self.l}, q={self.q})', 
                    fontsize=18, fontweight='bold', color='#2c3e50', pad=20)
    
    def draw_support(self, ax, x, y):
        """绘制支撑"""
        ground_width = 0.8
        
        # 绘制支撑底座
        ax.plot([x-ground_width/3, x+ground_width/3], [y, y], 'k-', linewidth=3)
        
        # 绘制地面下方的填充区域
        hatch_width = ground_width/2
        hatch_height = 0.3
        
        # 绘制斜线纹理
        num_lines = 8
        for i in range(num_lines+1):
            x_start = x - hatch_width/2 + i * (hatch_width / num_lines)
            x_end = x_start + (hatch_width / num_lines)
            ax.plot([x_start, x_end], [y - hatch_height, y], 'k-', linewidth=1, alpha=0.7)
    
    def draw_original_structure(self):
        """绘制原始结构图"""
        ax = self.original_canvas.axes
        ax.clear()
        
        self.draw_basic_structure(ax)
        points = self.get_points()
        
        # 绘制C点集中力
        arrow_style = {
            'width': 0.02, 
            'head_width': 0.15, 
            'head_length': 0.25,
            'fc': '#e74c3c', 
            'ec': '#c0392b', 
            'linewidth': 2
        }
        
        ax.arrow(points['C'][0], points['C'][1]+0.5, 0, -0.3, **arrow_style)
        
        # 添加力的标签
        force_label_style = {
            'fontsize': 16, 
            'fontweight': 'bold', 
            'color': '#e74c3c',
            'bbox': dict(facecolor='white', alpha=0.9, edgecolor='#e74c3c', 
                        pad=5, boxstyle="round,pad=0.3"),
            'va': 'bottom'
        }
        
        ax.text(points['C'][0]+0.2, points['C'][1]+0.5, f'F={self.F}', **force_label_style)
        
        # 绘制BE段均布荷载
        num_arrows = 10
        arrow_spacing = self.l / num_arrows
        arrow_length = 0.6
        arrow_start_x = points['E'][0] + arrow_length + 0.3
        
        # 绘制均布荷载的箭头
        q_arrow_style = {
            'width': 0.02, 
            'head_width': 0.12, 
            'head_length': 0.18,
            'fc': '#3498db', 
            'ec': '#2980b9', 
            'linewidth': 2
        }
        
        # 绘制箭头
        for i in range(num_arrows + 1):
            y_pos = points['E'][1] - i * arrow_spacing
            ax.arrow(arrow_start_x, y_pos, -arrow_length, 0, **q_arrow_style)
        
        # 添加荷载标签
        q_label_style = {
            'fontsize': 16, 
            'fontweight': 'bold', 
            'color': '#3498db',
            'bbox': dict(facecolor='white', alpha=0.9, edgecolor='#3498db', 
                        pad=5, boxstyle="round,pad=0.3"),
            'va': 'center'
        }
        
        ax.text(arrow_start_x+0.2, (points['E'][1]+points['B'][1])/2, f'q={self.q}', **q_label_style)
        
        # 绘制箭头上方的虚线
        x_coords = [points['E'][0]+arrow_length+0.3, points['B'][0]+arrow_length+0.3]
        y_coords = [points['E'][1], points['B'][1]]
        ax.plot(x_coords, y_coords, '--', color='#3498db', linewidth=2.5, alpha=0.8)
        
        # 添加尺寸标注
        self.add_dimension_lines(ax, points)
        
        # 设置坐标轴范围
        self.set_axis_limits(ax)
        
        self.original_canvas.draw()
    
    def add_dimension_lines(self, ax, points):
        """添加尺寸标注线"""
        # 标注水平长度l
        y_dim = points['A'][1] - 0.6
        ax.annotate('', xy=(points['A'][0], y_dim), xytext=(points['B'][0], y_dim),
                   arrowprops=dict(arrowstyle='<->', color='#7f8c8d', lw=2))
        ax.text((points['A'][0] + points['B'][0])/2, y_dim-0.25, f'l = {self.l}',
               ha='center', va='top', fontsize=14, fontweight='bold', color='#7f8c8d')
        
        # 标注垂直长度l
        x_dim = points['A'][0] - 0.6
        ax.annotate('', xy=(x_dim, points['A'][1]), xytext=(x_dim, points['D'][1]),
                   arrowprops=dict(arrowstyle='<->', color='#7f8c8d', lw=2))
        ax.text(x_dim-0.25, (points['A'][1] + points['D'][1])/2, f'l = {self.l}',
               ha='right', va='center', fontsize=14, fontweight='bold', color='#7f8c8d')
        
        # 标注a长度
        x_dim = (points['C'][0] + points['D'][0])/2
        y_dim = points['C'][1] - 0.4
        ax.annotate('', xy=(points['C'][0], y_dim), xytext=(points['D'][0], y_dim),
                   arrowprops=dict(arrowstyle='<->', color='#7f8c8d', lw=2))
        ax.text(x_dim, y_dim-0.25, f'a = {self.a}',
               ha='center', va='top', fontsize=14, fontweight='bold', color='#7f8c8d')
    
    def draw_mp_diagram(self):
        """绘制MP图"""
        ax = self.mp_canvas.axes
        ax.clear()
        
        self.draw_basic_structure(ax)
        points = self.get_points()
        
        # 抽取点
        point_D = (0, self.a/2)
        
        # 定义三个点
        point_E = (self.l, 0)
        point_B = (5 * self.l / 4, -self.l)
        point_mid = (7 * self.l / 8, -self.l / 3)
        
        # 准备 x 和 y 坐标数组
        x_points = np.array([point_E[0], point_mid[0], point_B[0]])
        y_points = np.array([point_E[1], point_mid[1], point_B[1]])
        
        # 使用 numpy.polyfit 拟合二次抛物线
        coeffs = np.polyfit(y_points, x_points, 2)
        a_fit, b_fit, c_fit = coeffs
        
        # 生成用于绘制曲线的 y 坐标
        y_curve = np.linspace(min(y_points), max(y_points), 100)
        # 计算对应的 x 坐标
        x_curve = a_fit * y_curve**2 + b_fit * y_curve + c_fit
        
        # 绘制拟合曲线
        ax.plot(x_curve, y_curve, color='#e74c3c', linewidth=3.5, alpha=0.9, label='MP曲线')
        
        # 绘制其他线段
        mp_line_style = {'color': '#e74c3c', 'linewidth': 3.5, 'alpha': 0.9}
        
        ax.plot([points['C'][0], point_D[0]], [points['C'][1], point_D[1]], **mp_line_style)
        ax.plot([points['D'][0], point_D[0]], [points['D'][1], point_D[1]], **mp_line_style)
        ax.plot([points['B'][0], point_B[0]], [points['B'][1], point_B[1]], **mp_line_style)
        


        val_df = self.F * self.a
        val_B = self.q*self.l**2/8

        # 添加MP图的标注
        ax.text(point_D[0]+0.2, point_D[1], f'{val_df:.2f}', color='#e74c3c', fontsize=16, 
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, 
                edgecolor='#e74c3c', pad=4, boxstyle="round,pad=0.3"))
        
        # 在曲线上添加标注点
        ax.text(point_B[0]+0.2, point_B[1], f'{val_B:.2f}', color='#e74c3c', fontsize=14,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='#e74c3c', 
                         pad=3, boxstyle="round,pad=0.3"))
        
        # 设置坐标轴范围
        self.set_axis_limits(ax)
        
        self.mp_canvas.draw()
    
    def draw_result_diagram(self):
        """绘制结果图 - 优化缩放，使弯矩最大值等于l/3"""
        ax = self.result_canvas.axes
        ax.clear()
        
        self.draw_basic_structure(ax)
        points = self.get_points()
        
        # 计算参数
        i = 2 * self.EI / self.l
        F1p = self.F * self.a
        F2p = 3 * self.q * self.l / 8
        delta1 = (4 * F1p + F2p * self.l) / (17 * i)
        delta2 = (3 * F1p + 5 * F2p * self.l) * self.l / (51 * i)
        
        # 计算所有弯矩值
        moment_values = [
            self.F * self.a,
            3 * i * delta1,
            abs(delta1 * 2 * i - 3 * i / self.l * delta2),
            abs(3 * i / self.l * delta2 - delta1 * i),
            6 * i / self.l * delta2 + self.q * self.l**2 / 8
        ]
        
        # 找到最大弯矩值
        max_moment = max(moment_values)
        
        # 优化缩放：使最大弯矩值等于l/3
        target_size = self.l / 3
        zoom_factor = target_size / max_moment if max_moment > 0 else 1.0
        
        # 定义缩放后的点
        point_DF = (0, zoom_factor * self.F * self.a)
        point1 = (0, zoom_factor * (3 * i * delta1))
        point2 = (zoom_factor * (delta1 * 2 * i - 3 * i / self.l * delta2), 0)
        point3 = (zoom_factor * (3 * i / self.l * delta2 - delta1 * i), -self.l)
        point4 = (self.l + zoom_factor * (6 * i / self.l * delta2 + self.q * self.l**2 / 8), -self.l)
        
        # 绘制弯矩线
        moment_line_style = {'color': '#9b59b6', 'linewidth': 3.5, 'alpha': 0.9}
        
        ax.plot([points['C'][0], point_DF[0]], [points['C'][1], point_DF[1]], **moment_line_style)
        ax.plot([points['E'][0], point1[0]], [points['E'][1], point1[1]], **moment_line_style)
        ax.plot([point2[0], point3[0]], [point2[1], point3[1]], **moment_line_style)
        ax.plot([point4[0], points['B'][0]], [point4[1], points['B'][1]], **moment_line_style)
        ax.plot([points['D'][0], point_DF[0]], [points['D'][1], point_DF[1]], **moment_line_style)
        ax.plot([points['D'][0], point1[0]], [points['D'][1], point1[1]], **moment_line_style)
        
        # 获取每个点的值
        val_df = self.F * self.a
        val_1 = 3 * i * delta1
        val_2 = (delta1 * 2 * i - 3 * i / self.l * delta2)
        val_3 = (3 * i / self.l * delta2 - delta1 * i)
        val_4 = (6 * i / self.l * delta2 + self.q * self.l**2 / 8)
        
        # 标注值
        moment_text_style = {
            'color': '#9b59b6', 
            'fontsize': 14, 
            'fontweight': 'bold',
            'bbox': dict(facecolor='white', alpha=0.9, edgecolor='#9b59b6', 
                        pad=4, boxstyle="round,pad=0.3")
        }
        
        ax.text(point_DF[0]-0.3, point_DF[1], f'{val_df:.2f}', ha='right', **moment_text_style)
        ax.text(point1[0]+0.5, point1[1]+0.1, f'{val_1:.2f}', ha='right', **moment_text_style)
        ax.text(point2[0]+0.1, point2[1]+0.1, f'{val_2:.2f}', ha='left', **moment_text_style)
        ax.text(point3[0]-0.3, point3[1]+0.1, f'{val_3:.2f}', ha='right', **moment_text_style)
        ax.text(point4[0]-0.3, point4[1]+0.1, f'{val_4:.2f}', ha='right', **moment_text_style)
        
        # 添加标注点
        moment_point_style = {'marker': 'o', 'markersize': 8, 'markerfacecolor': '#9b59b6', 
                             'markeredgecolor': '#2c3e50', 'markeredgewidth': 2}
        
        ax.plot(point_DF[0], point_DF[1], **moment_point_style)
        ax.plot(point1[0], point1[1], **moment_point_style)
        ax.plot(point2[0], point2[1], **moment_point_style)
        ax.plot(point3[0], point3[1], **moment_point_style)
        ax.plot(point4[0], point4[1], **moment_point_style)
        
        # 绘制抛物线
        point_E = (self.l, 0)
        y_curve = np.linspace(min(point_E[1], point4[1]), max(point_E[1], point4[1]), 100)
        
        x_points = np.array([point_E[0], (7 * self.l / 8, -self.l / 3)[0], point4[0]])
        y_points = np.array([point_E[1], (7 * self.l / 8, -self.l / 3)[1], point4[1]])
        
        # 使用 polyfit 进行二次拟合
        coeffs = np.polyfit(y_points, x_points, 2)
        a_fit, b_fit, c_fit = coeffs
        x_curve = a_fit * y_curve**2 + b_fit * y_curve + c_fit
        
        ax.plot(x_curve, y_curve, color='#9b59b6', linewidth=3.5, alpha=0.9)
        

        
        # 设置坐标轴范围
        self.set_axis_limits(ax)
        
        self.result_canvas.draw()
    
    def draw_delta1_diagram(self):
        """绘制Δ1图"""
        ax = self.delta1_canvas.axes
        ax.clear()
        
        self.draw_basic_structure(ax)
        points = self.get_points()
        
        # 定义点
        point1 = (-self.a/2, 0)
        point2 = (self.a/4, -self.l)
        
        # 绘制线段
        delta_line_style = {'color': '#2ecc71', 'linewidth': 3.5, 'alpha': 0.9}
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], **delta_line_style)
        
        # 标签
        delta_text_style = {
            'fontsize': 14, 
            'fontweight': 'bold',
            'color': '#2ecc71',
            'bbox': dict(facecolor='white', alpha=0.9, edgecolor='#2ecc71', 
                        pad=4, boxstyle="round,pad=0.3")
        }
        
        ax.text(point1[0]-0.3, point1[1]-0.2, '2i', **delta_text_style)
        ax.text(point2[0]+0.2, point2[1], 'i', ha='left', **delta_text_style)
        
        # 绘制以D点为圆心的带箭头圆弧
        center = points['D']
        radius = 0.6
        start_angle = 45
        end_angle = 135
        
        start_rad = np.deg2rad(start_angle)
        end_rad = np.deg2rad(end_angle)
        
        start_point = (center[0] + radius * np.cos(start_rad), center[1] + radius * np.sin(start_rad))
        end_point = (center[0] + radius * np.cos(end_rad), center[1] + radius * np.sin(end_rad))
        
        arc = patches.FancyArrowPatch(
            start_point, end_point,
            arrowstyle='<-',
            connectionstyle='arc3, rad=0.5',
            color='#3498db',
            linestyle='--',
            linewidth=3,
            mutation_scale=25
        )
        ax.add_patch(arc)



        ax.text(center[0]+0.2, center[1]+0.2, 'k11=5i', fontsize=14, fontweight='bold',
                color='#3498db', bbox=dict(facecolor='white', alpha=0.9, 
                edgecolor='#3498db', pad=3, boxstyle="round,pad=0.3"))
        
        # 定义点
        point3 = (0, -self.l/5)
        ax.plot([points['E'][0], point3[0]], [points['E'][1], point3[1]], **delta_line_style)
        ax.text(point3[0]+0.3, point3[1]-0.2, '3i', **delta_text_style)
        
        # 绘制k21
        arrow_style = {
            'width': 0.02, 
            'head_width': 0.15, 
            'head_length': 0.18,
            'fc': '#3498db', 
            'ec': '#2980b9', 
            'linewidth': 2
        }
        
        K21 = f"k21=-3i/{self.l}"
        
        ax.arrow(points['E'][0]+0.15, points['E'][1], 0.25, 0, **arrow_style)
        ax.text(points['E'][0]+0.15, points['C'][1]-0.4, K21,
                fontsize=14, fontweight='bold', color='#3498db',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='#3498db', 
                         pad=3, boxstyle="round,pad=0.3"), va='bottom')
        
        # 设置坐标轴范围
        self.set_axis_limits(ax)
        
        self.delta1_canvas.draw()
    
    def draw_delta2_diagram(self):
        """绘制Δ2图"""
        ax = self.delta2_canvas.axes
        ax.clear()
        
        self.draw_basic_structure(ax)
        points = self.get_points()
        
        # 绘制以D点为圆心的带箭头圆弧
        center = points['D']
        radius = 0.6
        start_angle = 45
        end_angle = 135
        
        start_rad = np.deg2rad(start_angle)
        end_rad = np.deg2rad(end_angle)
        
        start_point = (center[0] + radius * np.cos(start_rad), center[1] + radius * np.sin(start_rad))
        end_point = (center[0] + radius * np.cos(end_rad), center[1] + radius * np.sin(end_rad))
        
        arc = patches.FancyArrowPatch(
            start_point, end_point,
            arrowstyle='<-',
            connectionstyle='arc3, rad=0.5',
            color='#3498db',
            linestyle='--',
            linewidth=3,
            mutation_scale=25
        )
        ax.add_patch(arc)
        
        K12 = f"k21=-3i/{self.l}"
        ax.text(center[0]+0.2, center[1]+0.2, K12, fontsize=14, fontweight='bold',
                color='#3498db', bbox=dict(facecolor='white', alpha=0.9, 
                edgecolor='#3498db', pad=3, boxstyle="round,pad=0.3"))
        
        # 绘制k22
        arrow_style = {
            'width': 0.02, 
            'head_width': 0.15, 
            'head_length': 0.18,
            'fc': '#3498db', 
            'ec': '#2980b9', 
            'linewidth': 2
        }
        
        ax.arrow(points['E'][0]+0.15, points['E'][1], 0.25, 0, **arrow_style)

        K22 = f"k22=12i/{self.l**2}"
        ax.text(points['E'][0]+0.15, points['C'][1]-0.4, K22,
                fontsize=14, fontweight='bold', color='#3498db',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='#3498db', 
                         pad=3, boxstyle="round,pad=0.3"), va='bottom')
        
        # 定义点
        point1 = (self.l/5, 0)
        point2 = (-self.l/5, -self.l)
        point3 = (3/5*self.l, -self.l)
        
        # 连接点
        delta2_line_style = {'color': '#f39c12', 'linewidth': 3.5, 'alpha': 0.9}
        
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], **delta2_line_style)
        ax.plot([point3[0], points['E'][0]], [point3[1], points['E'][1]], **delta2_line_style)
        ax.plot([point2[0], points['A'][0]], [point2[1], points['A'][1]], **delta2_line_style)
        ax.plot([point3[0], points['B'][0]], [point3[1], points['B'][1]], **delta2_line_style)
        
        # 标签
        delta2_text_style = {
            'fontsize': 14, 
            'fontweight': 'bold',
            'color': '#f39c12',
            'bbox': dict(facecolor='white', alpha=0.9, edgecolor='#f39c12', 
                        pad=4, boxstyle="round,pad=0.3")
        }
        a=f"3i/{self.l}"
        b=f'6i/{self.l}'
        ax.text(point3[0]-0.4, point3[1], b, **delta2_text_style)
        ax.text(point1[0]+0.2, point1[1]-0.3, a, **delta2_text_style)
        ax.text(point2[0]-0.4, point2[1], a, **delta2_text_style)
        
        # 添加关键点标记
        delta2_point_style = {'marker': 'o', 'markersize': 8, 'markerfacecolor': '#f39c12', 
                             'markeredgecolor': '#2c3e50', 'markeredgewidth': 2}
        
        ax.plot(point1[0], point1[1], **delta2_point_style)
        ax.plot(point2[0], point2[1], **delta2_point_style)
        ax.plot(point3[0], point3[1], **delta2_point_style)
        
        # 设置坐标轴范围
        self.set_axis_limits(ax)
        
        self.delta2_canvas.draw()
    
    def set_axis_limits(self, ax):
        """设置坐标轴范围"""
        # 计算合适的坐标轴范围
        x_min = -self.a - 1.5
        x_max = self.l + 2.5
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用程序图标和名称
    app.setApplicationName("结构分析系统")
    app.setApplicationVersion("2.0")
    
    window = StructuralAnalysisApp()
    window.show()

    sys.exit(app.exec_())

