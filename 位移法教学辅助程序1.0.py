
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QSizePolicy, QWidget, QLabel, QLineEdit, QPushButton,
                            QGroupBox, QComboBox, QFormLayout, QFrame, QTabWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, Arc

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 全局样式
APP_STYLE = """
QMainWindow {
    background-color: #f5f7fa;
}

QWidget {
    font-family: 'Segoe UI', Arial, sans-serif;
}

QGroupBox {
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    margin-top: 1.5ex;
    padding: 10px;
    font-weight: bold;
    color: #374151;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 5px;
}

QPushButton {
    background-color: #4f46e5;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: 500;
    min-height: 30px;
}

QPushButton:hover {
    background-color: #4338ca;
}

QPushButton:pressed {
    background-color: #3730a3;
}

QLineEdit {
    border: 1px solid #d1d5db;
    border-radius: 4px;
    padding: 6px;
    background-color: #ffffff;
}

QLineEdit:focus {
    border: 1px solid #4f46e5;
}

QComboBox {
    border: 1px solid #d1d5db;
    border-radius: 4px;
    padding: 6px;
    background-color: #ffffff;
    min-height: 30px;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid #d1d5db;
}

QTabWidget::pane {
    border: 1px solid #d1d5db;
    border-radius: 6px;
    background-color: #ffffff;
}

QTabBar::tab {
    background-color: #e5e7eb;
    color: #4b5563;
    padding: 8px 16px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
    font-weight: 500;
}

QTabBar::tab:selected {
    background-color: #ffffff;
    color: #4f46e5;
    border-bottom: 2px solid #4f46e5;
}
"""

class MplCanvas(FigureCanvas):
    """Matplotlib画布类"""
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        # 创建一个 Figure 对象
        self.fig = Figure(figsize=(width, height), dpi=dpi) 
        self.axes = self.fig.add_subplot(111)
        
        # 设置Figure的紧凑布局，使图形更好地填充画布
        self.fig.tight_layout(pad=1.0)
        
        super(MplCanvas, self).__init__(self.fig)  
        self.setParent(parent)  
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) 
        self.updateGeometry()


class FrameDiagram:
    """
    一个用于绘制框架结构原图和结果图的类。
    """
    def __init__(self, l=4, q=1, scale_factor=None):
        """
        初始化框架图。
        :param l: 杆件长度
        :param q: 均布荷载
        :param scale_factor: 结果图的缩放因子，如果为None则自动计算
        """
        self.l = l
        self.q = q
        # 如果未提供缩放因子，则自动计算
        if scale_factor is None:
            self.scale_factor = self._calculate_optimal_scale()
        else:
            self.scale_factor = scale_factor
            
        self.points = {'A': (0, 0), 'B': (0, l), 'C': (l, l)}
        self.connections = [('A', 'B'), ('C', 'B')]

    def _calculate_optimal_scale(self):
        """计算最佳缩放因子"""
        # 计算最大弯矩值
        max_moment = self.q * self.l **2 /12
        
        if max_moment <= 0:
            return 0.1
            
        # 使用对数缩放来处理大范围的q和l值
        # 目标是让弯矩图的高度约为杆长的20-30%
        target_ratio = 0.25  # 目标高度与杆长的比例
        
        # 基础缩放因子
        base_scale = target_ratio * self.l / max_moment
        
        # 对于非常大的弯矩值，使用对数调整
        if max_moment > 1000:
            log_adjustment = 1.0 / np.log10(max_moment / 100)
            return base_scale * log_adjustment
        
        return base_scale

    def _draw_hanging_rod(self, ax, side, point_coord):
        """
        绘制左侧或右侧的悬挂杆件。
        :param ax: Matplotlib的Axes对象
        :param side: 'left' 或 'right'
        :param point_coord: 连接点的坐标
        """
        px, py = point_coord
        # 固定悬挂杆件的大小，不随l变化
        rod_length = min(0.1, self.l * 0.02)
        marker_size = min(6, self.l * 0.2)  # 限制最大尺寸
        circle_radius = marker_size * 0.005

        # 绘制竖线和圆
        ax.plot([px, px], [py - circle_radius, py - rod_length], 'k-', linewidth=1.8)
        ax.plot(px, py, 'ko', markersize=marker_size, markerfacecolor='w')
        ax.plot(px, py - rod_length, 'ko', markersize=marker_size, markerfacecolor='w')

        # 根据方向绘制水平部分
        if side == 'right':
            ax.plot(px + rod_length, py, 'ko', markersize=marker_size, markerfacecolor='w')
            ax.plot([px + circle_radius, px + rod_length - circle_radius], [py, py], 'k-', linewidth=1.8)
        elif side == 'left':
            ax.plot(px - rod_length, py, 'ko', markersize=marker_size, markerfacecolor='w')
            ax.plot([px - circle_radius, px - rod_length + circle_radius], [py, py], 'k-', linewidth=1.8)

    def _draw_base_structure(self, ax):
        """绘制基础的杆系结构"""
        ax.set_aspect('equal', adjustable='box')
        for start, end in self.connections:
            ax.plot([self.points[start][0], self.points[end][0]],
                     [self.points[start][1], self.points[end][1]],
                     'k-', linewidth=2)
        self._draw_hanging_rod(ax, 'right', self.points['C'])
        self._draw_hanging_rod(ax, 'left', self.points['A'])

    def _add_point_labels(self, ax):
        """添加A, B, C点标签"""
        for point_name, coords in self.points.items():
            # 固定标签偏移量，不随l变化
            offset_x = -0.15 if point_name == 'B' else 0.05
            offset_y = 0 if point_name == 'B' else 0.05
            ax.text(coords[0] + offset_x, coords[1] + offset_y, point_name, fontsize=12)

    def draw_original_diagram(self, ax):
        """绘制包含荷载的原图"""
        ax.clear()
        ax.set_title("结构示意图", fontsize=14, fontweight='bold')
        self._draw_base_structure(ax)

        # 绘制BC段上的均布力
        q_load_y_start = self.points['B'][1] + min(0.2, self.l * 0.03)
        arrow_length = min(0.2, self.l * 0.025)
        head_length = min(0.05, self.l * 0.0075)
        head_width = min(0.05, self.l * 0.00625)
        num_arrows = min(21, max(11, int(self.l)))  # 根据l调整箭头数量
        arrow_spacing = self.l / (num_arrows - 1)

        ax.plot([self.points['B'][0], self.points['C'][0]], [q_load_y_start, q_load_y_start], 'k-', linewidth=1)
        for i in range(num_arrows):
            x_pos = self.points['B'][0] + i * arrow_spacing
            ax.arrow(x_pos, q_load_y_start, 0, -arrow_length,
                      head_width=head_width, head_length=head_length, fc='black', ec='black', linewidth=0.5,
                      length_includes_head=True)
        self._add_point_labels(ax)

        # 设置坐标轴和背景
        self._set_axes_properties(ax)
        
    def _set_axes_properties(self, ax):
        """设置坐标轴属性"""
        # 设置背景色
        ax.set_facecolor('#f9fafb')
        
        # 根据杆件长度动态调整边距
        constraint_size = min(0.3, self.l * 0.05)
        margin_x = max(1.0, constraint_size + 0.5)  # 确保约束完全显示
        margin_y = max(1.0, constraint_size + 0.5)
        
        # 计算弯矩图可能的最大范围
        max_moment_extent = self.l / 4  # AB段弯矩线最大可能延伸
        
        # 调整坐标轴范围，确保弯矩图和所有元素完整显示
        ax.set_xlim(-max(margin_x, max_moment_extent + 0.5), self.l + margin_x)
        ax.set_ylim(-margin_y, self.l + max(margin_y * 1.3, self.l/4))  # 确保载常数图的抛物线也能显示
        
        # 添加坐标轴标签
        ax.set_xlabel('X 坐标 (m)', fontsize=10)
        ax.set_ylabel('Y 坐标 (m)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 强制设置纵横比为1:1
        ax.set_aspect('equal')

    def draw_result_diagram(self, ax):
        """绘制结果图（弯矩图）"""
        ax.clear()
        ax.set_title("弯矩分析结果", fontsize=14, fontweight='bold')
        self._draw_base_structure(ax)

        # 绘制BC段抛物线弯矩图
        x_coords_bc = np.array([0, 0.5 * self.l, self.l])
        y_coords_bc = np.array([self.l + self.scale_factor * self.q * self.l ** 2 / 24,
                                self.l - self.scale_factor * 3 * self.q * self.l ** 2 / 48,
                                self.l + self.scale_factor * self.q * self.l ** 2 / 12])
        p_bc = np.poly1d(np.polyfit(x_coords_bc, y_coords_bc, 2))
        x_smooth_bc = np.linspace(0, self.l, 100)
        y_smooth_bc = p_bc(x_smooth_bc)

        ax.plot(x_smooth_bc, y_smooth_bc, 'r-', linewidth=1.5)
        ax.plot([self.l, self.l], [self.l, y_coords_bc[2]], 'r-', linewidth=1.5)
        ax.plot([0, 0], [self.l, y_coords_bc[0]], 'r-', linewidth=1.5)
        ax.fill_between(x_smooth_bc, self.l, y_smooth_bc, color='red', alpha=0.2, hatch='|')

        # 绘制AB段线性弯矩图
        moment_b_ab = -self.scale_factor * self.q * self.l ** 2 / 24
        ax.plot([0, moment_b_ab], [0, self.l], 'r-', linewidth=1.5)
        ax.plot([0, moment_b_ab], [self.l, self.l], 'r-', linewidth=1.5)
        ax.fill([0, moment_b_ab, 0], [0, self.l, self.l], color='red', alpha=0.2, hatch='|')

        # 添加弯矩值标注
        val_c = self.q * self.l ** 2 / 12
        val_b = self.q * self.l ** 2 / 24
        
        # 添加带背景的文本标注，使其更清晰
        text_props = dict(
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'),
            fontsize=10
        )
        
        ax.text(self.l, y_coords_bc[2] + 0.1, f'{val_c:.2f} kN·m', ha='center', va='bottom', **text_props)
        ax.text(0, y_coords_bc[0] + 0.1, f'{val_b:.2f} kN·m', ha='right', va='bottom', **text_props)
        ax.text(moment_b_ab - 0.1, self.l, f'{val_b:.2f} kN·m', ha='right', va='center', **text_props)
        
        self._add_point_labels(ax)

        # 设置坐标轴和背景
        self._set_result_axes_properties(ax, moment_b_ab, y_coords_bc)
        
    def _set_result_axes_properties(self, ax, moment_b_ab, y_coords_bc):
        """设置结果图坐标轴属性"""
        # 设置背景色
        ax.set_facecolor('#f9fafb')
        
        # 计算适当的坐标轴范围
        x_min = min(moment_b_ab * 1.2, -self.l * 0.15)
        x_max = self.l * 1.15
        y_max = max(y_coords_bc[2] * 1.1, self.l * 1.15)
        y_min = -self.l * 0.15
        
        # 确保图形填充整个画布
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # 添加坐标轴标签和网格
        ax.set_xlabel('X 坐标 (m)', fontsize=10)
        ax.set_ylabel('Y 坐标 (m)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 添加弯矩图标题和缩放因子信息
        ax.text(self.l/2, y_max * 0.95, f'弯矩图 (缩放系数: {self.scale_factor:.4f})',
                fontsize=12, ha='center', va='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='#4f46e5', boxstyle='round,pad=0.3'))

    def draw_load_constant_diagram(self, ax):
        """绘制载常数图"""
        ax.clear()
        ax.set_title("载常数图", fontsize=14, fontweight='bold')
        
        # 定义关键点坐标
        A = np.array([0, 0])
        B = np.array([0, self.l])
        C = np.array([self.l, self.l])
        
        # 绘制主杆件ABC
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k-', linewidth=3, label='主杆件')
        ax.plot([B[0], C[0]], [B[1], C[1]], 'k-', linewidth=3)
        
        # 标注点A, B, C
        ax.text(A[0]-0.3, A[1]-0.2, 'A', fontsize=14, fontweight='bold')
        ax.text(B[0]-0.3, B[1]+0.1, 'B', fontsize=14, fontweight='bold')
        ax.text(C[0]+0.1, C[1]+0.1, 'C', fontsize=14, fontweight='bold')
        
        # 定义三个控制点用于抛物线拟合
        point1 = B + np.array([0, self.l/5])  # B点上方l/5处
        point3 = C + np.array([0, self.l/5])  # C点上方l/5处
        bc_midpoint = (B + C) / 2  # BC中点
        point2 = bc_midpoint + np.array([0, -self.l/11])  # BC中点下方l/11处
        
        # 绘制连接线
        ax.plot([B[0], point1[0]], [B[1], point1[1]], 'r-', linewidth=2)
        ax.plot([point3[0], C[0]], [point3[1], C[1]], 'r-', linewidth=2)
        
        # 使用三点拟合二次抛物线
        x_coords = np.array([point1[0], point2[0], point3[0]])
        y_coords = np.array([point1[1], point2[1], point3[1]])
        
        # 拟合二次多项式
        coeffs = np.polyfit(x_coords, y_coords, 2)
        
        # 生成平滑的抛物线点
        x_smooth = np.linspace(B[0], C[0], 100)
        y_smooth = np.polyval(coeffs, x_smooth)
        
        # 绘制红色抛物线 - 确保稳定显示
        ax.plot(x_smooth, y_smooth, 'r-', linewidth=2.5, label='弯矩线', solid_capstyle='round')
        
        # 填充抛物线下方区域 - 增强显示效果
        fill_x = np.concatenate([x_smooth, [C[0], B[0]]])
        fill_y = np.concatenate([y_smooth, [C[1], B[1]]])
        ax.fill(fill_x, fill_y, 'red', alpha=0.4, edgecolor='red', linewidth=1)
        
        # 在B上方点处标注ql^2/12的具体数值 - 确保稳定显示
        ql2_12_value = self.q * self.l**2 / 12
        ax.text(point1[0] - 0.3, point1[1] + 0.2, f'{ql2_12_value:.2f} kN·m', 
                fontsize=12, color='red', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', linewidth=1, boxstyle='round,pad=0.3'))
        
        # 绘制约束
        self._draw_constraints(ax, A, C)
        
        # 设置坐标轴 - 载常数图专用
        self._set_load_constant_axes_properties(ax, point1, point2, point3)

    def draw_shape_constant_diagram(self, ax):
        """绘制形常数图"""
        ax.clear()
        ax.set_title("形常数图", fontsize=14, fontweight='bold')
        
        # 定义关键点坐标
        A = np.array([0, 0])
        B = np.array([0, self.l])
        C = np.array([self.l, self.l])
        
        # 绘制主杆件ABC
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k-', linewidth=3, label='主杆件')
        ax.plot([B[0], C[0]], [B[1], C[1]], 'k-', linewidth=3)
        
        # 标注点A, B, C
        ax.text(A[0]-0.3, A[1]-0.2, 'A', fontsize=14, fontweight='bold')
        ax.text(B[0]-0.3, B[1]+0.1, 'B', fontsize=14, fontweight='bold')
        ax.text(C[0]+0.1, C[1]+0.1, 'C', fontsize=14, fontweight='bold')
        
        # 绘制AB段弯矩线（在AB左侧）
        AB_length = self.l
        moment_point_AB = np.array([B[0] - AB_length/5, B[1]])
        
        # 确保弯矩线稳定显示 - AB段
        ab_x_coords = [A[0], moment_point_AB[0], B[0]]
        ab_y_coords = [A[1], moment_point_AB[1], B[1]]
        
        # 绘制AB段弯矩线轮廓
        ax.plot(ab_x_coords, ab_y_coords, 'r-', linewidth=2.5, solid_capstyle='round')
        # 绘制AB段填充
        ax.fill(ab_x_coords, ab_y_coords, 'red', alpha=0.4, edgecolor='red', linewidth=1)
        
        # 标注3i（AB段）- 使用具体数值
        i_value = 9.45 * 10**7 /self.l
        three_i_value = 3 * i_value
        ax.text(moment_point_AB[0] - 0.3, moment_point_AB[1] + 0.2, f'3i = {three_i_value:.2e}', 
                fontsize=12, color='red', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.2'))
        
        # 绘制BC段弯矩线（在BC下侧）
        BC_length = self.l
        moment_point_BC = np.array([B[0], B[1] - BC_length/5])
        
        # 确保弯矩线稳定显示 - BC段
        bc_x_coords = [B[0], moment_point_BC[0], C[0]]
        bc_y_coords = [B[1], moment_point_BC[1], C[1]]
        
        # 绘制BC段弯矩线轮廓
        ax.plot(bc_x_coords, bc_y_coords, 'r-', linewidth=2.5, solid_capstyle='round')
        # 绘制BC段填充
        ax.fill(bc_x_coords, bc_y_coords, 'red', alpha=0.4, edgecolor='red', linewidth=1)
        
        # 标注3i（BC段）- 使用具体数值
        i_value = 9.45 * 10**7
        three_i_value = 3 * i_value/self.l
        ax.text(moment_point_BC[0], moment_point_BC[1] - 0.3, f'3i = {three_i_value:.2e}', 
                fontsize=12, color='red', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.2'))
        
        # 在B点添加1/4圆弧和箭头
        arc_radius = min(0.4, self.l * 0.08)  # 根据杆件长度调整圆弧大小
        arc = Arc(B, 2*arc_radius, 2*arc_radius, angle=0, theta1=0, theta2=90, color='blue', linewidth=2)
        ax.add_patch(arc)
        
        # 在圆弧右侧添加箭头
        arrow_start = B + np.array([arc_radius * np.cos(np.radians(10)), arc_radius * np.sin(np.radians(10))])
        arrow_end = B + np.array([arc_radius * np.cos(np.radians(0)), arc_radius * np.sin(np.radians(0))])
        arrow_head = FancyArrowPatch(arrow_start, arrow_end,
                                    arrowstyle='->', mutation_scale=12, color='blue', linewidth=2)
        ax.add_patch(arrow_head)
        
        # 绘制约束
        self._draw_constraints(ax, A, C)
        
        # 设置坐标轴 - 形常数图专用
        self._set_shape_constant_axes_properties(ax, moment_point_AB, moment_point_BC)

    def _set_load_constant_axes_properties(self, ax, point1, point2, point3):
        """设置载常数图坐标轴属性"""
        # 设置背景色
        ax.set_facecolor('#f9fafb')
        
        # 计算所有关键点的范围
        all_x = [0, self.l, point1[0], point2[0], point3[0]]
        all_y = [0, self.l, point1[1], point2[1], point3[1]]
        
        # 计算约束尺寸
        constraint_size = min(0.3, self.l * 0.05)
        
        # 计算边距，确保所有元素都能显示
        margin_x = max(1.0, constraint_size + 0.5)
        margin_y = max(1.0, constraint_size + 0.5, (max(all_y) - self.l) + 0.3)  # 确保抛物线顶部显示
        
        # 设置坐标轴范围
        ax.set_xlim(-margin_x, self.l + margin_x)
        ax.set_ylim(-margin_y, max(all_y) + margin_y)
        
        # 添加坐标轴标签
        ax.set_xlabel('X 坐标 (m)', fontsize=10)
        ax.set_ylabel('Y 坐标 (m)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 强制设置纵横比为1:1
        ax.set_aspect('equal')

    def _set_shape_constant_axes_properties(self, ax, moment_point_AB, moment_point_BC):
        """设置形常数图坐标轴属性"""
        # 设置背景色
        ax.set_facecolor('#f9fafb')
        
        # 计算所有关键点的范围
        all_x = [0, self.l, moment_point_AB[0], moment_point_BC[0]]
        all_y = [0, self.l, moment_point_AB[1], moment_point_BC[1]]
        
        # 计算约束尺寸
        constraint_size = min(0.3, self.l * 0.05)
        
        # 计算边距，确保弯矩图完全显示
        margin_x = max(1.0, constraint_size + 0.5, abs(min(all_x)) + 0.3)  # 确保左侧弯矩图显示
        margin_y = max(1.0, constraint_size + 0.5, abs(min(all_y) - 0) + 0.3)  # 确保下方弯矩图显示
        
        # 设置坐标轴范围
        ax.set_xlim(min(all_x) - margin_x, self.l + margin_x)
        ax.set_ylim(min(all_y) - margin_y, self.l + margin_y)
        
        # 添加坐标轴标签
        ax.set_xlabel('X 坐标 (m)', fontsize=10)
        ax.set_ylabel('Y 坐标 (m)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 强制设置纵横比为1:1
        ax.set_aspect('equal')

    def _draw_constraints(self, ax, A, C):
        """绘制A点和C点的约束"""
        # 根据杆件长度调整约束尺寸
        constraint_size = min(0.3, self.l * 0.05)
        circle_radius = min(0.06, self.l * 0.01)
        
        # A点约束
        circle_A = Circle(A, circle_radius, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle_A)
        
        # A点左侧约束
        support_left = A + np.array([-constraint_size, 0])
        ax.plot([A[0], support_left[0]], [A[1], support_left[1]], 'k-', linewidth=2)
        circle_A_left = Circle(support_left, circle_radius, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle_A_left)
        
        # A点下侧约束
        support_down = A + np.array([0, -constraint_size])
        ax.plot([A[0], support_down[0]], [A[1], support_down[1]], 'k-', linewidth=2)
        circle_A_down = Circle(support_down, circle_radius, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle_A_down)
        
        # C点约束
        circle_C = Circle(C, circle_radius, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle_C)
        
        # C点右约束
        support_right = C + np.array([constraint_size, 0])
        ax.plot([C[0], support_right[0]], [C[1], support_right[1]], 'k-', linewidth=2)
        circle_C_support = Circle(support_right, circle_radius, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle_C_support)
        
        # C点下侧约束
        support_more = C + np.array([0, -constraint_size])
        ax.plot([C[0], support_more[0]], [C[1], support_more[1]], 'k-', linewidth=2)
        circle_C_right = Circle(support_more, circle_radius, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle_C_right)


class DisplacementMethodApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("位移法结构分析系统")
        self.resize(1400, 900)  # 增加窗口大小
        self.setStyleSheet(APP_STYLE)
        
        # 创建中央部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)
        
        # 左侧控制面板
        left_panel = QFrame()
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(15)
        
        # 标题
        title_label = QLabel("位移法教学辅助程序")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #1e40af;")
        left_layout.addWidget(title_label)
        
        # 参数组
        param_group = QGroupBox("结构参数")
        param_layout = QFormLayout()
        param_layout.setVerticalSpacing(10)
        param_layout.setHorizontalSpacing(10)
        
        # 创建带单位的输入行
        def create_input_row(unit, default_value):
            layout = QHBoxLayout()
            input_field = QLineEdit(default_value)
            input_field.setFixedWidth(120)
            layout.addWidget(input_field)
            layout.addWidget(QLabel(unit))
            return layout, input_field
        
        # 杆件长度 l
        l_layout, self.l_edit = create_input_row("m", "4.0")
        param_layout.addRow("杆件长度 l:", l_layout)

        # 均布荷载 q
        q_layout, self.q_edit = create_input_row("kN/m", "1.0")
        param_layout.addRow("均布荷载 q:", q_layout)
        
        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)
        
        # 视图控制组
        view_group = QGroupBox("视图控制")
        view_layout = QVBoxLayout()
        view_layout.setSpacing(15)
        
        # 缩放模式选择
        scale_mode_layout = QHBoxLayout()
        scale_mode_layout.addWidget(QLabel("缩放模式:"))
        
        self.scale_combo = QComboBox()
        self.scale_combo.addItem("自动缩放", None)
        self.scale_combo.addItem("大图 (0.5)", 0.5)
        self.scale_combo.addItem("中图 (0.05)", 0.05)
        self.scale_combo.addItem("小图 (0.01)", 0.01)
        self.scale_combo.addItem("微图 (0.001)", 0.001)
        self.scale_combo.currentIndexChanged.connect(self.on_scale_changed)
        scale_mode_layout.addWidget(self.scale_combo)
        
        view_layout.addLayout(scale_mode_layout)
        
        # 自定义缩放输入
        custom_scale_layout = QHBoxLayout()
        custom_scale_layout.addWidget(QLabel("自定义缩放:"))
        
        self.custom_scale_edit = QLineEdit("0.01")
        self.custom_scale_edit.setFixedWidth(80)
        custom_scale_layout.addWidget(self.custom_scale_edit)
        
        self.apply_scale_btn = QPushButton("应用")
        self.apply_scale_btn.setStyleSheet("""
            QPushButton {
                padding: 4px 8px;
                min-height: 25px;
                max-width: 60px;
            }
        """)
        self.apply_scale_btn.clicked.connect(self.apply_custom_scale)
        custom_scale_layout.addWidget(self.apply_scale_btn)
        
        view_layout.addLayout(custom_scale_layout)
        
        view_group.setLayout(view_layout)
        left_layout.addWidget(view_group)
        
        # 更新按钮
        self.update_btn = QPushButton("更新分析结果")
        self.update_btn.setFixedHeight(40)
        self.update_btn.setStyleSheet("font-size: 14px;")
        self.update_btn.clicked.connect(self.update_plot)
        left_layout.addWidget(self.update_btn)
        
        # 添加伸缩项
        left_layout.addStretch()
        
        # 右侧选项卡和绘图区
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建选项卡控件
        self.tab_widget = QTabWidget()
        
        # 原图选项卡
        self.original_tab = QWidget()
        original_layout = QVBoxLayout(self.original_tab)
        self.canvas_original = MplCanvas(self, width=10, height=8, dpi=100)
        original_layout.addWidget(self.canvas_original)
        self.tab_widget.addTab(self.original_tab, "结构示意图")
        
        # 载常数图选项卡
        self.load_constant_tab = QWidget()
        load_constant_layout = QVBoxLayout(self.load_constant_tab)
        self.canvas_load_constant = MplCanvas(self, width=10, height=8, dpi=100)
        load_constant_layout.addWidget(self.canvas_load_constant)
        self.tab_widget.addTab(self.load_constant_tab, "载常数图")
        
        # 形常数图选项卡
        self.shape_constant_tab = QWidget()
        shape_constant_layout = QVBoxLayout(self.shape_constant_tab)
        self.canvas_shape_constant = MplCanvas(self, width=10, height=8, dpi=100)
        shape_constant_layout.addWidget(self.canvas_shape_constant)
        self.tab_widget.addTab(self.shape_constant_tab, "形常数图")
        
        # 结果图选项卡
        self.result_tab = QWidget()
        result_layout = QVBoxLayout(self.result_tab)
        self.canvas_result = MplCanvas(self, width=10, height=8, dpi=100)
        result_layout.addWidget(self.canvas_result)
        self.tab_widget.addTab(self.result_tab, "弯矩分析结果")
        
        # 将选项卡添加到右侧面板
        right_layout.addWidget(self.tab_widget)
        
        # 添加左右面板
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)  # 右侧区域占据更多空间
        
        # 初始化参数
        self.scale_factor = None  # 使用自动缩放
        
        # 初始绘制
        self.update_plot()
    
    def on_scale_changed(self):
        """处理缩放模式选择变化"""
        self.scale_factor = self.scale_combo.currentData()
        self.update_plot()
    
    def apply_custom_scale(self):
        """应用自定义缩放值"""
        try:
            scale_value = float(self.custom_scale_edit.text())
            if scale_value > 0:
                self.scale_factor = scale_value
                # 更新下拉菜单为自定义选项
                self.scale_combo.setCurrentIndex(0)  # 先切换到自动缩放
                self.scale_combo.setItemText(0, f"自定义 ({scale_value:.4f})")
                self.scale_combo.setItemData(0, scale_value)
                self.update_plot()
        except ValueError:
            pass  # 忽略无效输入
    
    def update_plot(self):
        """更新图表"""
        try:
            l = float(self.l_edit.text())
            q = float(self.q_edit.text())
            
            # 限制输入值范围，防止极端值导致显示问题
            l = max(0.1, min(l, 1000))  # 限制长度在0.1到1000之间
            q = max(0.01, min(q, 10000))  # 限制荷载在0.01到10000之间
        except ValueError:
            # 如果输入无效，使用默认值
            l = 4.0
            q = 1.0
        
        # 创建框架图对象
        frame = FrameDiagram(l=l, q=q, scale_factor=self.scale_factor)
        
        # 如果使用的是自动缩放，更新自定义缩放输入框的值
        if self.scale_factor is None:
            self.custom_scale_edit.setText(f"{frame.scale_factor:.4f}")
            # 恢复下拉菜单的自动缩放文本
            self.scale_combo.setItemText(0, "自动缩放")
            self.scale_combo.setItemData(0, None)
        
        # 绘制所有图表
        frame.draw_original_diagram(self.canvas_original.axes)
        frame.draw_result_diagram(self.canvas_result.axes)
        frame.draw_load_constant_diagram(self.canvas_load_constant.axes)
        frame.draw_shape_constant_diagram(self.canvas_shape_constant.axes)
        
        # 更新Figure的紧凑布局
        self.canvas_original.fig.tight_layout(pad=0.8)
        self.canvas_result.fig.tight_layout(pad=0.8)
        self.canvas_load_constant.fig.tight_layout(pad=0.8)
        self.canvas_shape_constant.fig.tight_layout(pad=0.8)
        
        # 强制刷新画布，确保弯矩图稳定显示
        self.canvas_original.draw_idle()
        self.canvas_result.draw_idle()
        self.canvas_load_constant.draw_idle()
        self.canvas_shape_constant.draw_idle()
        
        # 立即刷新
        self.canvas_original.draw()
        self.canvas_result.draw()
        self.canvas_load_constant.draw()
        self.canvas_shape_constant.draw()
        
        # 更新窗口标题显示当前参数
        self.setWindowTitle(f"位移法结构分析系统 - 杆长: {l}m, 荷载: {q}kN/m")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 使用Fusion样式以获得更好的外观
    window = DisplacementMethodApp()
    window.show()
    sys.exit(app.exec_())

