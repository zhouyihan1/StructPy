import numpy as np
import pandas as pd

# 参数
h = 4
l = 6
e = 2.1e5
i = 1.4e4
ei = e * i * 1e-6

def calculate_displacement(F, q):
    """向量化计算位移"""
    FB = 0.5 * q * l + 0.5 * F * h * l
    displacement = (
        0.5 * l * h * 2 / 3 * FB * l +
        2 / 3 * l * q * l ** 2 / 8 * 0.5 * h +
        0.5 * l * h * F * h * 2 / 3 +
        h * 0.25 * F * h * 0.5 * 0.5 * h
    ) / ei
    return displacement

# ==================== 分层采样函数 ====================
def stratified_sampling(low, high, n_samples, n_bins=10):
    """在 [low, high] 范围内分层均匀采样，避免数据集中"""
    bins = np.linspace(low, high, n_bins + 1)
    samples = []
    per_bin = n_samples // n_bins
    for i in range(n_bins):
        samples.append(np.random.uniform(bins[i], bins[i+1], per_bin))
    return np.concatenate(samples)

# ==================== 生成核心训练集 ====================
print("生成核心训练数据...")
core_samples = 10000
F_core = stratified_sampling(1, 1000, core_samples)
q_core = stratified_sampling(1, 1000, core_samples)

# 向量化计算
disp_core = calculate_displacement(F_core, q_core)

# ==================== 生成补充训练集 ====================
print("生成补充训练数据...")
supplementary_samples = 2000
F_supp = stratified_sampling(1000, 5000, supplementary_samples)
q_supp = stratified_sampling(1000, 5000, supplementary_samples)

disp_supp = calculate_displacement(F_supp, q_supp)

# ==================== 合并并保存 ====================
F_all = np.concatenate([F_core, F_supp])
q_all = np.concatenate([q_core, q_supp])
disp_all = np.concatenate([disp_core, disp_supp])

df = pd.DataFrame({
    "Load_F": F_all,
    "q": q_all,
    "Displacement": disp_all
})

df.to_csv("data.csv", index=False)
print(f"训练集 data.csv 生成成功，共包含 {len(df)} 条数据。")
