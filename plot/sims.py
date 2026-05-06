import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import platform

# --- 1. 解决中文乱码问题 ---
def set_ch_font():
    system = platform.system()
    if system == "Windows":
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 微软雅黑
    elif system == "Darwin": # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei'] # Linux/通用黑体
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

set_ch_font()

# --- 2. 准备新数据 ---
data = {
    'Model': ['EL_LSTM', 'LF_DNN', 'TFN', 'LMF', 'MFN', 'MulT', 'Self-MM', 'TETFN', 'MOE-AHL'],
    'Acc-5': [21.23, 39.74, 39.30, 40.53, 39.47, 37.94, 41.53, 41.79, 48.14],
    'Acc-3': [54.27, 64.33, 65.12, 64.68, 65.73, 64.77, 65.47, 63.24, 66.95],
    'Acc-2': [69.37, 77.02, 78.38, 77.77, 77.9, 78.56, 80.04, 81.18, 81.40],
    'F1': [56.82, 77.27, 78.62, 77.88, 77.88, 79.66, 80.44, 80.24, 81.41],
    'Corr': [0.545, 0.555, 0.591, 0.575, 0.582, 0.564, 0.595, 0.576, 0.593]
}

df = pd.DataFrame(data)

# --- 3. 绘图配置 ---
fig, ax1 = plt.subplots(figsize=(14, 7))
x = np.arange(len(df['Model']))
width = 0.18 # 柱状图宽度

# 绘制左轴指标 (百分比类)
rects1 = ax1.bar(x - 1.5*width, df['Acc-5'], width, label='Acc-5', color='#3498db', edgecolor='white')
rects2 = ax1.bar(x - 0.5*width, df['Acc-3'], width, label='Acc-3', color='#2ecc71', edgecolor='white')
rects3 = ax1.bar(x + 0.5*width, df['Acc-2'], width, label='Acc-2', color='#e74c3c', edgecolor='white')
rects4 = ax1.bar(x + 1.5*width, df['F1'], width, label='F1', color='#f1c40f', edgecolor='white')

# 左轴美化
ax1.set_ylabel('得分 / 百分比 (%)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.set_xticks(x)
ax1.set_xticklabels(df['Model'], rotation=25)
ax1.legend(loc='upper left', ncol=2)
ax1.grid(axis='y', linestyle=':', alpha=0.5)

# --- 4. 绘制右轴指标 (Corr) ---
ax2 = ax1.twinx()
ax2.plot(x, df['Corr'], color='#8e44ad', marker='o', markersize=8,
         linewidth=2.5, label='Corr (右轴)', markeredgecolor='white')

# 右轴美化 (针对新 Corr 数据范围 0.54-0.60 优化)
ax2.set_ylabel('相关系数 (Corr)', fontsize=12, color='#8e44ad', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#8e44ad')
ax2.set_ylim(0.50, 0.65) # 调整范围让波动更明显
ax2.legend(loc='upper right')

# --- 5. 整体修饰 ---
plt.title('Multi-modal Model Performance Comparison with CH-SIMS', fontsize=15, fontweight='bold', pad=20)
fig.tight_layout()

# 如需保存图片，取消下面这行的注释
plt.savefig('sims.png', dpi=300)
plt.show()