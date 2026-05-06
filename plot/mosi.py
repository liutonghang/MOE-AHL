import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. 指定默认字体为黑体（SimHei）或其他系统中已安装的中文基础字体
plt.rcParams['font.sans-serif'] = ['SimHei']

# 2. 解决坐标轴负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd
import numpy as np

# 1. 填入你的新数据
data = {
    'Model': ['EL_LSTM', 'LF_DNN', 'TFN', 'LMF', 'MFN', 'MulT', 'MISA', 'Self-MM', 'MOE-AHL'],
    'Acc-7': [35.39, 34.52, 34.46, 33.82, 35.8, 36.91, 41.37, 46.67, 48.54],
    'Acc-5': [40.15, 38.05, 39.39, 38.13, 40.47, 42.68, 47.08, 53.47, 55.10],
    'Acc-2': [78.48, 78.63, 79.08, 79.18, 78.87, 80.98, 83.54, 85.46, 85.74],
    'F1': [78.51, 78.63, 79.11, 79.15, 78.90, 80.95, 83.58, 85.43, 85.74],
    'Corr': [0.669, 0.658, 0.673, 0.651, 0.670, 0.702, 0.778, 0.796, 0.7948]
}

df = pd.DataFrame(data)

# 2. 设置绘图风格 (针对新数据微调了布局)
fig, ax1 = plt.subplots(figsize=(13, 7))
x = np.arange(len(df['Model']))
width = 0.18

# 3. 绘制柱状图 (Acc-7, Acc-5, Acc-2, F1)
# 使用了更现代的配色方案
rects1 = ax1.bar(x - 1.5*width, df['Acc-7'], width, label='Acc-7', color='#3498db', zorder=3)
rects2 = ax1.bar(x - 0.5*width, df['Acc-5'], width, label='Acc-5', color='#2ecc71', zorder=3)
rects3 = ax1.bar(x + 0.5*width, df['Acc-2'], width, label='Acc-2', color='#e74c3c', zorder=3)
rects4 = ax1.bar(x + 1.5*width, df['F1'], width, label='F1', color='#f1c40f', zorder=3)

# 设置左轴属性
ax1.set_ylabel('数值 / 百分比 (%)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.set_xticks(x)
ax1.set_xticklabels(df['Model'], rotation=30, ha='right')
ax1.legend(loc='upper left', ncol=2, frameon=True)
ax1.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# 4. 绘制 Corr 折线图 (使用双 Y 轴)
ax2 = ax1.twinx()
ax2.plot(x, df['Corr'], color='#8e44ad', marker='D', markersize=6, linewidth=2.5,
         label='Corr (右轴)', markeredgecolor='white', zorder=5)

# 设置右轴属性 (根据新数据的 Corr 范围 0.65-0.80 进行了优化)
ax2.set_ylabel('相关系数 (Corr)', fontsize=12, color='#8e44ad', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#8e44ad')
ax2.set_ylim(0.60, 0.85)
ax2.legend(loc='upper right')

# 5. 标题与保存
plt.title('Multi-modal Model Performance Comparison with MOSI', fontsize=15, fontweight='bold', pad=20)
fig.tight_layout()

# 如需保存图片，取消下面这行的注释
plt.savefig('mosi.png', dpi=300)
plt.show()