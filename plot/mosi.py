import matplotlib.pyplot as plt
import numpy as np

# 1. 数据准备 (排除了 MAE 指标)
models = ['TFN', 'LMF', 'MFN', 'MulT', 'MISA', 'Self-MM', 'TETFN', 'HTRN', 'MOE-AHL']
acc7 = [34.4, 33.8, 35.8, 36.9, 41.3, 46.6, 45.7, 45.9, 48.5]
acc5 = [39.3, 38.1, 40.4, 42.6, 47.0, 53.4, 53.6, 52.0, 55.1]
acc2 = [79.0, 79.1, 78.8, 80.9, 83.5, 85.4, 85.3, 86.3, 85.7]
f1 = [79.1, 79.1, 78.9, 80.9, 83.5, 85.4, 85.3, 86.4, 85.7]
corr = [0.673, 0.651, 0.670, 0.702, 0.778, 0.796, 0.798, 0.794, 0.794]

x = np.arange(len(models))  # 模型标签的位置
width = 0.2  # 柱子的宽度

# 创建画布和主坐标轴 (保持与 MOSEI 相同的画布宽度)
fig, ax1 = plt.subplots(figsize=(14, 6))

# 2. 在主 Y 轴绘制分组柱状图 (Acc-7, Acc-5, Acc-2, F1)
rects1 = ax1.bar(x - 1.5*width, acc7, width, label='Acc-7')
rects2 = ax1.bar(x - 0.5*width, acc5, width, label='Acc-5')
rects3 = ax1.bar(x + 0.5*width, acc2, width, label='Acc-2')
rects4 = ax1.bar(x + 1.5*width, f1, width, label='F1')

# 设置主 Y 轴相关属性
ax1.set_ylabel('Percentage / Score')
ax1.set_title('Experimental results of comparison on the MOSI dataset')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend(loc='upper left')

# 3. 创建副 Y 轴并绘制 Corr 的折线图
ax2 = ax1.twinx()
ax2.plot(x, corr, color='black', marker='o', linestyle='-', linewidth=2, label='Corr')

# 设置副 Y 轴相关属性
ax2.set_ylabel('Corr', color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.legend(loc='upper right')

# 4. 优化布局并显示/保存图表
plt.tight_layout()
plt.savefig('mosi.png', dpi=300)
plt.show()