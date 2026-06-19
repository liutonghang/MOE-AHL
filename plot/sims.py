import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据（排除了 MAE 指标）
models = ['TFN', 'LMF', 'MFN', 'MulT', 'Self-MM', 'TETFN', 'HTRN', 'MOE-AHL']
acc5 = [39.30, 40.53, 39.47, 37.94, 41.53, 41.79, 43.98, 48.14]
acc3 = [65.12, 64.68, 65.73, 64.77, 65.47, 63.24, 66.71, 66.95]
acc2 = [78.38, 77.77, 77.90, 78.56, 80.04, 81.18, 80.31, 81.40]
f1 = [78.62, 77.88, 77.88, 79.66, 80.44, 80.24, 80.23, 81.41]
corr = [0.591, 0.575, 0.582, 0.564, 0.595, 0.576, 0.628, 0.593]

x = np.arange(len(models))  # 模型标签的位置
width = 0.2  # 柱子的宽度

# 创建画布和主坐标轴
fig, ax1 = plt.subplots(figsize=(12, 6))

# 2. 在主 Y 轴绘制分组柱状图 (Acc-5, Acc-3, Acc-2, F1)
rects1 = ax1.bar(x - 1.5*width, acc5, width, label='Acc-5')
rects2 = ax1.bar(x - 0.5*width, acc3, width, label='Acc-3')
rects3 = ax1.bar(x + 0.5*width, acc2, width, label='Acc-2')
rects4 = ax1.bar(x + 1.5*width, f1, width, label='F1')

# 设置主 Y 轴相关属性
ax1.set_ylabel('Percentage / Score')
ax1.set_title('Experimental results of comparison on the CH-SIMS dataset')
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
plt.savefig('sims.png', dpi=300)
plt.show()