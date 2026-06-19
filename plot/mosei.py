import matplotlib.pyplot as plt
import numpy as np

# 1. 数据准备 (排除了 MAE 指标)
models = ['TFN', 'LMF', 'MFN', 'MulT', 'MISA', 'Self-MM', 'TETFN', 'HTRN', 'MOE-AHL']
acc7 = [51.6, 51.5, 51.3, 52.8, 52.0, 53.8, 53.9, 52.9, 54.7]
acc5 = [53.1, 52.9, 52.7, 54.1, 53.6, 55.5, 55.7, 54.9, 56.4]
acc2 = [81.8, 83.4, 82.8, 84.6, 84.6, 85.1, 86.2, 86.4, 86.3]
f1 = [81.7, 83.3, 82.8, 84.5, 84.6, 84.9, 86.1, 86.5, 86.1]
corr = [0.714, 0.716, 0.718, 0.733, 0.751, 0.764, 0.769, 0.778, 0.775]

x = np.arange(len(models))  # 模型标签的位置
width = 0.2  # 柱子的宽度

# 创建画布和主坐标轴 (因为模型多了一个，这里稍微加宽了画布到 14)
fig, ax1 = plt.subplots(figsize=(14, 6))

# 2. 在主 Y 轴绘制分组柱状图 (Acc-7, Acc-5, Acc-2, F1)
rects1 = ax1.bar(x - 1.5*width, acc7, width, label='Acc-7')
rects2 = ax1.bar(x - 0.5*width, acc5, width, label='Acc-5')
rects3 = ax1.bar(x + 0.5*width, acc2, width, label='Acc-2')
rects4 = ax1.bar(x + 1.5*width, f1, width, label='F1')

# 设置主 Y 轴相关属性
ax1.set_ylabel('Percentage / Score')
ax1.set_title('Experimental results of comparison on the MOSEI dataset')
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
plt.savefig('mosei.png', dpi=300)
plt.show()