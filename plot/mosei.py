import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 准备数据 (已加入 TETFN 模型数据)
data = {
    'Model': ['EL_LSTM', 'LF_DNN', 'TFN', 'LMF', 'MFN', 'MulT', 'MISA', 'Self-MM', 'TETFN', 'MOE-AHL'],
    'Acc-7': [50.01, 50.83, 51.6, 51.59, 51.34, 52.84, 52.05, 53.87, 53.90, 54.73],
    'Acc-5': [51.16, 51.97, 53.1, 52.99, 52.76, 54.18, 53.63, 55.53, 55.78, 56.45],
    'Acc-2': [80.79, 82.74, 81.89, 83.48, 82.86, 84.63, 84.67, 85.15, 86.21, 86.32],
    'F1': [80.67, 82.52, 81.74, 83.36, 82.85, 84.52, 84.66, 84.90, 86.11, 86.13],
    'Corr': [0.682, 0.708, 0.714, 0.716, 0.718, 0.733, 0.751, 0.764, 0.769, 0.775]
}

df = pd.DataFrame(data)

# 2. 设置绘图参数
fig, ax1 = plt.subplots(figsize=(12, 7))
x = np.arange(len(df['Model']))
width = 0.18  # 柱状图宽度

# 3. 绘制 Acc7, Acc5, Acc2, F1 (使用左侧 Y 轴)
rects1 = ax1.bar(x - 1.5*width, df['Acc-7'], width, label='Acc-7', color='#3498db', zorder=3)
rects2 = ax1.bar(x - 0.5*width, df['Acc-5'], width, label='Acc-5', color='#2ecc71', zorder=3)
rects3 = ax1.bar(x + 0.5*width, df['Acc-2'], width, label='Acc-2', color='#e74c3c', zorder=3)
rects4 = ax1.bar(x + 1.5*width, df['F1'], width, label='F1', color='#f1c40f', zorder=3)

# 设置左轴标签和范围
ax1.set_ylabel('Percentage (%) / F1 Score', fontsize=12)
ax1.set_ylim(0, 105) # 留出顶部空间放图例
ax1.set_xticks(x)
ax1.set_xticklabels(df['Model'], rotation=45)
ax1.legend(loc='upper left', ncol=2)
ax1.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# 4. 绘制 Corr (使用右侧 Y 轴)
ax2 = ax1.twinx()
ax2.plot(x, df['Corr'], color='#8e44ad', marker='o', linewidth=2, label='Corr (Right Axis)', zorder=5)

# 设置右轴标签
ax2.set_ylabel('Correlation (Corr)', fontsize=12, color='#8e44ad')
ax2.tick_params(axis='y', labelcolor='#8e44ad')
ax2.set_ylim(0.6, 0.85)
ax2.legend(loc='upper right')

# 5. 标题与布局优化
plt.title('Multi-modal Model Performance Comparison with MOSEI Database', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

# 保存或显示
plt.savefig('mosei.png', dpi=300)
# plt.show()