import matplotlib.pyplot as plt
import numpy as np

# ===================== 1. 全局设置（解决后端和中文显示问题） =====================
# 切换为TkAgg后端（解决tostring_rgb报错）
import matplotlib

matplotlib.use('TkAgg')

# 设置中文字体（按需调整，避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
# plt.rcParams['font.sans-serif'] = ['PingFang SC']  # Mac
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # Linux
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ===================== 2. 准备数据（所有指标为小数，严格对应表格） =====================
# 横坐标：评价指标名称
metrics = ['Acc-7', 'Acc-5', 'Acc-2', 'F1', 'Corr']

# 各模型的指标数值（均为小数，0-1区间）
models_data = {
    'EL_LSTM': [0.3539, 0.4015, 0.7848, 0.7851, 0.669],
    'LF_DNN': [0.3452, 0.3805, 0.7863, 0.7863, 0.658],
    'TFN': [0.3446, 0.3939, 0.7908, 0.7911, 0.673],
    'LMF': [0.3382, 0.3813, 0.7918, 0.7915, 0.651],
    'MFN': [0.358, 0.4047, 0.7887, 0.7890, 0.670],
    'MulT': [0.3691, 0.4268, 0.8098, 0.8095, 0.702],
    'MISA': [0.4137, 0.4708, 0.8354, 0.8358, 0.778],
    'Self-MM': [0.4667, 0.5347, 0.8546, 0.8543, 0.796],
    'MOE-AHL': [0.4854, 0.5510, 0.8574, 0.8574, 0.7948]
}

# 定义配色和标记（9个模型对应不同样式，避免混淆）
colors = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12',
    '#9b59b6', '#1abc9c', '#e67e22', '#34495e', '#8e44ad'
]
markers = ['o', 's', '^', '*', 'x', 'D', 'p', 'h', 'v']

# ===================== 3. 绘图（单Y轴连续折线版） =====================
# 创建画布（适配多模型显示）
fig, ax = plt.subplots(figsize=(12, 7))

# -------------- 循环绘制每个模型的完整连续折线 --------------
for idx, (model_name, scores) in enumerate(models_data.items()):
    # 绘制连续折线
    ax.plot(metrics, scores,
            color=colors[idx],
            marker=markers[idx],
            linewidth=2.2,
            markersize=8,
            label=model_name,
            alpha=0.85)

    # 为每个数据点添加数值标注（可选，如需隐藏可注释此段）
    for i, score in enumerate(scores):
        ax.text(i, score + 0.008, f'{score:.4f}',  # 保留四位小数
                ha='center', va='bottom', fontsize=7,
                color=colors[idx], alpha=0.8)

# ===================== 4. 图表美化 =====================
# 坐标轴设置
ax.set_xlabel('评价指标', fontsize=14, labelpad=12, fontweight='medium')
ax.set_ylabel('值', fontsize=13, labelpad=10)
ax.set_ylim(0.3, 0.9)  # Y轴范围适配所有指标（0.3~0.9）
ax.tick_params(axis='both', labelsize=11)

# 网格线（增强可读性）
ax.grid(True, alpha=0.25, linestyle=':', linewidth=1)

# 图例（放在右侧，避免遮挡折线）
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
          fontsize=10, frameon=True, shadow=True,
          ncol=1, framealpha=0.9)

# 标题
ax.set_title('MOSI',
             fontsize=16, pad=20, fontweight='bold')

# 调整布局（防止标签/图例被截断）
plt.tight_layout()

# ===================== 5. 保存图片 + 显示 =====================
# 保存高清图片（当前目录，可修改路径）
plt.savefig('model_metrics_decimal_mosi.png', dpi=300, bbox_inches='tight')
print("图片已保存为：model_metrics_decimal_mosi.png")

# 显示图表
plt.show()