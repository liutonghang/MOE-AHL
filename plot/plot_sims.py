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
metrics = ['Acc-5', 'Acc-3', 'Acc-2', 'F1', 'Corr']

# 各模型的指标数值（均为小数，0-1区间）
models_data = {
    'EL_LSTM': [0.2123, 0.5427, 0.6937, 0.5682, 0.545],
    'LF_DNN': [0.3974, 0.6433, 0.7702, 0.7727, 0.555],
    'TFN': [0.3930, 0.6512, 0.7838, 0.7862, 0.591],
    'LMF': [0.4053, 0.6468, 0.7777, 0.7788, 0.575],
    'MFN': [0.3947, 0.6573, 0.7790, 0.7788, 0.582],
    'MulT': [0.3794, 0.6477, 0.7856, 0.7966, 0.564],
    'Self-MM': [0.4153, 0.6547, 0.8004, 0.8044, 0.595],
    'MOE-AHL': [0.4814, 0.6695, 0.8140, 0.8141, 0.593]
}

# 定义配色和标记（8个模型对应不同样式，避免混淆）
colors = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12',
    '#9b59b6', '#1abc9c', '#e67e22', '#34495e'
]
markers = ['o', 's', '^', '*', 'x', 'D', 'p', 'h']

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
        ax.text(i, score + 0.008, f'{score:.4f}',
                ha='center', va='bottom', fontsize=7,
                color=colors[idx], alpha=0.8)

# ===================== 4. 图表美化 =====================
# 坐标轴设置
ax.set_xlabel('评价指标', fontsize=14, labelpad=12, fontweight='medium')
ax.set_ylabel('值', fontsize=13, labelpad=10)
ax.set_ylim(0.2, 0.9)  # Y轴范围适配所有指标（0.2~0.9）
ax.tick_params(axis='both', labelsize=11)

# 网格线（增强可读性）
ax.grid(True, alpha=0.25, linestyle=':', linewidth=1)

# 图例（放在右侧，避免遮挡折线）
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
          fontsize=10, frameon=True, shadow=True,
          ncol=1, framealpha=0.9)

# 标题
ax.set_title('CH-SIMS',
             fontsize=16, pad=20, fontweight='bold')

# 调整布局（防止标签/图例被截断）
plt.tight_layout()

# ===================== 5. 保存图片 + 显示 =====================
# 保存高清图片（当前目录，可修改路径）
plt.savefig('model_metrics_decimal_smis.png', dpi=300, bbox_inches='tight')
print("图片已保存为：model_metrics_decimal_sims.png")

# 显示图表
plt.show()