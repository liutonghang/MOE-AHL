import matplotlib.pyplot as plt
import numpy as np

# ===================== 1. 全局环境配置 =====================
import matplotlib

matplotlib.use('TkAgg')  # 解决渲染/显示报错

# 字体配置（兼容中英文，避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # Windows
# plt.rcParams['font.sans-serif'] = ['PingFang SC', 'DejaVu Sans']  # Mac
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']  # Linux
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# ===================== 2. 数据准备（小数格式+独立横坐标） =====================
data = {
    'CH-SIMS': {
        'metrics': ['Acc-5', 'Acc-3', 'Acc-2', 'F1', 'Corr'],  # 专属横坐标
        'AHL': [0.4376, 0.6652, 0.7982, 0.7982, 0.602],
        'MOE-AHL': [0.4814, 0.6695, 0.8140, 0.8141, 0.593]
    },
    'MOSEI': {
        'metrics': ['Acc-7', 'Acc-5', 'Acc-2', 'F1', 'Corr'],  # 专属横坐标
        'AHL': [0.5327, 0.5503, 0.8539, 0.8536, 0.777],
        'MOE-AHL': [0.5473, 0.5645, 0.8632, 0.8613, 0.775]
    },
    'MOSI': {
        'metrics': ['Acc-7', 'Acc-5', 'Acc-2', 'F1', 'Corr'],  # 专属横坐标
        'AHL': [0.4956, 0.5519, 0.8598, 0.8595, 0.7985],
        'MOE-AHL': [0.4854, 0.5510, 0.8574, 0.8574, 0.7948]
    }
}

# 固定样式（跨子图统一，便于对比）
styles = {
    'AHL': {'color': '#e74c3c', 'marker': 'o', 'linestyle': '-', 'offset': 0.008},
    'MOE-AHL': {'color': '#3498db', 'marker': 's', 'linestyle': '--', 'offset': 0.008}
}

# ===================== 3. 创建画布（1行3列子图） =====================
fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)
dataset_names = ['CH-SIMS', 'MOSEI', 'MOSI']

# ===================== 4. 循环绘制每个数据集 =====================
for idx, (dataset, ax) in enumerate(zip(dataset_names, axes)):
    # 获取当前数据集的横坐标和数值
    current_metrics = data[dataset]['metrics']
    ahl_scores = data[dataset]['AHL']
    moe_ahl_scores = data[dataset]['MOE-AHL']

    # 绘制AHL模型折线
    ax.plot(current_metrics, ahl_scores,
            color=styles['AHL']['color'],
            marker=styles['AHL']['marker'],
            linestyle=styles['AHL']['linestyle'],
            linewidth=2.2, markersize=8, label='AHL', alpha=0.85)

    # 绘制MOE-AHL模型折线
    ax.plot(current_metrics, moe_ahl_scores,
            color=styles['MOE-AHL']['color'],
            marker=styles['MOE-AHL']['marker'],
            linestyle=styles['MOE-AHL']['linestyle'],
            linewidth=2.2, markersize=8, label='MOE-AHL', alpha=0.85)

    # AHL数值标注（统一保留4位小数）
    for i, score in enumerate(ahl_scores):
        ax.text(i, score + styles['AHL']['offset'], f'{score:.4f}',
                ha='center', va='bottom', fontsize=8,
                color=styles['AHL']['color'], alpha=0.9)

    # MOE-AHL数值标注（统一保留4位小数）
    for i, score in enumerate(moe_ahl_scores):
        ax.text(i, score + styles['MOE-AHL']['offset'], f'{score:.4f}',
                ha='center', va='bottom', fontsize=8,
                color=styles['MOE-AHL']['color'], alpha=0.9)

    # ===================== 子图美化 =====================
    # 坐标轴
    ax.set_xlabel('Evaluation Metrics', fontsize=12, labelpad=10, fontweight='medium')
    if idx == 0:  # 仅第一个子图显示Y轴标签
        ax.set_ylabel('Value', fontsize=12, labelpad=10)

    # 动态适配Y轴范围（覆盖当前数据集所有数值，预留微小余量）
    all_scores = ahl_scores + moe_ahl_scores
    ax.set_ylim(min(all_scores) - 0.02, max(all_scores) + 0.02)

    # 横坐标标签优化（轻微旋转避免重叠）
    ax.set_xticklabels(current_metrics, rotation=15, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # 网格线
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=1)

    # 子图标题
    ax.set_title(f'{dataset} Ablation Experiment',
                 fontsize=14, pad=15, fontweight='bold')

    # 图例
    ax.legend(loc='upper right', fontsize=9, frameon=True, shadow=True, framealpha=0.9)

# ===================== 全局设置 =====================
# 总标题
fig.suptitle('Ablation Experiment Results (CH-SIMS/MOSEI/MOSI)',
             fontsize=16, y=0.98, fontweight='bold')

# 调整布局（防止标签/图例截断）
plt.tight_layout(rect=[0, 0, 1, 0.95])

# ===================== 保存+显示 =====================
plt.savefig('ablation_experiment_decimal_final.png', dpi=300, bbox_inches='tight')
print("图片已保存为：ablation_experiment_decimal_final.png")

# 显示图表
plt.show()