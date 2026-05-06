import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体（如果需要显示中文，请确保系统中已安装相应字体）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 数据准备 ---
# Corr已乘以100以对齐刻度，保持图表美观
data = {
    "CH-SIMS": {
        "metrics": ['Acc-5', 'Acc-3', 'Acc-2', 'F1', 'Corr'],
        "AHL": [43.76, 66.52, 79.82, 79.82, 60.2],
        "MOE-AHL": [48.14, 66.95, 81.40, 81.41, 59.3]
    },
    "MOSEI": {
        "metrics": ['Acc-7', 'Acc-5', 'Acc-2', 'F1', 'Corr'],
        "AHL": [53.27, 55.03, 85.39, 85.36, 77.7],
        "MOE-AHL": [54.73, 56.45, 86.32, 86.13, 77.5]
    },
    "MOSI": {
        "metrics": ['Acc-7', 'Acc-5', 'Acc-2', 'F1', 'Corr'],
        "AHL": [49.56, 55.19, 85.98, 85.95, 79.85],
        "MOE-AHL": [48.54, 55.10, 85.74, 85.74, 79.48]
    }
}

# --- 2. 绘图配置 ---
fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=100)
colors = ['#aec6cf', '#ffb347']
width = 0.38

# --- 3. 循环生成子图 ---
for i, (ds_name, values) in enumerate(data.items()):
    ax = axes[i]
    metrics = values["metrics"]
    x = np.arange(len(metrics))

    # 绘制柱状图
    rects1 = ax.bar(x - width / 2, values["AHL"], width, label='AHL', color=colors[0], edgecolor='white', linewidth=0.7)
    rects2 = ax.bar(x + width / 2, values["MOE-AHL"], width, label='MOE-AHL', color=colors[1], edgecolor='white',
                    linewidth=0.7)


    # 添加数字标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')


    autolabel(rects1)
    autolabel(rects2)

    # 设置标题和标签
    ax.set_title(f'{ds_name} Ablation Results', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 105)  # 留出顶部空间给数值标签

    if i == 0:
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.legend(frameon=True, loc='upper left')

    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('ablation_results.png', dpi=300)
plt.show()