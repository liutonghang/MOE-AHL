import matplotlib.pyplot as plt
import numpy as np

# 模型名称
models = ["HTRN", "TFN", "LMF", "MFN", "MulT", "MISA", "Self-MM", "TETFN", "MOE-AHL"]

# 数据（不含MAE）
acc7 = [45.9, 34.4, 33.8, 35.8, 36.9, 41.3, 46.6, 45.7, 48.5]
acc5 = [52.0, 39.3, 38.1, 40.4, 42.6, 47.0, 53.4, 53.6, 55.1]
acc2 = [86.3, 79.0, 79.1, 78.8, 80.9, 83.5, 85.4, 85.3, 85.7]
f1   = [86.4, 79.1, 79.1, 78.9, 80.9, 83.5, 85.4, 85.3, 85.7]
corr = [0.794, 0.673, 0.651, 0.670, 0.702, 0.778, 0.796, 0.798, 0.794]

x = np.arange(len(models))

plt.figure(figsize=(13, 6))

# ===== 折线 =====
plt.plot(x, acc7, marker='o', linewidth=1.8, label='Acc-7')
plt.plot(x, acc5, marker='o', linewidth=1.8, label='Acc-5')
plt.plot(x, acc2, marker='o', linewidth=1.8, label='Acc-2')
plt.plot(x, f1,   marker='o', linewidth=1.8, label='F1')
plt.plot(x, corr, marker='o', linewidth=1.8, label='Corr')

# ===== 强调 MOE-AHL =====
moe_idx = len(models) - 1

for y in [acc7, acc5, acc2, f1, corr]:
    plt.scatter(moe_idx, y[moe_idx], s=150)
    plt.annotate(
        "MOE-AHL",
        xy=(moe_idx, y[moe_idx]),
        xytext=(moe_idx - 2, y[moe_idx] + 0.5),
        arrowprops=dict(arrowstyle="->", linewidth=1)
    )

# ===== 坐标与样式 =====
plt.xticks(x, models, rotation=30)
plt.ylabel("Score")
plt.title("MOSI Dataset Comparison Results ")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

# ===== 保存图片 =====
plt.savefig("mosi.png", dpi=300, bbox_inches='tight')

plt.show()