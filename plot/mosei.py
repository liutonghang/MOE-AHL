import matplotlib.pyplot as plt
import numpy as np

# 模型名称
models = ["HTRN", "TFN", "LMF", "MFN", "MulT", "MISA", "Self-MM", "TETFN", "MOE-AHL"]

# 数据（不含MAE）
acc7 = [52.9, 51.6, 51.5, 51.3, 52.8, 52.0, 53.8, 53.9, 54.7]
acc5 = [54.9, 53.1, 52.9, 52.7, 54.1, 53.6, 55.5, 55.7, 56.4]
acc2 = [86.4, 81.8, 83.4, 82.8, 84.6, 84.6, 85.1, 86.2, 86.3]
f1   = [86.5, 81.7, 83.3, 82.8, 84.5, 84.6, 84.9, 86.1, 86.1]
corr = [0.778, 0.714, 0.716, 0.718, 0.733, 0.751, 0.764, 0.769, 0.775]

x = np.arange(len(models))

plt.figure(figsize=(13, 6))

# 折线图
plt.plot(x, acc7, marker='o', linewidth=1.8, label='Acc-7')
plt.plot(x, acc5, marker='o', linewidth=1.8, label='Acc-5')
plt.plot(x, acc2, marker='o', linewidth=1.8, label='Acc-2')
plt.plot(x, f1,   marker='o', linewidth=1.8, label='F1')
plt.plot(x, corr, marker='o', linewidth=1.8, label='Corr')

# ===== 强调 MOE-AHL =====
moe_idx = len(models) - 1

for y in [acc7, acc5, acc2, f1, corr]:
    plt.scatter(moe_idx, y[moe_idx], s=140)
    plt.annotate(
        "MOE-AHL",
        xy=(moe_idx, y[moe_idx]),
        xytext=(moe_idx - 2, y[moe_idx] + 0.5),
        arrowprops=dict(arrowstyle="->", linewidth=1)
    )

plt.xticks(x, models, rotation=30)
plt.ylabel("Score")
plt.title("MOSEI Dataset Comparison Results")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

# ===== 保存图片 =====
plt.savefig("mosei.png", dpi=300, bbox_inches='tight')

plt.show()