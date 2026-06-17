import matplotlib.pyplot as plt
import numpy as np

# 模型名称
models = ["HTRN", "TFN", "LMF", "MFN", "MulT", "Self-MM", "TETFN", "MOE-AHL"]

# 指标数据（不含MAE）
acc5 = [43.98, 39.30, 40.53, 39.47, 37.94, 41.53, 41.79, 48.14]
acc3 = [66.71, 65.12, 64.68, 65.73, 64.77, 65.47, 63.24, 66.95]
acc2 = [80.31, 78.38, 77.77, 77.90, 78.56, 80.04, 81.18, 81.40]
f1   = [80.23, 78.62, 77.88, 77.88, 79.66, 80.44, 80.24, 81.41]
corr = [0.628, 0.591, 0.575, 0.582, 0.564, 0.595, 0.576, 0.593]

x = np.arange(len(models))

plt.figure(figsize=(12, 6))

plt.plot(x, acc5, marker='o', linewidth=1.8, label='Acc-5')
plt.plot(x, acc3, marker='o', linewidth=1.8, label='Acc-3')
plt.plot(x, acc2, marker='o', linewidth=1.8, label='Acc-2')
plt.plot(x, f1,   marker='o', linewidth=1.8, label='F1')
plt.plot(x, corr, marker='o', linewidth=1.8, label='Corr')

# 强调 MOE-AHL
moe_idx = len(models) - 1

for y in [acc5, acc3, acc2, f1, corr]:
    plt.scatter(moe_idx, y[moe_idx], s=120)
    plt.annotate(
        "MOE-AHL",
        xy=(moe_idx, y[moe_idx]),
        xytext=(moe_idx - 1.5, y[moe_idx] + 1),
        arrowprops=dict(arrowstyle="->", linewidth=1)
    )

plt.xticks(x, models, rotation=30)
plt.ylabel("Score")
plt.title("CH-SIMS Dataset Comparison Results")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

# ====== 保存图片（关键）======
plt.savefig("sims.png", dpi=300, bbox_inches='tight')

plt.show()