import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据
data = torch.load("./outputs/save/predictor_outputs.pt")
preds = data["preds"].numpy()
labels = data["labels"].numpy()

# 计算误差指标
mse = mean_squared_error(labels, preds)
r2 = r2_score(labels, preds)

# 创建图像和坐标轴
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(labels, preds, alpha=0.5, s=10)
ideal_line, = ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'r--', lw=2, label="Ideal (Prediction = Ground Truth)")

# 坐标轴设置
ax.set_xlabel("True Accuracy", fontsize=32)
ax.set_ylabel("Predicted Accuracy", fontsize=32)
ax.set_ylim(70, 100)
ax.tick_params(axis='both', labelsize=28)
# ax.set_xlim(70, 100)
ax.grid(True)

# legend 放在图外顶部
fig.legend(
    handles=[ideal_line],
    loc='upper center',
    ncol=1,
    fontsize=26,
    bbox_to_anchor=(0.5, 0.98)  # 向上移动 legend
)

# 留出 legend 空间
plt.tight_layout()
plt.subplots_adjust(top=0.8)

# 保存图像
plt.savefig("./outputs/accuracy_predictor_scatter_logits.png", dpi=300)
plt.show()
