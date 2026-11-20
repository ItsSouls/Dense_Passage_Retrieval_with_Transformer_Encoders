import json
import matplotlib.pyplot as plt

with open("outputs/train/C_recall/history.json") as f:
    history = json.load(f)

epochs = [d["epoch"] for d in history]
val_loss = [d["val_loss"] for d in history]
recall = [d["recall@10"] for d in history]

fig, ax1 = plt.subplots(figsize=(8, 4.5))

ax1.plot(epochs, val_loss, "o-", color="tab:blue", label="val_loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Validation loss", color="tab:blue")
ax1.set_xticks(epochs)

ax2 = ax1.twinx()
ax2.plot(epochs, recall, "s-", color="tab:orange", label="Recall@10")
ax2.set_ylabel("Recall@10 (validation)", color="tab:orange")

best_loss = epochs[val_loss.index(min(val_loss))]
best_recall = epochs[recall.index(max(recall))]
ax1.axvline(best_loss, color="tab:blue", linestyle="--", alpha=0.5)
ax1.axvline(best_recall, color="tab:orange", linestyle="--", alpha=0.5)

plt.title(f"Run C: best val_loss at epoch {best_loss}, best Recall@10 at epoch {best_recall}")
fig.tight_layout()
plt.savefig("recall_vs_loss.png", dpi=200)