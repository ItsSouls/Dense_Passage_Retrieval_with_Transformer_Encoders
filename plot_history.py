import json
import matplotlib.pyplot as plt

runs = [
    ("A: lr 1e-4, dropout 0.1", "outputs/train/A_base"),
    ("B: lr 1e-4, dropout 0.2", "outputs/train/B_dropout02"),
    ("C: lr 3e-4, dropout 0.1", "outputs/train/C_lr3e-4"),
    ("D: lr 3e-4, dropout 0.2", "outputs/train/D_lr3e-4_d02"),
]

for name, folder in runs:
    with open(f"{folder}/history.json") as f:
        history = json.load(f)
    epochs = [d["epoch"] for d in history]
    val_loss = [d["val_loss"] for d in history]
    plt.plot(epochs, val_loss, label=name)

plt.axhline(3.436, color="gray", linestyle="--", label="random baseline")
plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.title("DPR on FiQA (mean pooling)")
plt.legend()
plt.xticks(range(1, 21))
plt.savefig("val_loss_curves.png", dpi=200)