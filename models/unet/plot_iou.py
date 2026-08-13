import json
import matplotlib.pyplot as plt

with open("history_v2.json", "r") as f:
    history = json.load(f)

epochs   = [h["epoch"]      for h in history]
val_iou  = [h["iou"]        for h in history]
val_loss = [h["val_loss"]   for h in history]

plt.figure(figsize=(10, 5))
plt.plot(epochs, val_iou, color="#1D9E75", linewidth=2, marker="o", markersize=4, label="Validation IoU")
plt.axhline(y=0.6085, color="#1D9E75", linestyle="--", linewidth=1, alpha=0.5, label="Best IoU 0.6085")
plt.axhline(y=0.5601, color="#E24B4A", linestyle="--", linewidth=1, alpha=0.5, label="U-Net baseline 0.5601")
plt.axvline(x=10,     color="#BA7517", linestyle=":",  linewidth=1.5, label="Backbone unfrozen (epoch 10)")

plt.title("SegFormer-B2 v2 — Validation IoU per Epoch", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("segformer_iou_graph.png", dpi=150, bbox_inches="tight")
print("Saved segformer_iou_graph.png")
