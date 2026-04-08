import json
import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from main import optimal_modes

# ===== 1. 读取 JSON 文件 =====
json_path = optimal_modes()  
outpng = Path(json_path).with_suffix(".png")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ===== 2. 为不同 mode_id 设置颜色 =====
mode_colors = {
    1: "tab:grey", # 1 是基准方案(固定路线模式)，颜色较淡
    2: "tab:blue", # 2 是方案二deviated，颜色较亮
    3: "tab:orange", # 3 是方案三DRT，颜色较亮
    4: "tab:red", # 4 是方案四hub-and-spoke，颜色较亮
}

mode_labels = {
    1: "Mode 1",
    2: "Mode 2",
    3: "Mode 3",
    4: "Mode 4",
}

# 为避免同一场景的四个点完全重叠，给不同 mode 一个很小的偏移
offsets = {
    1: (-0.03, -0.03, -0.8),
    2: (-0.03,  0.03, -0.3),
    3: ( 0.03, -0.03,  0.3),
    4: ( 0.03,  0.03,  0.8),
}

# ===== 3. 创建 3D 图 =====

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# 为了避免 legend 重复，只给每个 mode_id 添加一次标签
added_label = set()

for row in data:
    x = row["ht"]
    y = row["hs"]
    z = row["lambda"]
    mode_id = row["mode_id"]

    dx, dy, dz = offsets.get(mode_id, (0, 0, 0))

    label = mode_labels[mode_id] if mode_id not in added_label else None
    if label is not None:
        added_label.add(mode_id)

    ax.scatter(
        x + dx,
        y + dy,
        z + dz,
        color=mode_colors.get(mode_id, "black"),
        s=60,
        alpha=0.85,
        label=label
    )

# ===== 4. 坐标轴和标题 =====
ax.set_xlabel("ht")
ax.set_ylabel("hs")
ax.set_zlabel("lambda")
ax.set_title("3D Scatter Plot of Scenarios by Mode ID")

# 设置刻度，更清楚
ax.set_xticks([0.0, 0.5, 1.0])
ax.set_yticks([0.0, 0.5, 1.0])
ax.set_zticks([20, 40, 60])

#
# ===== 5. 图例 =====
ax.legend(title="mode_id")

plt.tight_layout()
plt.show()
plt.savefig(outpng, dpi=600, bbox_inches="tight")