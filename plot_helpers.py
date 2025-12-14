import os
import numpy as np
import matplotlib.pyplot as plt

def _plot_skeleton_3d(ax, positions_j3, parents, title):
    """
    positions_j3: (J, 3) numpy
    parents: list[int] length J, parent index or -1
    """
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # joints
    ax.scatter(
        positions_j3[:, 0],
        positions_j3[:, 1],
        positions_j3[:, 2],
        s=10,
        depthshade=True,
    )

    # bones
    for j, p in enumerate(parents):
        if p is None or p < 0:
            continue
        xs = [positions_j3[p, 0], positions_j3[j, 0]]
        ys = [positions_j3[p, 1], positions_j3[j, 1]]
        zs = [positions_j3[p, 2], positions_j3[j, 2]]
        ax.plot(xs, ys, zs, linewidth=2)

    # keep aspect roughly equal
    mins = positions_j3.min(axis=0)
    maxs = positions_j3.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.5 + 1e-9)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

def save_fk_3d_plots(pred_pos_tm, gt_pos_tm, parents, out_path, t_idx=0, elev=20, azim=-60):
    """
    pred_pos_tm, gt_pos_tm: torch.Tensor (1, T, J, 3)
    out_path: path to png
    """
    pred_np = pred_pos_tm.detach().cpu().numpy()
    gt_np = gt_pos_tm.detach().cpu().numpy()

    pred_j3 = pred_np[0, t_idx]  # (J, 3)
    gt_j3 = gt_np[0, t_idx]      # (J, 3)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    _plot_skeleton_3d(ax1, pred_j3, parents, f"Pred FK (t={t_idx})")
    _plot_skeleton_3d(ax2, gt_j3, parents, f"GT FK (t={t_idx})")

    ax1.view_init(elev=elev, azim=azim)
    ax2.view_init(elev=elev, azim=azim)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)