from rot_utils import sixd_to_euler
import torch


def write_bvh(template_lines, rotations, out_path):
    out = []
    i = 0
    while i < len(template_lines):
        line = template_lines[i]
        out.append(line)
        if line.startswith("Frame Time"):
            i += 1
            break
        i += 1

    F, J, _ = rotations.shape

    for f in range(F):
        out.append("0.0 0.0 0.0 ")
        e = sixd_to_euler(rotations[f])

        frame_euler = []

        for j in range(J):
            x, y, z = e[j]
            frame_euler.extend([z.item(), y.item(), x.item()])

        out.append(" ".join(f"{v:.6f}" for v in frame_euler) + "\n")

    with open(out_path, "w") as f:
        f.writelines(out)