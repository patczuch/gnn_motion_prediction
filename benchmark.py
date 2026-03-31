import os
import torch
import numpy as np
from torch_geometric.data import Data
from motionpredictor import Model
from dataset import BVHMotionDataset
import pymotion.rotations.ortho6d_torch as sixd_torch
import pymotion.rotations.quat_torch as quat_torch
import pymotion.ops.skeleton_torch as skeleton_torch
from mdc_mdcss import torch_mdc_mdcss_metrics
import config


def npss(pred_signal: np.ndarray, gt_signal: np.ndarray) -> float:
    pred_ps = np.abs(np.fft.rfft(pred_signal, axis=0)) ** 2  # (F//2+1, D)
    gt_ps = np.abs(np.fft.rfft(gt_signal, axis=0)) ** 2

    pred_total = pred_ps.sum(axis=0, keepdims=True)
    pred_total = np.where(pred_total < 1e-10, 1.0, pred_total)
    gt_total = gt_ps.sum(axis=0, keepdims=True)
    gt_total = np.where(gt_total < 1e-10, 1.0, gt_total)

    pred_norm = pred_ps / pred_total
    gt_norm = gt_ps / gt_total

    return ((pred_norm - gt_norm) ** 2).mean()


def run_benchmark():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    checkpoint_path = "./checkpoints/model_20260331-080031-best.pth"
    model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    dataset_paths = config.eval_data_paths
    context = config.context_length
    gen_frames = config.gen_frames
    rotation_dim = config.rotation_dim

    dataset = BVHMotionDataset(dataset_paths, context=context, step=config.benchmark_step_size)

    model = Model().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test samples: {len(dataset)}")
    print(f"Context: {context}, Gen frames: {gen_frames}")

    total_frames = context + gen_frames

    cases = ["with_root", "no_root"]
    metrics = {c: {
        "l2p": [], "l2q": [],
        "rot_mae": [], "rot_rmse": [],
        "pos_mae": [], "pos_rmse": [],
        "root_pos_mae": [], "root_pos_rmse": [],
        "npss": [],
        "mdc_pred": [], "mdc_gt": [],
        "mdcss_pred": [], "mdcss_gt": [],
    } for c in cases}

    skipped = 0

    with torch.no_grad():
        for idx in range(len(dataset)):
            filepath, start = dataset.samples[idx]
            feats = dataset.cache[filepath]  # (F, J, C)
            root_positions = dataset.root_pos_cache[filepath]  # (F, 3)
            skeleton = dataset.get_skeleton(filepath)
            J = feats.shape[1]

            if start + total_frames > feats.shape[0]:
                skipped += 1
                continue

            all_gt = feats[start: start + total_frames]  # (total_frames, J, C)
            all_gt_rot = all_gt[:, :, :rotation_dim]      # (total_frames, J, rotation_dim)

            # --- Prepare root positions (delta from first context frame) ---
            root_pos_window = root_positions[start:start + total_frames]  # (total_frames, 3)
            root_pos_origin = root_pos_window[0:1]  # (1, 3)
            root_pos_window_delta = root_pos_window - root_pos_origin

            root_pos_context = root_pos_window_delta[:context].reshape(context * 3)  # (context*3,)
            root_pos_gt_delta = root_pos_window_delta[context:context + gen_frames]  # (gen_frames, 3)

            # --- Run model ---
            context_frames = all_gt_rot[:context]  # (context, J, rotation_dim)
            x_input = context_frames.permute(1, 0, 2).reshape(J, -1).to(device)
            edge_index = skeleton["edges"].to(device)
            batch = torch.zeros(J, dtype=torch.long, device=device)
            src_graph = Data(
                x=x_input, edge_index=edge_index, batch=batch,
                root_pos=root_pos_context.to(device),
            )

            pred_rot, pred_root_pos = model(src_graph)
            pred_seq = pred_rot.view(J, gen_frames, rotation_dim)  # (J, gen_frames, rotation_dim)

            # Predicted root pos delta: (1, gen_frames, 3)
            pred_root_pos_delta = pred_root_pos.view(1, gen_frames, 3)
            # GT root pos delta: (1, gen_frames, 3)
            gt_root_pos_delta = root_pos_gt_delta.unsqueeze(0).to(device)

            gt_gen_rot = all_gt_rot[context: context + gen_frames].to(device)  # (gen_frames, J, rotation_dim)
            gt_seq = gt_gen_rot.permute(1, 0, 2)  # (J, gen_frames, rotation_dim)

            pred_mat = sixd_torch.to_matrix(pred_seq.reshape(-1, 3, 2))
            gt_mat = sixd_torch.to_matrix(gt_seq.reshape(-1, 3, 2))

            pred_quat = quat_torch.from_matrix(pred_mat).view(J, gen_frames, 4)
            gt_quat = quat_torch.from_matrix(gt_mat).view(J, gen_frames, 4)

            parents_t = torch.tensor(skeleton["parents"], device=device).long()
            offsets_t = torch.tensor(skeleton["offsets"], device=device).to(pred_rot.dtype)

            bone_lengths = offsets_t.norm(dim=-1).unsqueeze(0)  # (1, J)

            pred_quat_fk = pred_quat.unsqueeze(0).permute(0, 2, 1, 3)
            gt_quat_fk = gt_quat.unsqueeze(0).permute(0, 2, 1, 3)

            offsets_fk = offsets_t.unsqueeze(0).unsqueeze(0).expand(1, gen_frames, J, 3)

            # Use predicted/gt root pos deltas converted back to absolute for FK
            pred_global_pos = pred_root_pos_delta  # (1, gen_frames, 3)
            gt_global_pos = gt_root_pos_delta      # (1, gen_frames, 3)

            pred_pos_fk, _ = skeleton_torch.fk(pred_quat_fk, pred_global_pos, offsets_fk, parents_t)
            gt_pos_fk, _ = skeleton_torch.fk(gt_quat_fk, gt_global_pos, offsets_fk, parents_t)

            # Full sequence for MDC/MDCSS
            ctx_gt_rot = all_gt_rot[:context].to(device)
            ctx_mat = sixd_torch.to_matrix(ctx_gt_rot.reshape(-1, 3, 2))
            ctx_quat = quat_torch.from_matrix(ctx_mat).view(context, J, 4)

            full_pred_quat = torch.cat([ctx_quat, pred_quat.permute(1, 0, 2)], dim=0)
            full_gt_quat_raw = sixd_torch.to_matrix(all_gt_rot.to(device).reshape(-1, 3, 2))
            full_gt_quat = quat_torch.from_matrix(full_gt_quat_raw).view(total_frames, J, 4)

            full_pred_quat_fk = full_pred_quat.unsqueeze(0)
            full_gt_quat_fk = full_gt_quat.unsqueeze(0)

            offsets_full = offsets_t.unsqueeze(0).unsqueeze(0).expand(1, total_frames, J, 3)

            # Full root pos: context (gt delta) + generated (pred/gt delta)
            ctx_root_delta = root_pos_window_delta[:context].to(device)  # (context, 3)
            full_pred_root = torch.cat([ctx_root_delta, pred_root_pos_delta.squeeze(0)], dim=0).unsqueeze(0)  # (1, total_frames, 3)
            full_gt_root = root_pos_window_delta.to(device).unsqueeze(0)  # (1, total_frames, 3)

            full_pred_pos, _ = skeleton_torch.fk(full_pred_quat_fk, full_pred_root, offsets_full, parents_t)
            full_gt_pos, _ = skeleton_torch.fk(full_gt_quat_fk, full_gt_root, offsets_full, parents_t)

            for case in cases:
                if case == "no_root":
                    p_pos = pred_pos_fk.clone()
                    g_pos = gt_pos_fk.clone()
                    p_pos[:, :, 0, :] = 0.0
                    g_pos[:, :, 0, :] = 0.0

                    fp_pos = full_pred_pos.clone()
                    fg_pos = full_gt_pos.clone()
                    fp_pos[:, :, 0, :] = 0.0
                    fg_pos[:, :, 0, :] = 0.0
                else:
                    p_pos = pred_pos_fk
                    g_pos = gt_pos_fk
                    fp_pos = full_pred_pos
                    fg_pos = full_gt_pos

                # ---- L2P: mean per-frame per-joint L2 position error ----
                pos_diff = (p_pos - g_pos).norm(dim=-1)  # (1, gen_frames, J)
                l2p = pos_diff.mean().item()
                metrics[case]["l2p"].append(l2p)

                # ---- L2Q: mean per-frame per-joint L2 quaternion error ----
                pq = pred_quat.permute(1, 0, 2)  # (gen_frames, J, 4)
                gq = gt_quat.permute(1, 0, 2)
                dot = (pq * gq).sum(dim=-1, keepdim=True)
                pq_aligned = torch.where(dot < 0, -pq, pq)
                quat_diff = (pq_aligned - gq).norm(dim=-1)  # (gen_frames, J)
                l2q = quat_diff.mean().item()
                metrics[case]["l2q"].append(l2q)

                # ---- Joint rotation MAE & RMSE (geodesic difference) ----
                dot_val = (pq_aligned * gq).sum(dim=-1).clamp(-1.0, 1.0)  # (gen_frames, J)
                angle_rad = 2.0 * torch.acos(dot_val.abs())  # geodesic angle in radians
                angle_deg = angle_rad * (180.0 / torch.pi)
                rot_mae = angle_deg.mean().item()
                rot_rmse = angle_deg.pow(2).mean().sqrt().item()
                metrics[case]["rot_mae"].append(rot_mae)
                metrics[case]["rot_rmse"].append(rot_rmse)

                # ---- Joint position MAE & RMSE ----
                pos_err = (p_pos - g_pos).reshape(-1, 3)  # (gen_frames*J, 3)
                pos_abs = pos_err.abs()
                pos_mae = pos_abs.mean().item()
                pos_rmse = pos_err.pow(2).mean().sqrt().item()
                metrics[case]["pos_mae"].append(pos_mae)
                metrics[case]["pos_rmse"].append(pos_rmse)

                # ---- Root position MAE & RMSE ----
                root_err = (p_pos[:, :, 0, :] - g_pos[:, :, 0, :]).reshape(-1, 3)
                root_mae = root_err.abs().mean().item()
                root_rmse = root_err.pow(2).mean().sqrt().item()
                metrics[case]["root_pos_mae"].append(root_mae)
                metrics[case]["root_pos_rmse"].append(root_rmse)

                # ---- NPSS (on generated portion of joint angles as quaternions) ----
                pq_np = pq_aligned.cpu().numpy().reshape(gen_frames, -1)  # (gen_frames, J*4)
                gq_aligned_np = gq.cpu().numpy().reshape(gen_frames, -1)
                if gen_frames >= 4:
                    npss_val = npss(pq_np, gq_aligned_np)
                    metrics[case]["npss"].append(npss_val)

                # ---- MDC & MDCSS ----
                fps = int(round(1.0 / skeleton["frame_time"])) if skeleton["frame_time"] > 0 else 30
                clip_frames = min(gen_frames, total_frames - 2)

                if total_frames >= 4 and clip_frames >= 2:
                    try:
                        mdc_pred, mdcss_pred = torch_mdc_mdcss_metrics(
                            fp_pos, bone_lengths, fps, clip_frames
                        )
                        mdc_gt, mdcss_gt = torch_mdc_mdcss_metrics(
                            fg_pos, bone_lengths, fps, clip_frames
                        )
                        metrics[case]["mdc_pred"].append(mdc_pred.mean().item())
                        metrics[case]["mdc_gt"].append(mdc_gt.mean().item())
                        metrics[case]["mdcss_pred"].append(mdcss_pred.mean().item())
                        metrics[case]["mdcss_gt"].append(mdcss_gt.mean().item())
                    except Exception as e:
                        print(f"  MDC/MDCSS error for sample {idx}: {e}")

            if (idx + 1) % 500 == 0 or idx == len(dataset) - 1:
                print(f"  Processed {idx + 1}/{len(dataset)} samples...")

    # ---- Print results ----
    os.makedirs("./benchmarks", exist_ok=True)
    out_file_path = f"./benchmarks/benchmark_{model_name}.txt"

    with open(out_file_path, "w", encoding="utf-8") as out_file:
        def log_print(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, file=out_file, **kwargs)

        log_print(f"\nSkipped {skipped} samples (not enough frames).")
        log_print(f"Evaluated {len(metrics['with_root']['l2p'])} samples.\n")

        for case in cases:
            m = metrics[case]
            n = len(m["l2p"])
            if n == 0:
                log_print(f"[{case}] No valid samples.")
                continue

            log_print(f"{'=' * 60}")
            log_print(f"  Case: {case}")
            log_print(f"{'=' * 60}")
            log_print(f"  L2P (mean joint pos error):    {np.mean(m['l2p']):.6f}")
            log_print(f"  L2Q (mean joint quat error):   {np.mean(m['l2q']):.6f}")
            if m["npss"]:
                log_print(f"  NPSS:                          {np.mean(m['npss']):.6f}")
            else:
                log_print(f"  NPSS:                          N/A (too few frames)")
            log_print(f"  Joint Rotation MAE (deg):      {np.mean(m['rot_mae']):.6f}")
            log_print(f"  Joint Rotation RMSE (deg):     {np.mean(m['rot_rmse']):.6f}")
            log_print(f"  Joint Position MAE:            {np.mean(m['pos_mae']):.6f}")
            log_print(f"  Joint Position RMSE:           {np.mean(m['pos_rmse']):.6f}")
            log_print(f"  Root Position MAE:             {np.mean(m['root_pos_mae']):.6f}")
            log_print(f"  Root Position RMSE:            {np.mean(m['root_pos_rmse']):.6f}")
            if m["mdc_pred"]:
                mdc_diff = np.abs(np.array(m["mdc_pred"]) - np.array(m["mdc_gt"]))
                mdcss_diff = np.abs(np.array(m["mdcss_pred"]) - np.array(m["mdcss_gt"]))
                log_print(f"  MDC (pred):                    {np.mean(m['mdc_pred']):.6f}")
                log_print(f"  MDC (gt):                      {np.mean(m['mdc_gt']):.6f}")
                log_print(f"  MDC (pred-gt):                 {np.mean(mdc_diff):.6f}")
                log_print(f"  MDCSS (pred):                  {np.mean(m['mdcss_pred']):.6f}")
                log_print(f"  MDCSS (gt):                    {np.mean(m['mdcss_gt']):.6f}")
                log_print(f"  MDCSS (pred-gt):               {np.mean(mdcss_diff):.6f}")
            else:
                log_print(f"  MDC / MDCSS:                   N/A")
            log_print()


if __name__ == "__main__":
    run_benchmark()
