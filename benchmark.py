import os
import csv
import shutil
import torch
import numpy as np
from datetime import datetime
from torch_geometric.data import Data
from motionpredictor import Model
from dataset import BVHMotionDataset, yaw_rotmat, rotate_rot6, rotate_vec
import pymotion.rotations.ortho6d_torch as sixd_torch
import pymotion.rotations.quat_torch as quat_torch
import pymotion.ops.skeleton_torch as skeleton_torch
from train import positions_from_global_rotmats
from mdc_mdcss import torch_mdc_mdcss_metrics
import config


def _migrate_csv_schema(csv_path, headers):
    if not os.path.isfile(csv_path):
        return
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] == headers:
        return

    old_headers = rows[0]
    shutil.copy2(csv_path, csv_path + ".bak")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows[1:]:
            d = dict(zip(old_headers, row))
            writer.writerow([d.get(h, "ALL" if h == "dataset" else "N/A") for h in headers])
    print(f"  migrated {csv_path} to the current schema (backup: {csv_path}.bak)")


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

    checkpoint_path = config.eval_checkpoint_path
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
    ALL = "ALL"

    def new_metrics():
        return {
            "l2p": [], "l2q": [],
            "rot_mae": [], "rot_rmse": [],
            "pos_mae": [], "pos_rmse": [],
            "root_pos_mae": [], "root_pos_rmse": [],
            "npss": [],
            "mdc_pred": [], "mdc_gt": [],
            "mdcss_pred": [], "mdcss_gt": [],
        }

    dataset_names = [os.path.basename(os.path.normpath(p)) for p in dataset_paths]
    metrics = {(c, d): new_metrics() for c in cases for d in dataset_names + [ALL]}

    def dataset_of(filepath):
        return os.path.basename(os.path.dirname(filepath))

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
            root_pos_ctx_in = root_pos_context
            if dataset.facing_normalization:
                theta = dataset.yaw_cache[filepath][start + context - 1]
                R_inv, R_fwd = yaw_rotmat(-theta), yaw_rotmat(theta).to(device)
                context_frames = rotate_rot6(context_frames, R_inv)
                root_pos_ctx_in = rotate_vec(
                    root_pos_context.view(context, 3), R_inv
                ).reshape(context * 3)

            x_input = context_frames.permute(1, 0, 2).reshape(J, -1)
            if dataset.skeleton_geometry:
                x_input = torch.cat([x_input, skeleton["geom"]], dim=1)
            x_input = x_input.to(device)
            edge_index = skeleton["edges"].to(device)
            batch = torch.zeros(J, dtype=torch.long, device=device)
            src_graph = Data(
                x=x_input, edge_index=edge_index, batch=batch,
                root_pos=root_pos_ctx_in.to(device),
            )

            pred_rot, pred_root_pos = model(src_graph)
            pred_seq = pred_rot.view(J, gen_frames, rotation_dim)  # (J, gen_frames, rotation_dim)

            # Predicted root pos delta: (1, gen_frames, 3)
            pred_root_pos_delta = pred_root_pos.view(1, gen_frames, 3)

            if dataset.facing_normalization:
                pred_seq = rotate_rot6(pred_seq, R_fwd)
                pred_root_pos_delta = rotate_vec(pred_root_pos_delta, R_fwd)
            # GT root pos delta: (1, gen_frames, 3)
            gt_root_pos_delta = root_pos_gt_delta.unsqueeze(0).to(device)

            gt_gen_rot = all_gt_rot[context: context + gen_frames].to(device)  # (gen_frames, J, rotation_dim)
            gt_seq = gt_gen_rot.permute(1, 0, 2)  # (J, gen_frames, rotation_dim)

            # Convert global 6D -> rotation matrices
            pred_mat = sixd_torch.to_matrix(pred_seq.reshape(-1, 3, 2)).view(J, gen_frames, 3, 3)
            gt_mat = sixd_torch.to_matrix(gt_seq.reshape(-1, 3, 2)).view(J, gen_frames, 3, 3)

            # Global quats (for converting to local quats for L2Q/rot metrics)
            pred_global_quat = quat_torch.from_matrix(pred_mat.reshape(-1, 3, 3)).view(J, gen_frames, 4)
            gt_global_quat = quat_torch.from_matrix(gt_mat.reshape(-1, 3, 3)).view(J, gen_frames, 4)

            parents_t = torch.tensor(skeleton["parents"], device=device).long()
            offsets_t = torch.tensor(skeleton["offsets"], device=device).to(pred_rot.dtype)
            offsets_b = offsets_t.unsqueeze(0)  # (1, J, 3)

            bone_lengths = offsets_t[1:].norm(dim=-1).unsqueeze(0)  # (1, J-1)

            # Convert global -> local quats for quaternion-based metrics
            pred_local_quat = skeleton_torch.from_global_rotations(
                pred_global_quat.permute(1, 0, 2), parents_t
            ).permute(1, 0, 2)  # (J, gen_frames, 4)
            gt_local_quat = skeleton_torch.from_global_rotations(
                gt_global_quat.permute(1, 0, 2), parents_t
            ).permute(1, 0, 2)  # (J, gen_frames, 4)

            # Compute positions from global rotation matrices
            pred_R_tm = pred_mat.permute(1, 0, 2, 3).unsqueeze(0)  # (1, gen_frames, J, 3, 3)
            gt_R_tm = gt_mat.permute(1, 0, 2, 3).unsqueeze(0)

            pred_global_pos = pred_root_pos_delta  # (1, gen_frames, 3)
            gt_global_pos = gt_root_pos_delta      # (1, gen_frames, 3)

            pred_pos_fk = positions_from_global_rotmats(pred_R_tm, pred_global_pos, offsets_b, parents_t)
            gt_pos_fk = positions_from_global_rotmats(gt_R_tm, gt_global_pos, offsets_b, parents_t)

            # Full sequence for MDC/MDCSS
            ctx_gt_rot = all_gt_rot[:context].to(device)
            ctx_mat = sixd_torch.to_matrix(ctx_gt_rot.reshape(-1, 3, 2)).view(context, J, 3, 3)

            # Full sequence global rotmats
            full_pred_mat = torch.cat([ctx_mat, pred_mat.permute(1, 0, 2, 3)], dim=0)  # (total_frames, J, 3, 3)
            full_gt_mat = sixd_torch.to_matrix(all_gt_rot.to(device).reshape(-1, 3, 2)).view(total_frames, J, 3, 3)

            full_pred_R = full_pred_mat.unsqueeze(0)  # (1, total_frames, J, 3, 3)
            full_gt_R = full_gt_mat.unsqueeze(0)

            ctx_root_delta = root_pos_window_delta[:context].to(device)  # (context, 3)
            full_pred_root = torch.cat([ctx_root_delta, pred_root_pos_delta.squeeze(0)], dim=0).unsqueeze(0)
            full_gt_root = root_pos_window_delta.to(device).unsqueeze(0)

            full_pred_pos = positions_from_global_rotmats(full_pred_R, full_pred_root, offsets_b, parents_t)
            full_gt_pos = positions_from_global_rotmats(full_gt_R, full_gt_root, offsets_b, parents_t)

            ds_name = dataset_of(filepath)

            for case in cases:
                # Every metric is recorded against both its own eval set and the
                # aggregate, so the two views can never drift apart.
                sinks = [metrics[(case, ds_name)], metrics[(case, ALL)]]

                def rec(key, value, _sinks=sinks):
                    for s in _sinks:
                        s[key].append(value)

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
                rec("l2p", l2p)

                # ---- L2Q: mean per-frame per-joint L2 quaternion error ----
                pq = pred_local_quat.permute(1, 0, 2)  # (gen_frames, J, 4)
                gq = gt_local_quat.permute(1, 0, 2)
                dot = (pq * gq).sum(dim=-1, keepdim=True)
                pq_aligned = torch.where(dot < 0, -pq, pq)
                quat_diff = (pq_aligned - gq).norm(dim=-1)  # (gen_frames, J)
                l2q = quat_diff.mean().item()
                rec("l2q", l2q)

                # ---- Joint rotation MAE & RMSE (geodesic difference) ----
                dot_val = (pq_aligned * gq).sum(dim=-1).clamp(-1.0, 1.0)  # (gen_frames, J)
                angle_rad = 2.0 * torch.acos(dot_val.abs())  # geodesic angle in radians
                angle_deg = angle_rad * (180.0 / torch.pi)
                rot_mae = angle_deg.mean().item()
                rot_rmse = angle_deg.pow(2).mean().sqrt().item()
                rec("rot_mae", rot_mae)
                rec("rot_rmse", rot_rmse)

                # ---- Joint position MAE & RMSE ----
                pos_err = (p_pos - g_pos).reshape(-1, 3)  # (gen_frames*J, 3)
                pos_abs = pos_err.abs()
                pos_mae = pos_abs.mean().item()
                pos_rmse = pos_err.pow(2).mean().sqrt().item()
                rec("pos_mae", pos_mae)
                rec("pos_rmse", pos_rmse)

                # ---- Root position MAE & RMSE ----
                root_err = (p_pos[:, :, 0, :] - g_pos[:, :, 0, :]).reshape(-1, 3)
                root_mae = root_err.abs().mean().item()
                root_rmse = root_err.pow(2).mean().sqrt().item()
                rec("root_pos_mae", root_mae)
                rec("root_pos_rmse", root_rmse)

                # ---- NPSS (on generated portion of joint angles as quaternions) ----
                pq_np = pq_aligned.cpu().numpy().reshape(gen_frames, -1)  # (gen_frames, J*4)
                gq_aligned_np = gq.cpu().numpy().reshape(gen_frames, -1)
                if gen_frames >= 4:
                    npss_val = npss(pq_np, gq_aligned_np)
                    rec("npss", npss_val)

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
                        rec("mdc_pred", mdc_pred.mean().item())
                        rec("mdc_gt", mdc_gt.mean().item())
                        rec("mdcss_pred", mdcss_pred.mean().item())
                        rec("mdcss_gt", mdcss_gt.mean().item())
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
        log_print(f"Evaluated {len(metrics[('with_root', ALL)]['l2p'])} samples.")
        log_print(f"Facing normalization: {dataset.facing_normalization}\n")

        for case, ds_name in [(c, d) for c in cases for d in dataset_names + [ALL]]:
            m = metrics[(case, ds_name)]
            n = len(m["l2p"])
            if n == 0:
                log_print(f"[{case} / {ds_name}] No valid samples.\n")
                continue

            log_print(f"{'=' * 60}")
            log_print(f"  Case: {case}   Dataset: {ds_name}   (n={n})")
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
                mdc_pred_arr = np.array(m["mdc_pred"])
                mdc_gt_arr = np.array(m["mdc_gt"])
                mdcss_pred_arr = np.array(m["mdcss_pred"])
                mdcss_gt_arr = np.array(m["mdcss_gt"])
                mdc_mean_pred = np.mean(mdc_pred_arr)
                mdc_mean_gt = np.mean(mdc_gt_arr)
                mdcss_mean_pred = np.mean(mdcss_pred_arr)
                mdcss_mean_gt = np.mean(mdcss_gt_arr)
                mdc_pct = np.where(mdc_gt_arr != 0, (mdc_pred_arr - mdc_gt_arr) / np.abs(mdc_gt_arr) * 100.0, np.nan)
                mdcss_pct = np.where(mdcss_gt_arr != 0, (mdcss_pred_arr - mdcss_gt_arr) / np.abs(mdcss_gt_arr) * 100.0, np.nan)
                log_print(f"  MDC (pred):                    {mdc_mean_pred:.6f}")
                log_print(f"  MDC (gt):                      {mdc_mean_gt:.6f}")
                log_print(f"  MDC (pred-gt %):               {np.nanmean(mdc_pct):.4f}%")
                log_print(f"  MDCSS (pred):                  {mdcss_mean_pred:.6f}")
                log_print(f"  MDCSS (gt):                    {mdcss_mean_gt:.6f}")
                log_print(f"  MDCSS (pred-gt %):             {np.nanmean(mdcss_pct):.4f}%")
            else:
                log_print(f"  MDC / MDCSS:                   N/A")
                mdc_pred_arr = mdc_gt_arr = mdcss_pred_arr = mdcss_gt_arr = None
                mdc_mean_pred = mdc_mean_gt = mdcss_mean_pred = mdcss_mean_gt = None
                mdc_pct = mdcss_pct = None
            log_print()

            # ---- Append to CSV ----
            csv_path = "./benchmarks/benchmarks.csv"
            csv_exists = os.path.isfile(csv_path)
            csv_headers = [
                "timestamp", "model", "case", "dataset", "n_samples",
                "l2p", "l2q", "npss",
                "rot_mae_deg", "rot_rmse_deg",
                "pos_mae", "pos_rmse",
                "root_pos_mae", "root_pos_rmse",
                "mdc_pred", "mdc_gt", "mdc_pct_diff",
                "mdcss_pred", "mdcss_gt", "mdcss_pct_diff",
            ]
            _migrate_csv_schema(csv_path, csv_headers)
            with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
                if not csv_exists:
                    writer.writeheader()
                row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model": model_name,
                    "case": case,
                    "dataset": ds_name,
                    "n_samples": n,
                    "l2p": f"{np.mean(m['l2p']):.6f}",
                    "l2q": f"{np.mean(m['l2q']):.6f}",
                    "npss": f"{np.mean(m['npss']):.6f}" if m["npss"] else "N/A",
                    "rot_mae_deg": f"{np.mean(m['rot_mae']):.6f}",
                    "rot_rmse_deg": f"{np.mean(m['rot_rmse']):.6f}",
                    "pos_mae": f"{np.mean(m['pos_mae']):.6f}",
                    "pos_rmse": f"{np.mean(m['pos_rmse']):.6f}",
                    "root_pos_mae": f"{np.mean(m['root_pos_mae']):.6f}",
                    "root_pos_rmse": f"{np.mean(m['root_pos_rmse']):.6f}",
                    "mdc_pred": f"{mdc_mean_pred:.6f}" if mdc_mean_pred is not None else "N/A",
                    "mdc_gt": f"{mdc_mean_gt:.6f}" if mdc_mean_gt is not None else "N/A",
                    "mdc_pct_diff": f"{np.nanmean(mdc_pct):.4f}" if mdc_pct is not None else "N/A",
                    "mdcss_pred": f"{mdcss_mean_pred:.6f}" if mdcss_mean_pred is not None else "N/A",
                    "mdcss_gt": f"{mdcss_mean_gt:.6f}" if mdcss_mean_gt is not None else "N/A",
                    "mdcss_pct_diff": f"{np.nanmean(mdcss_pct):.4f}" if mdcss_pct is not None else "N/A",
                }
                writer.writerow(row)


if __name__ == "__main__":
    run_benchmark()
