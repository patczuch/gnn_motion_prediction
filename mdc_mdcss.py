import torch


def _safe_norm(v, dim=-1, eps=1e-12):
    return torch.sqrt(v.pow(2).sum(dim=dim) + eps)


def mdc_mdcss_differentiable(g_pos, bone_lengths, fps, clip_frames, eps=1e-8):
    g_vel = g_pos[..., 1:, :, :] - g_pos[..., :-1, :, :]
    v1, v0 = g_vel[..., 1:, :, :], g_vel[..., :-1, :, :]

    vel_cosines = (v1 * v0).sum(dim=-1) / (_safe_norm(v1) * _safe_norm(v0) + eps)
    vel_angles = torch.acos(vel_cosines.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    acc_norms = _safe_norm(v1 - v0)

    scale = bone_lengths.sum(dim=-1)
    while scale.dim() < vel_angles.dim():
        scale = scale.unsqueeze(-1)
    if not torch.is_tensor(fps):
        fps = torch.as_tensor(float(fps), device=g_pos.device, dtype=g_pos.dtype)
    fps_b = fps.reshape((-1,) + (1,) * (vel_angles.dim() - 1)) if fps.dim() else fps

    per_joint_metric = fps_b * vel_angles.pow(2) * acc_norms / scale.clamp(min=eps)
    mdc_metric = per_joint_metric.max(dim=-1).values  # (..., F-2)

    if mdc_metric.shape[-1] < clip_frames or clip_frames < 2:
        return mdc_metric, None

    window_metric = mdc_metric.unfold(-1, clip_frames, 1)
    fft_vals_windowed = torch.fft.fft(window_metric, dim=-1).real
    mdcss_metric = fft_vals_windowed[..., 1:].max(dim=-1).values
    return mdc_metric, mdcss_metric


def mdc_mdcss_loss(pred_pos, gt_pos, bone_lengths, fps, clip_frames):
    mdc_p, mdcss_p = mdc_mdcss_differentiable(pred_pos, bone_lengths, fps, clip_frames)
    mdc_g, mdcss_g = mdc_mdcss_differentiable(gt_pos, bone_lengths, fps, clip_frames)

    loss = (mdc_p - mdc_g).abs().mean()
    if mdcss_p is not None:
        loss = loss + (mdcss_p - mdcss_g).abs().mean()
    return loss


def torch_mdc_mdcss_metrics(
        g_pos: torch.Tensor,
        bone_lengths: torch.Tensor,
        fps: int,
        clip_frames: int
):
    """
    Args:
        g_pos: Global joint positions (after FK), size: (A, F, J, P).
        bone_lengths: Sum of bones lengths in skeleton. Can be substituted with skeleton height (even better)
            or skipped if operating on one skeleton.
        fps: Animation frame rate.
        clip_frames: Number of frames in a clip (assuming A = entire animation sequence, not e.g. 20-frame animation clip).
        A - animation sequence (entire animation)
        F - frames
        J - joints
        P - global joint positions

    Returns: MDC and MDCSS metrics measuring dynamics of animation. MDC measures per-frame-per-joint changes
        in acceleration, while MDCSS focuses more on spectrum of MDC (treated as signal) in a window (animation clip).
    """
    g_vel = g_pos[..., 1:, :, :] - g_pos[..., :-1, :, :]
    vel_cosines = (g_vel[..., 1:, :, :] * g_vel[..., :-1, :, :]).sum(dim=-1)
    norms = g_vel[..., 1:, :, :].norm(dim=-1) * g_vel[..., :-1, :, :].norm(dim=-1)
    norms[norms == 0.] = 1
    vel_cosines = vel_cosines / norms
    vel_angles = torch.acos(vel_cosines.clamp(-1, 1))
    acc_norms = (g_vel[..., 1:, :, :] - g_vel[..., :-1, :, :]).norm(dim=-1)
    per_joint_metric = fps * vel_angles.pow(2) * acc_norms / bone_lengths.sum(dim=-1)
    mdc_metric = per_joint_metric.max(dim=-1).values
    # Gdyby A = klipy animacji tę część można sobie darować
    ##
    sliding_window_frames = clip_frames
    window_metric = mdc_metric.unfold(1, sliding_window_frames, 1)
    ##
    fft_vals_windowed = torch.fft.fft(window_metric, dim=-1).real
    mdcss_metric = fft_vals_windowed[..., 1:].max(dim=-1).values
    return mdc_metric, mdcss_metric