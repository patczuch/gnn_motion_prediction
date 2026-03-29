import torch

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