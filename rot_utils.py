import torch


def euler_to_sixd(euler_deg):
    R = euler_to_matrix_zyx(euler_deg)
    a1 = R[..., :, 0]
    a2 = R[..., :, 1]
    return torch.cat([a1, a2], dim=-1)


def euler_to_matrix_zyx(euler_deg):
    x, y, z = torch.deg2rad(euler_deg[..., 0]), torch.deg2rad(euler_deg[..., 1]), torch.deg2rad(euler_deg[..., 2])

    cx, cy, cz = torch.cos(x), torch.cos(y), torch.cos(z)
    sx, sy, sz = torch.sin(x), torch.sin(y), torch.sin(z)

    Rx = torch.stack([
        torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], dim=-1),
        torch.stack([torch.zeros_like(cx), cx, -sx], dim=-1),
        torch.stack([torch.zeros_like(cx), sx,  cx], dim=-1),
    ], dim=-2)

    Ry = torch.stack([
        torch.stack([ cy, torch.zeros_like(cy), sy], dim=-1),
        torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], dim=-1),
        torch.stack([-sy, torch.zeros_like(cy), cy], dim=-1),
    ], dim=-2)

    Rz = torch.stack([
        torch.stack([cz, -sz, torch.zeros_like(cz)], dim=-1),
        torch.stack([sz,  cz, torch.zeros_like(cz)], dim=-1),
        torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)], dim=-1),
    ], dim=-2)

    return Rz @ Ry @ Rx


def sixd_to_euler(v6):
    R = sixd_to_matrix(v6)
    rad = matrix_to_euler_zyx(R)
    return torch.rad2deg(rad)


def matrix_to_euler_zyx(R):
    R = torch.clamp(R, -1., 1.)
    x = torch.zeros(R.shape[0], device=R.device)
    y = torch.zeros(R.shape[0], device=R.device)
    z = torch.zeros(R.shape[0], device=R.device)

    sy = torch.sqrt(R[:,0,0] * R[:,0,0] + R[:,1,0] * R[:,1,0])
    singular = sy < 1e-6

    x[~singular] = torch.atan2(R[~singular,2,1], R[~singular,2,2])
    y[~singular] = torch.atan2(-R[~singular,2,0], sy[~singular])
    z[~singular] = torch.atan2(R[~singular,1,0], R[~singular,0,0])

    x[singular] = torch.atan2(-R[singular,1,2], R[singular,1,1])
    y[singular] = torch.atan2(-R[singular,2,0], sy[singular])
    z[singular] = 0

    return torch.stack((x, y, z), dim=-1)


def sixd_to_matrix(v6):
    a1 = v6[..., 0:3]
    a2 = v6[..., 3:6]

    b1 = torch.nn.functional.normalize(a1, dim=-1)
    a2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack((b1, b2, b3), dim=-1)