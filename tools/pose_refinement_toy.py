"""
CPU sanity check for camera-pose refinement (no NeRF training, no dataset needed).

A small analytic scene (a cluster of coloured Gaussian blobs, about the size of the
real payload in normalised units) is rendered from a clean orbit. The poses are then
corrupted like data/generate_postprocessed_dataset.py does (correlated radial noise on
the camera distance, correlated pointing noise around the local X/Y axes), and ONLY the
pose table (model/residuals_embedder.py) is optimised against the clean images, with
the same ray deformation, regulariser and optimizer settings as training/trainer.py.

The scene is known exactly, so this is the best case for pose refinement: if rotation
does not converge here, it will not converge in the full joint optimisation either.

Usage:
    python -m tools.pose_refinement_toy                       # compares a few settings
    python -m tools.pose_refinement_toy --only before rot10 norotreg10   # a subset
    python -m tools.pose_refinement_toy --res_lr 2e-3 --res_lr_rot 2e-2  # one custom run
    python -m tools.pose_refinement_toy --rotation_only        # pointing noise only
"""

import argparse
import math

import torch

from model.residuals_embedder import ResEmbedder
from training.pose_utils import corrected_poses, pose_errors


def make_scene(generator):
    centres = (torch.rand(24, 3, generator=generator) - 0.5) * 0.35
    scales = 0.02 + 0.04 * torch.rand(24, 1, generator=generator)
    colours = torch.rand(24, 3, generator=generator)
    return centres, scales, colours


def render(rays_o, rays_d, scene, near, far, n_samples=48):
    centres, scales, colours = scene
    t = torch.linspace(0.0, 1.0, n_samples)
    t = near[:, None] + (far - near)[:, None] * t[None]                  # [R, S]
    pts = rays_o[:, None] + rays_d[:, None] * t[..., None]               # [R, S, 3]
    d2 = (pts[:, :, None] - centres[None, None]).square().sum(-1)        # [R, S, G]
    w = torch.exp(-0.5 * d2 / scales.T[None].square())                   # [R, S, G]
    sigma = 60.0 * w.sum(-1)
    rgb = (w[..., None] * colours[None, None]).sum(-2) / (w.sum(-1, keepdim=True) + 1e-6)
    delta = (far - near)[:, None] / n_samples
    alpha = 1.0 - torch.exp(-sigma * delta)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * trans
    return (weights[..., None] * rgb).sum(1)


def look_at(centre):
    forward = -centre / centre.norm()
    up = torch.tensor([0.0, 0.0, 1.0])
    right = torch.linalg.cross(forward, up); right = right / right.norm()
    true_up = torch.linalg.cross(right, forward)
    pose = torch.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, true_up, -forward, centre
    return pose


def rodrigues(v):
    theta = v.norm()
    if theta < 1e-12:
        return torch.eye(3)
    k = v / theta
    K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return torch.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K


def gauss_markov(n, sigma, rho, generator):
    x = torch.zeros(n)
    x[0] = sigma * torch.randn(1, generator=generator)
    for k in range(1, n):
        x[k] = rho * x[k - 1] + sigma * math.sqrt(1 - rho ** 2) * torch.randn(1, generator=generator)
    return x


def build_problem(n_cams, res, fov_deg, radial_sigma, pointing_sigma_deg, seed):
    g = torch.Generator().manual_seed(seed)
    scene = make_scene(g)

    gt = []
    for k in range(n_cams):
        az = 2 * math.pi * k / n_cams
        dist = 0.78 + 0.22 * math.sin(2 * az)            # 0.56 .. 1.0 like the real orbit
        el = 0.3 * math.sin(az)
        gt.append(look_at(torch.tensor([dist * math.cos(el) * math.cos(az),
                                        dist * math.cos(el) * math.sin(az),
                                        dist * math.sin(el)])))
    gt = torch.stack(gt)

    noisy = gt.clone()
    radial = gauss_markov(n_cams, radial_sigma, 0.9, g)
    tx = gauss_markov(n_cams, math.radians(pointing_sigma_deg), 0.9, g)
    ty = gauss_markov(n_cams, math.radians(pointing_sigma_deg), 0.9, g)
    for k in range(n_cams):
        noisy[k, :3, 3] = gt[k, :3, 3] * (1 + radial[k])
        noisy[k, :3, :3] = gt[k, :3, :3] @ rodrigues(torch.tensor([tx[k], ty[k], 0.0]))

    f = res / (2 * math.tan(math.radians(fov_deg) / 2))
    v, u = torch.meshgrid(torch.arange(res) + 0.5, torch.arange(res) + 0.5, indexing="ij")
    cam_dirs = torch.stack([(u - res / 2) / f, -(v - res / 2) / f, -torch.ones_like(u)], -1).reshape(-1, 3)
    cam_dirs = cam_dirs / cam_dirs.norm(dim=-1, keepdim=True)

    def rays(poses):
        d = torch.einsum("nij,pj->npi", poses[:, :3, :3], cam_dirs)
        o = poses[:, None, :3, 3].expand_as(d)
        return o.reshape(-1, 3), d.reshape(-1, 3)

    with torch.no_grad():
        o, d = rays(gt)
        dist = o.norm(dim=-1)
        target = render(o, d, scene, dist - 0.4, dist + 0.4)
    o_in, d_in = rays(noisy)
    idx = torch.arange(n_cams).repeat_interleave(res * res)
    return scene, gt, noisy, o_in, d_in, idx, target


def run(problem, res_lr, res_lr_rot, steps, batch, reg_w, rot_reg_w, weight_decay,
        lin_range, ang_range_deg, seed):
    """Optimise only the pose table, with the optimizer groups and priors of training/trainer.py."""
    scene, gt, noisy, o_in, d_in, idx, target = problem
    torch.manual_seed(seed)
    res_model = ResEmbedder(len(gt), 6, lin_range=lin_range, ang_range=math.radians(ang_range_deg), device="cpu")
    opt = torch.optim.AdamW([
        {"params": [res_model.trans_embeddings.weight], "lr": res_lr, "weight_decay": weight_decay},
        {"params": [res_model.rot_embeddings.weight], "lr": res_lr_rot, "weight_decay": weight_decay},
    ])
    history = []
    for step in range(steps + 1):
        if step % (steps // 4) == 0:
            with torch.no_grad():
                history.append(pose_errors(corrected_poses(res_model, noisy), gt))
        if step == steps:
            break
        sel = torch.randint(0, len(idx), (batch,))
        dt, dR = res_model(idx[sel])
        o = o_in[sel] + dt
        d = torch.einsum("bij,bj->bi", dR, d_in[sel])
        dist = o_in[sel].norm(dim=-1)
        loss = (render(o, d, scene, dist - 0.4, dist + 0.4) - target[sel]).square().mean()
        loss = loss + reg_w * res_model.trans_embeddings.weight.square().mean()
        loss = loss + rot_reg_w * res_model.rot_embeddings.weight.square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return history


# name, res_lr, res_lr_rot, reg, rot_reg, weight_decay, lin_range, ang_range_deg
SETTINGS = {
    # The old clamp(+-5) * range is the same as clamp(+-1) * (5 * range) with the learning
    # rate, prior weight and weight decay rescaled accordingly (lr / 5, reg * 25, decay * 5).
    "before": ("before this change (clamp +-5, shared lr, AdamW decay 0.01)", 4e-4, 4e-4, 2.5e-2, 2.5e-2, 0.05, 0.5, 2.5),
    "shared": ("clamp +-1, ang_range 1 deg, shared lr 2e-3", 2e-3, 2e-3, 1e-3, 1e-3, 0.0, 0.1, 1.0),
    "rot10": ("clamp +-1, ang_range 1 deg, res_lr_rot 2e-2", 2e-3, 2e-2, 1e-3, 1e-3, 0.0, 0.1, 1.0),
    "norotreg": ("... + no rotation prior, shared lr 2e-3", 2e-3, 2e-3, 1e-3, 0.0, 0.0, 0.1, 1.0),
    "norotreg10": ("... + no rotation prior, res_lr_rot 2e-2", 2e-3, 2e-2, 1e-3, 0.0, 0.0, 0.1, 1.0),
    "noreg10": ("... + no pose priors at all, res_lr_rot 2e-2", 2e-3, 2e-2, 0.0, 0.0, 0.0, 0.1, 1.0),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--res", type=int, default=48)
    p.add_argument("--fov_deg", type=float, default=40.0)
    p.add_argument("--res_lr", type=float, default=None)
    p.add_argument("--res_lr_rot", type=float, default=None)
    p.add_argument("--rotation_only", action="store_true")
    p.add_argument("--only", nargs="*", default=None, help="subset of SETTINGS keys to run")
    args = p.parse_args()

    radial = 0.0 if args.rotation_only else 0.03
    problem = build_problem(30, args.res, args.fov_deg, radial, 0.20, seed=0)
    print("input poses:", pose_errors(problem[2], problem[1]))

    if args.res_lr is not None:
        settings = [("custom", args.res_lr, args.res_lr_rot or args.res_lr, 1e-3, 1e-3, 0.0, 0.1, 1.0)]
    else:
        settings = [SETTINGS[k] for k in (args.only or SETTINGS)]
    for name, *cfg in settings:
        hist = run(problem, *cfg[:2], args.steps, args.batch, *cfg[2:], seed=0)
        print(f"\n{name}")
        for i, h in enumerate(hist):
            print(f"  step {i * args.steps // 4:5d}: t_err {h['translation']:.5f}  r_err {h['rotation_deg']:.4f} deg")


if __name__ == "__main__":
    main()
