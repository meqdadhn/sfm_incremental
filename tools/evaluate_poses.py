#!/usr/bin/env python3
"""Compares estimated poses against ground truth after a similarity alignment of the camera centers.

Both files: one line per image, "name qw qx qy qz tx ty tz [...]", with x_cam = R X + t.

    python3 tools/evaluate_poses.py <gt_poses.txt> <out/poses.txt>
"""
import sys

import numpy as np


def load(path):
    poses = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        v = line.split()
        w, x, y, z = map(float, v[1:5])
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        t = np.array(list(map(float, v[5:8])))
        poses[v[0]] = (R, t, -R.T @ t)
    return poses


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    E[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ E @ Vt
    s = np.trace(np.diag(sig) @ E) / (S ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def main():
    gt, est = load(sys.argv[1]), load(sys.argv[2])
    names = sorted(set(gt) & set(est))
    print(f"{len(names)} / {len(gt)} images registered")
    s, R, t = umeyama(np.array([est[n][2] for n in names]), np.array([gt[n][2] for n in names]))
    centers = np.array([gt[n][2] for n in names])
    extent = np.linalg.norm(centers - centers.mean(0), axis=1).max()
    rot_err, pos_err = [], []
    for n in names:
        R_e, _, C_e = est[n]
        R_g, _, C_g = gt[n]
        dR = (R_e @ R.T) @ R_g.T
        rot_err.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        pos_err.append(np.linalg.norm(s * R @ C_e + t - C_g))
    rot_err, pos_err = np.array(rot_err), np.array(pos_err)
    print(f"rotation error [deg]: mean {rot_err.mean():.4f}  max {rot_err.max():.4f}")
    print(f"position error [gt units]: mean {pos_err.mean():.4f}  max {pos_err.max():.4f}  (scene radius {extent:.2f})")


if __name__ == "__main__":
    main()
