#!/usr/bin/env python3
"""Renders a textured box world from known cameras, for end-to-end tests of the pipeline.

Writes <out>/images/*.png, <out>/gt_poses.txt (name qw qx qy qz tx ty tz, x_cam = R X + t)
and <out>/config.yaml ready for sfm_main.

    python3 tools/render_synthetic.py <out_dir> [--views 20]
"""
import argparse
import os

import cv2
import numpy as np

W, H = 960, 720
K = np.array([[800.0, 0, 480.0], [0, 800.0, 360.0], [0, 0, 1]])
DIST = np.array([-0.08, 0.02, 0.0005, -0.0004, 0.0])


def texture(seed, size):
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 120, np.uint8)
    for _ in range(size * size // 1100):
        c = tuple(int(x) for x in rng.integers(0, 255, 3))
        p = tuple(int(x) for x in rng.integers(0, size, 2))
        if rng.random() < 0.5:
            cv2.circle(img, p, int(rng.integers(3, 28)), c, -1, cv2.LINE_AA)
        else:
            q = (p[0] + int(rng.integers(4, 40)), p[1] + int(rng.integers(4, 40)))
            cv2.rectangle(img, p, q, c, -1)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def box(center, size):
    """Five visible faces (no bottom) of an axis-aligned box as (origin, u_axis, v_axis)."""
    c, s = np.array(center, float), np.array(size, float) / 2
    x, y, z = np.eye(3)
    faces = []
    for axis, a, b in ((0, 1, 2), (1, 0, 2)):
        for sign in (-1, 1):
            o = c.copy()
            o[axis] += sign * s[axis]
            o[a] -= s[a]
            o[b] -= s[b]
            faces.append((o, np.eye(3)[a] * 2 * s[a], np.eye(3)[b] * 2 * s[b]))
    o = c + np.array([-s[0], -s[1], s[2]])
    faces.append((o, x * 2 * s[0], y * 2 * s[1]))
    return faces


def scene():
    quads = [(np.array([-12.0, -12.0, 0.0]), np.array([24.0, 0, 0]), np.array([0, 24.0, 0]))]  # ground
    quads.append((np.array([-12.0, 8.0, 0.0]), np.array([24.0, 0, 0]), np.array([0, 0, 8.0])))  # back wall
    quads.append((np.array([-10.0, -12.0, 0.0]), np.array([0, 20.0, 0]), np.array([0, 0, 6.0])))  # left wall
    quads += box((0, 0, 1.0), (2, 2, 2)) + box((3, 2, 0.75), (1.5, 3, 1.5)) + box((-3, 1.5, 1.5), (1.2, 1.2, 3))
    quads += box((2.5, -2.5, 0.5), (1, 1, 1)) + box((-2, -3, 0.9), (2.2, 0.8, 1.8))
    return quads


def bilinear(tex, x, y):
    x0 = np.clip(np.floor(x).astype(int), 0, tex.shape[1] - 2)
    y0 = np.clip(np.floor(y).astype(int), 0, tex.shape[0] - 2)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    t = tex.astype(np.float32)
    top = t[y0, x0] * (1 - fx) + t[y0, x0 + 1] * fx
    bot = t[y0 + 1, x0] * (1 - fx) + t[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bot * fy


def look_at(C, target, up=np.array([0, 0, 1.0])):
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ C


def render(R, t, quads, textures, rays_cam):
    C = -R.T @ t
    d = rays_cam @ R  # world directions (R^T * ray), N x 3
    best = np.full(len(d), np.inf)
    color = np.zeros((len(d), 3), np.float32)
    for (o, u, v), tex in zip(quads, textures):
        n = np.cross(u, v)
        denom = d @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            s = ((o - C) @ n) / denom
        P = C + d * s[:, None]
        rel = P - o
        a = rel @ u / (u @ u)
        b = rel @ v / (v @ v)
        hit = (s > 0.05) & (s < best) & (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)
        if not hit.any():
            continue
        best[hit] = s[hit]
        color[hit] = bilinear(tex, a[hit] * (tex.shape[1] - 1), b[hit] * (tex.shape[0] - 1))
    return color.reshape(H, W, 3).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--views", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    quads = scene()
    # ~150 texels per meter on every face, no tiling (repeated texture would create false matches).
    textures = [texture(i, int(min(4096, 150 * max(np.linalg.norm(u), np.linalg.norm(v))))) for i, (o, u, v) in enumerate(quads)]
    # Distorted pixel -> normalized ray, so the rendered images carry the lens distortion.
    px = np.stack(np.meshgrid(np.arange(W), np.arange(H)), -1).reshape(-1, 1, 2).astype(np.float64)
    und = cv2.undistortPoints(px, K, DIST).reshape(-1, 2)
    rays = np.hstack([und, np.ones((len(und), 1))])

    rng = np.random.default_rng(0)
    lines = []
    for i in range(args.views):
        a = -1.0 + 2.0 * i / max(1, args.views - 1)
        C = np.array([9.0 * np.sin(a), -9.0 * np.cos(a), 2.5 + 0.6 * np.sin(3 * a)]) + rng.normal(0, 0.2, 3)
        R, t = look_at(C, np.array([0.0, 0.5, 0.8]) + rng.normal(0, 0.3, 3))
        img = render(R, t, quads, textures, rays)
        name = f"view_{i:03d}.png"
        cv2.imwrite(os.path.join(args.out, "images", name), img)
        w, x, y, z = quat(R)
        lines.append(f"{name} {w} {x} {y} {z} {t[0]} {t[1]} {t[2]}")
        print("rendered", name)
    with open(os.path.join(args.out, "gt_poses.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "config.yaml"), "w") as f:
        f.write(f"""io:
  image_dir: images
  output_dir: out
cameras:
  - id: 0
    fx: {K[0, 0]}
    fy: {K[1, 1]}
    cx: {K[0, 2]}
    cy: {K[1, 2]}
    dist: [{', '.join(str(v) for v in DIST)}]
""")


def quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return w, x, y, z


if __name__ == "__main__":
    main()
