#!/usr/bin/env python3
"""Tiny GT-SDF overfit check for the bunny medial head."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
import trimesh.transformations as trafo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source import medial_field, points_to_surf_medial, sdf


def to_unit_cube(mesh):
    mesh = mesh.copy()
    center = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    mesh.apply_transform(trafo.translation_matrix(-center))
    mesh.apply_transform(trafo.scale_matrix(1.0 / mesh.extents.max()))
    return mesh


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_query_coords", type=int, default=1)
    parser.add_argument("--random_batches", type=int, default=0,
                        help="sample a fresh inside batch every step instead of overfitting one batch")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mesh = to_unit_cube(trimesh.load(REPO_ROOT / "data" / "bunny.obj", force="mesh"))
    pts = mesh.vertices[:, :3].astype(np.float32)
    queries = medial_field.make_medial_training_queries(
        mesh, 4096, near_surface_ratio=0.45, far_query_pts_ratio=0.35, seed=args.seed)
    phi = -sdf.get_signed_distance(mesh, queries)
    inside = np.flatnonzero(phi < -1e-5)
    if inside.shape[0] < args.batch_size:
        raise RuntimeError("Not enough inside queries for debug overfit.")
    ids = rng.choice(inside, args.batch_size, replace=False)
    query_batch = queries[ids].astype(np.float32)

    backbone, train_opt = points_to_surf_medial.load_pretrained_backbone(
        str(REPO_ROOT / "models" / "p2s_vanilla_model_149.pth"),
        str(REPO_ROOT / "models" / "p2s_vanilla_params.pth"),
        device)
    model = points_to_surf_medial.PointsToSurfMedialModel(
        backbone, use_query_coords=bool(args.use_query_coords)).to(device)
    import scipy.spatial as spatial
    kdtree = spatial.cKDTree(pts)
    patch_rng = np.random.RandomState(0)
    global_rng = np.random.RandomState(1)
    batch = medial_field._make_query_batch(
        query_batch, pts, kdtree, train_opt, patch_rng, global_rng, device)

    weights = medial_field.get_default_weights()
    weights["eikonal"] = 0.0
    weights["surface_reg"] = 0.0
    weights["orthogonality"] = 0.1

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    losses = []
    for step in range(args.steps):
        if args.random_batches:
            ids = rng.choice(inside, args.batch_size, replace=False)
            batch = medial_field._make_query_batch(
                queries[ids].astype(np.float32), pts, kdtree, train_opt, patch_rng, global_rng, device)
        optimizer.zero_grad()
        loss, parts = medial_field.compute_medial_losses_gt_sdf(
            model, batch, train_opt, mesh, weights=weights)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step in {0, 1, 2, 4, 9, 19, 49, args.steps - 1}:
            print("step {:03d} total={:.6f} max={:.6f} ins={:.6f} orth={:.6f}".format(
                step, losses[-1],
                float(parts.get("Maximality", torch.tensor(0.0)).detach().cpu()),
                float(parts.get("Inscription", torch.tensor(0.0)).detach().cpu()),
                float(parts.get("Orthogonality", torch.tensor(0.0)).detach().cpu())))
    print("loss_start={:.6f} loss_end={:.6f} ratio={:.6f}".format(
        losses[0], losses[-1], losses[-1] / max(losses[0], 1e-12)))


if __name__ == "__main__":
    main()
