#!/usr/bin/env python3
"""Short box run to check medial losses and GT-axis metrics."""

import argparse
import json
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--num_query", type=int, default=512)
    parser.add_argument("--metric_query", type=int, default=512)
    parser.add_argument("--surface_points", type=int, default=5000)
    parser.add_argument("--axis_res", type=int, default=32)
    parser.add_argument("--axis_metric_points", type=int, default=512)
    parser.add_argument("--infer_batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(args.seed)

    mesh = to_unit_cube(trimesh.load(REPO_ROOT / "data" / "box.obj", force="mesh"))
    pts, _ = trimesh.sample.sample_surface(mesh, args.surface_points)
    pts = pts.astype(np.float32)

    train_queries = medial_field.make_medial_training_queries(
        mesh, args.num_query, near_surface_ratio=0.45, far_query_pts_ratio=0.35,
        seed=args.seed)
    train_phi = -sdf.get_signed_distance(mesh, train_queries)
    train_queries = train_queries[train_phi < -1e-5].astype(np.float32)
    if train_queries.shape[0] == 0:
        raise RuntimeError("No inside training queries sampled.")

    metric_queries = medial_field.make_medial_training_queries(
        mesh, args.metric_query, near_surface_ratio=0.45, far_query_pts_ratio=0.35,
        seed=args.seed + 1)
    metric_phi = -sdf.get_signed_distance(mesh, metric_queries)
    metric_inside = metric_phi < 0.0
    metric_queries = metric_queries[metric_inside].astype(np.float32)
    metric_phi = metric_phi[metric_inside].astype(np.float32)
    if metric_queries.shape[0] == 0:
        raise RuntimeError("No inside metric queries sampled.")

    gt_axis, gt_axis_radii = medial_field.approximate_box_medial_axis_from_bounds(
        mesh.bounds, grid_resolution=args.axis_res, max_points=30000, seed=args.seed)

    backbone, train_opt = points_to_surf_medial.load_pretrained_backbone(
        str(REPO_ROOT / "models" / "p2s_vanilla_model_149.pth"),
        str(REPO_ROOT / "models" / "p2s_vanilla_params.pth"),
        device)
    model = points_to_surf_medial.PointsToSurfMedialModel(
        backbone, use_query_coords=True).to(device)

    import scipy.spatial as spatial
    kdtree = spatial.cKDTree(pts)
    patch_rng = np.random.RandomState(0)
    global_rng = np.random.RandomState(1)
    weights = medial_field.get_default_weights()
    weights["eikonal"] = 0.0
    weights["surface_reg"] = 0.0
    weights["orthogonality"] = 0.1

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr)

    def make_batch():
        replace = train_queries.shape[0] < args.batch_size
        ids = rng.choice(train_queries.shape[0], args.batch_size, replace=replace)
        return medial_field._make_query_batch(
            train_queries[ids], pts, kdtree, train_opt, patch_rng, global_rng, device)

    def evaluate(tag):
        model.eval()
        med = medial_field.predict_medial_on_queries(
            model, train_opt, metric_queries, pts, device, batch_size=args.infer_batch)
        q_metrics, _ = medial_field.q_mdf_level_set_metrics(
            metric_queries, metric_phi, med, gt_axis, epsilon=0.01,
            inside_only=True, max_points=30000, rng=np.random.RandomState(args.seed),
            require_valid=True)
        projected = medial_field.evaluate_predicted_medial_surface(
            model, train_opt, metric_queries, pts, gt_axis, device,
            out_dir=None, tag=tag, batch_size=args.infer_batch,
            score_percentile=None, max_points=30000, mesh_gt_sdf=mesh)
        axis_field = medial_field.evaluate_medial_field_on_gt_axis(
            model, train_opt, gt_axis, gt_axis_radii, pts, device,
            batch_size=args.infer_batch, max_points=args.axis_metric_points,
            rng=np.random.RandomState(args.seed), epsilon=0.01)
        model.train()
        return {
            "tag": tag,
            "q_chamfer": q_metrics["chamfer_l2"],
            "q_count": q_metrics["q_mdf_level_set_count"],
            "q_underpred": q_metrics["q_mdf_underpred_fraction"],
            "q_band": q_metrics["q_mdf_valid_band_fraction"],
            "projected_chamfer": projected["chamfer_l2"],
            "projected_count": projected["count"],
            "projected_mean_abs_score": projected["mean_abs_score"],
            "gt_axis_mae": axis_field["gt_axis_residual_mae"],
            "gt_axis_rmse": axis_field["gt_axis_residual_rmse"],
            "gt_axis_valid": axis_field["gt_axis_valid_fraction"],
            "gt_axis_underpred": axis_field["gt_axis_underpred_fraction"],
        }

    history = []
    metrics = [evaluate("initial")]
    best_axis_metric = metrics[0]
    steps_per_epoch = max(1, int(np.ceil(train_queries.shape[0] / args.batch_size)))
    for epoch in range(args.epochs):
        parts_epoch = []
        for _ in range(steps_per_epoch):
            batch = make_batch()
            loss, parts = medial_field.compute_medial_losses_gt_sdf(
                model, batch, train_opt, mesh, weights=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            parts_epoch.append({
                key: float(value.detach().cpu())
                for key, value in parts.items()
                if key in {"Total", "Maximality", "Inscription", "Orthogonality"}
            })
        mean_parts = {
            key: float(np.mean([p[key] for p in parts_epoch if key in p]))
            for key in {"Total", "Maximality", "Inscription", "Orthogonality"}
        }
        history.append(mean_parts)
        metrics.append(evaluate(f"epoch_{epoch:03d}"))
        if metrics[-1]["gt_axis_mae"] < best_axis_metric["gt_axis_mae"]:
            best_axis_metric = metrics[-1]
        print(
            "epoch {epoch:03d} total={total:.6g} axis_mae={axis:.6g} "
            "proj_chamfer={proj:.6g} q_chamfer={q:.6g} q_under={under:.3f}".format(
                epoch=epoch, total=mean_parts["Total"],
                axis=metrics[-1]["gt_axis_mae"],
                proj=metrics[-1]["projected_chamfer"],
                q=metrics[-1]["q_chamfer"],
                under=metrics[-1]["q_underpred"] if metrics[-1]["q_underpred"] is not None else float("nan")),
            flush=True)

    print(json.dumps({
        "device": str(device),
        "train_inside_queries": int(train_queries.shape[0]),
        "metric_inside_queries": int(metric_queries.shape[0]),
        "gt_axis_points": int(gt_axis.shape[0]),
        "history": history,
        "metrics": metrics,
        "best_gt_axis_metric": best_axis_metric,
    }, indent=2))


if __name__ == "__main__":
    main()
