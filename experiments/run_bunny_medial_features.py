#!/usr/bin/env python3
"""
Run the bunny medial-head experiment without notebook state.

Training uses only DMF-style self-supervised losses from source.medial_field.
The optional GT medial approximation is used only for diagnostics/checkpoint
selection after optimizer steps; it is never passed to compute_medial_losses.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
import trimesh
import trimesh.transformations as trafo

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_DEFAULT))

from source import data_loader, medial_field, points_to_surf_medial, sdf
from source.base import point_cloud


def to_unit_cube(mesh):
    mesh = mesh.copy()
    center = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    mesh.apply_transform(trafo.translation_matrix(-center))
    mesh.apply_transform(trafo.scale_matrix(1.0 / mesh.extents.max()))
    return mesh


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_root', type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument('--out_dir', type=Path, default=Path('models/bunny_medial_features_run'))
    parser.add_argument('--model_name', type=str, default='p2s_vanilla')
    parser.add_argument('--model_epoch', type=int, default=149)
    parser.add_argument('--num_query_pts', type=int, default=4000)
    parser.add_argument('--num_points', type=int, default=50000)
    parser.add_argument('--train_query_limit', type=int, default=0)
    parser.add_argument('--near_surface_query_ratio', type=float, default=0.45)
    parser.add_argument('--far_query_pts_ratio', type=float, default=0.35)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--nepoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--diagnostic_every', type=int, default=1)
    parser.add_argument('--diagnostic_query_pts', type=int, default=512)
    parser.add_argument('--diagnostic_score_percentile', type=float, default=50.0)
    parser.add_argument('--gt_diagnostic_sample_count', type=int, default=7000)
    parser.add_argument('--infer_batch', type=int, default=64)
    parser.add_argument('--max_medial_points', type=int, default=30000)
    parser.add_argument('--orthogonality_weight', type=float, default=0.1)
    parser.add_argument('--use_gt_sdf', action='store_true',
                        help='use GT mesh SDF for medial losses and direct inside-query training')
    parser.add_argument('--use_query_coords', action='store_true',
                        help='concatenate query xyz to frozen bottleneck for the medial head')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def make_dataset(root, pts, query_pts):
    shape_name = 'bunny'
    for subdir in ('04_pts', '05_query_pts', '05_query_dist'):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    np.save(root / '04_pts' / '{}.xyz.npy'.format(shape_name), pts.astype(np.float32))
    np.save(root / '05_query_pts' / '{}.ply.npy'.format(shape_name), query_pts.astype(np.float32))
    np.save(root / '05_query_dist' / '{}.ply.npy'.format(shape_name),
            np.zeros(len(query_pts), dtype=np.float32))
    (root / 'testset.txt').write_text('{}\n'.format(shape_name))


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = (repo_root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_file = repo_root / 'models' / '{}_model_{}.pth'.format(args.model_name, args.model_epoch)
    param_file = repo_root / 'models' / '{}_params.pth'.format(args.model_name)
    bunny_obj = repo_root / 'data' / 'bunny.obj'

    mesh_gt = to_unit_cube(trimesh.load(bunny_obj, force='mesh'))
    pts = mesh_gt.vertices[:, :3].astype(np.float32)
    if pts.shape[0] > args.num_points:
        ids = np.random.default_rng(args.seed).choice(pts.shape[0], args.num_points, replace=False)
        pts = pts[ids]

    query_pts = medial_field.make_medial_training_queries(
        mesh_gt, args.num_query_pts,
        near_surface_ratio=args.near_surface_query_ratio,
        far_query_pts_ratio=args.far_query_pts_ratio,
        seed=args.seed)
    if args.use_gt_sdf:
        query_phi_gt = -sdf.get_signed_distance(mesh_gt, query_pts)
        inside_query_mask = query_phi_gt < -1e-5
        if not np.any(inside_query_mask):
            raise RuntimeError('use_gt_sdf is on, but no inside training queries were sampled.')
        print('use_gt_sdf: keeping {}/{} inside training queries'.format(
            int(inside_query_mask.sum()), len(query_pts)), flush=True)
        query_pts = query_pts[inside_query_mask]
    if args.train_query_limit > 0:
        query_pts = query_pts[:args.train_query_limit]

    tmp_root = Path('/tmp/points2surf_bunny_medial_features_run')
    make_dataset(tmp_root, pts, query_pts)

    backbone, train_opt = points_to_surf_medial.load_pretrained_backbone(
        str(model_file), str(param_file), device)
    model = points_to_surf_medial.PointsToSurfMedialModel(
        backbone, use_query_coords=args.use_query_coords).to(device)

    dataset = data_loader.PointcloudPatchDataset(
        root=str(tmp_root),
        shape_list_filename='testset.txt',
        points_per_patch=train_opt.points_per_patch,
        patch_radius=train_opt.patch_radius,
        patch_features=train_opt.outputs,
        seed=args.seed,
        center=train_opt.patch_center,
        cache_capacity=1,
        pre_processed_patches=True,
        sub_sample_size=train_opt.sub_sample_size,
        reconstruction=False,
        epsilon=-1,
        uniform_subsample=getattr(train_opt, 'uniform_subsample', 0),
        fixed_subsample=getattr(train_opt, 'fixed_subsample', 0),
        num_workers=0)
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    gt_medial_pts = np.empty((0, 3), dtype=np.float32)
    diagnostic_queries = np.empty((0, 3), dtype=np.float32)
    diagnostic_history = {}
    best_diagnostic = None
    best_diagnostic_epoch = None

    if args.diagnostic_every:
        print('building diagnostic queries and GT approximation...', flush=True)
        diagnostic_queries = medial_field.make_medial_training_queries(
            mesh_gt, args.diagnostic_query_pts,
            near_surface_ratio=args.near_surface_query_ratio,
            far_query_pts_ratio=args.far_query_pts_ratio,
            seed=args.seed + 1)
        gt_medial_pts, _ = medial_field.approximate_gt_medial_surface_from_mesh(
            mesh_gt, sample_count=args.gt_diagnostic_sample_count,
            k=32, max_points=args.max_medial_points, seed=args.seed + 2)
        point_cloud.write_ply(str(out_dir / 'gt_medial_opposing_normals_diagnostic.ply'), gt_medial_pts)
        print('diagnostic queries: {}, diagnostic GT points: {}'.format(
            len(diagnostic_queries), len(gt_medial_pts)), flush=True)
        print('running initial diagnostic...', flush=True)
        initial_diag = medial_field.evaluate_predicted_medial_surface(
            model, train_opt, diagnostic_queries, pts, gt_medial_pts, device,
            out_dir=str(out_dir), tag='initial', batch_size=args.infer_batch,
            score_percentile=args.diagnostic_score_percentile, max_points=args.max_medial_points,
            mesh_gt_sdf=mesh_gt if args.use_gt_sdf else None)
        diagnostic_history['initial'] = initial_diag
        best_diagnostic = initial_diag
        best_diagnostic_epoch = 'initial'
        if initial_diag.get('ply_file'):
            shutil.copyfile(initial_diag['ply_file'], out_dir / 'pred_medial_best_by_diagnostic.ply')
        print('initial diagnostic:', initial_diag, flush=True)

    weights = medial_field.get_default_weights()
    weights['orthogonality'] = args.orthogonality_weight
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    history = {k: [] for k in ['Total', 'Maximality', 'Inscription',
                               'Orthogonality', 'Eikonal', 'Surface Regularizer']}
    gt_kdtree = None
    gt_train_rng = np.random.RandomState(args.seed + 10)
    gt_patch_rng = np.random.RandomState(0)
    gt_global_rng = np.random.RandomState(1)
    gt_steps_per_epoch = None
    if args.use_gt_sdf:
        import scipy.spatial as spatial
        gt_kdtree = spatial.cKDTree(pts)
        gt_steps_per_epoch = max(1, int(np.ceil(len(query_pts) / args.batch_size)))
        print('use_gt_sdf direct training: {} inside queries, {} steps/epoch'.format(
            len(query_pts), gt_steps_per_epoch), flush=True)

    def make_gt_training_batch():
        replace = len(query_pts) < args.batch_size
        ids = gt_train_rng.choice(len(query_pts), args.batch_size, replace=replace)
        return medial_field._make_query_batch(
            query_pts[ids].astype(np.float32), pts, gt_kdtree, train_opt,
            gt_patch_rng, gt_global_rng, device)

    train_start_time = time.time()
    model.train()
    for epoch in range(args.nepoch):
        epoch_parts = {k: [] for k in history}
        t = epoch / max(args.nepoch - 1, 1)
        if args.use_gt_sdf:
            batch_iter = (make_gt_training_batch() for _ in range(gt_steps_per_epoch))
        else:
            batch_iter = loader
        for batch in batch_iter:
            for key in batch:
                batch[key] = batch[key].to(device)
            if args.use_gt_sdf:
                loss, parts = medial_field.compute_medial_losses_gt_sdf(
                    model, batch, train_opt, mesh_gt, weights=weights, t=t)
            else:
                loss, parts = medial_field.compute_medial_losses(
                    model, batch, train_opt, weights=weights, t=t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for key in history:
                if key in parts:
                    epoch_parts[key].append(float(parts[key].detach().cpu()))
        for key in history:
            history[key].append(float(np.mean(epoch_parts[key])) if epoch_parts[key] else 0.0)
        print('epoch {:03d} total={:.4f} max={:.4f} ins={:.4f} orth={:.4f}'.format(
            epoch, history['Total'][-1], history['Maximality'][-1],
            history['Inscription'][-1], history['Orthogonality'][-1]), flush=True)

        if args.diagnostic_every and ((epoch + 1) % args.diagnostic_every == 0):
            model.eval()
            tag = 'epoch_{:03d}'.format(epoch)
            print('running {} diagnostic...'.format(tag), flush=True)
            diag = medial_field.evaluate_predicted_medial_surface(
                model, train_opt, diagnostic_queries, pts, gt_medial_pts, device,
                out_dir=str(out_dir), tag=tag, batch_size=args.infer_batch,
                score_percentile=args.diagnostic_score_percentile, max_points=args.max_medial_points,
                mesh_gt_sdf=mesh_gt if args.use_gt_sdf else None)
            diagnostic_history[tag] = diag
            print('{} diagnostic:'.format(tag), diag, flush=True)
            if best_diagnostic is None or diag['chamfer_l2'] < best_diagnostic['chamfer_l2']:
                best_diagnostic = diag
                best_diagnostic_epoch = tag
                torch.save(model.medial_head.state_dict(), out_dir / 'best_medial_head_by_diagnostic.pth')
                if diag.get('ply_file'):
                    shutil.copyfile(diag['ply_file'], out_dir / 'pred_medial_best_by_diagnostic.ply')
                print('saved best diagnostic checkpoint:', best_diagnostic_epoch, flush=True)
            model.train()

    train_seconds = time.time() - train_start_time
    torch.save(model.medial_head.state_dict(), out_dir / 'medial_head_final.pth')
    projection_image = None
    best_ply = out_dir / 'pred_medial_best_by_diagnostic.ply'
    if best_ply.is_file() and gt_medial_pts.size:
        pred_best = trimesh.load(best_ply).vertices.astype(np.float32)
        projection_image = str(out_dir / 'best_vs_gt_projection.png')
        medial_field.write_medial_projection_image(
            pts, pred_best, gt_medial_pts, projection_image)
    summary = {
        'device': str(device),
        'train_query_count': int(len(query_pts)),
        'diagnostic_query_count': int(len(diagnostic_queries)),
        'gt_diagnostic_count': int(len(gt_medial_pts)),
        'weights': weights,
        'history': history,
        'diagnostic_history': diagnostic_history,
        'best_diagnostic_epoch': best_diagnostic_epoch,
        'best_diagnostic': best_diagnostic,
        'projection_image': projection_image,
        'train_seconds': float(train_seconds),
        'note': 'GT medial approximation is diagnostic only and is not used as a loss signal.',
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('saved summary:', out_dir / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
