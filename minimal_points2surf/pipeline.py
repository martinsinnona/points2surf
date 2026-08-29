"""Reusable data, model, training, and inference helpers for the minimal experiment."""

from dataclasses import dataclass
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy.spatial as spatial
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm.auto import tqdm

from source import data_loader, medial_field, sdf, sdf_nn
from source.base import point_cloud, utils
from source.points_to_surf_model import PointNetfeat, PointsToSurfModel


@dataclass
class EvaluationData:
    """Fixed inputs used by evaluation-only SDF metrics."""

    prediction_tree: spatial.cKDTree
    global_sample: np.ndarray
    eval_points: np.ndarray
    eval_sdf: np.ndarray
    medial_points: np.ndarray
    medial_sdf: np.ndarray
    secondary_global_sample: np.ndarray | None = None


@dataclass
class SDFGrid:
    """Differentiable regular-grid cache of the frozen learned SDF."""

    values: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


class MedialQueryDataset(Dataset):
    """Query-only interior samples for self-supervised medial training."""

    def __init__(self, query_points):
        self.query_points = torch.as_tensor(query_points, dtype=torch.float32)

    def __len__(self):
        return len(self.query_points)

    def __getitem__(self, index):
        return {
            'imp_surf_query_point_ms': self.query_points[index],
            'imp_surf_dist_sign_ms': torch.tensor(0.0),
        }


def seed_everything(seed):
    """Seed the random generators used by this small experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_unit_mesh(mesh_path):
    """Load a mesh and normalize its largest extent to one."""
    mesh = trimesh.load(mesh_path, force='mesh').copy()
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale(1.0 / mesh.extents.max())
    mesh.fix_normals()
    return mesh


def sample_training_data(mesh, num_surface_points, num_query_points,
                         query_offsets, volume_padding, seed):
    """Sample equal volume and multiscale near-surface SDF queries."""
    rng = np.random.RandomState(seed)
    surface_points, _ = trimesh.sample.sample_surface(mesh, num_surface_points, seed=seed)
    num_volume = num_query_points // 2
    num_near = num_query_points - num_volume
    volume_points = rng.uniform(mesh.bounds[0] - volume_padding,
                                mesh.bounds[1] + volume_padding,
                                size=(num_volume, 3))
    offsets = np.atleast_1d(query_offsets)
    counts = np.full(len(offsets), num_near // len(offsets), dtype=int)
    counts[:num_near % len(offsets)] += 1
    near_points = []
    for index, (offset, count) in enumerate(zip(offsets, counts)):
        band_points, face_ids = trimesh.sample.sample_surface(
            mesh, count, seed=seed + index + 1)
        band_points += rng.uniform(-offset, offset, size=(count, 1)) * mesh.face_normals[face_ids]
        near_points.append(band_points)
    query_points = np.concatenate((volume_points, *near_points)).astype(np.float32)
    query_sdf = -sdf.get_signed_distance(mesh, query_points).astype(np.float32)
    return surface_points.astype(np.float32), query_points, query_sdf


def write_training_dataset(dataset_dir, points, query_points, query_sdf):
    """Write the minimal on-disk dataset expected by Points2Surf."""
    for folder in ('04_pts', '05_query_pts', '05_query_dist'):
        (dataset_dir / folder).mkdir(parents=True, exist_ok=True)
    np.save(dataset_dir / '04_pts' / 'bunny.xyz.npy', points)
    np.save(dataset_dir / '05_query_pts' / 'bunny.ply.npy', query_points)
    np.save(dataset_dir / '05_query_dist' / 'bunny.ply.npy', query_sdf)
    (dataset_dir / 'trainset.txt').write_text('bunny\n')


def seed_data_loader_worker(worker_id):
    """Give each worker independent patch-sampling random generators."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    worker_dataset = torch.utils.data.get_worker_info().dataset
    for name in ('rng', 'rng_global_sample'):
        if hasattr(worker_dataset, name):
            getattr(worker_dataset, name).seed(worker_seed)


def make_training_loader(dataset_dir, patches_per_epoch, batch_size, workers,
                         seed, points_per_patch=300, global_sample_size=1000,
                         fixed_global_context=False, do_augmentation=True):
    """Create the local/global Points2Surf training loader."""
    dataset = data_loader.PointcloudPatchDataset(
        root=str(dataset_dir), shape_list_filename='trainset.txt',
        points_per_patch=points_per_patch, patch_radius=0.0,
        patch_features=['imp_surf_magnitude', 'imp_surf_sign'], epsilon=-1,
        seed=seed, center='mean', cache_capacity=1, pre_processed_patches=True,
        sub_sample_size=global_sample_size,
        uniform_subsample=fixed_global_context,
        fixed_subsample=fixed_global_context, num_workers=0,
        do_augmentation=do_augmentation,
    )
    sampler = data_loader.RandomPointcloudPatchSampler(
        dataset, patches_per_shape=patches_per_epoch, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=workers, persistent_workers=workers > 0,
                        worker_init_fn=seed_data_loader_worker,
                        generator=generator)
    return dataset, sampler, loader


def build_model(device, learning_rate, points_per_patch=300,
                global_sample_size=1000):
    """Build the vanilla two-branch Points2Surf model and optimizer."""
    model = PointsToSurfModel(
        net_size_max=1024, num_points=points_per_patch, output_dim=2,
        use_point_stn=True, use_feat_stn=True, sym_op='max', use_query_point=True,
        sub_sample_size=global_sample_size, do_augmentation=True,
        single_transformer=False, shared_transformation=True,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    return model, optimizer


def load_pretrained_weights(model, checkpoint_path, invert_sign=False):
    """Load vanilla Points2Surf weights into the current model."""
    state = utils.torch_load(
        checkpoint_path, map_location=next(model.parameters()).device)
    state = {name.removeprefix('module.'): value for name, value in state.items()}
    if invert_sign:
        state['fc4.weight'][1].neg_()
        state['fc4.bias'][1].neg_()
    model.load_state_dict(state)


def make_prediction_head_optimizer(model, learning_rate):
    """Freeze PointNet features and train the SDF and optional medial heads."""
    trainable_prefixes = ('fc2.', 'bn2.', 'fc3.', 'bn3.', 'fc4.',
                          'medial_head.')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(trainable_prefixes))
    model.prediction_head_only = True
    parameters = [parameter for parameter in model.parameters()
                  if parameter.requires_grad]
    return torch.optim.Adam(parameters, lr=learning_rate)


def make_medial_optimizer(model, learning_rate):
    """Freeze the complete SDF model and optimize only the medial head."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.medial_head.parameters():
        parameter.requires_grad_(True)
    model.eval()
    return torch.optim.Adam(model.medial_head.parameters(), lr=learning_rate)


def add_query_medial_head(model, hidden_size=64, hidden_layers=2, smooth=True,
                          softplus_beta=100.0):
    """Attach a small interior medial-field head M(x) that sees only query xyz."""
    if softplus_beta <= 0.0:
        raise ValueError('softplus_beta must be positive.')
    activation = lambda: (nn.Softplus(beta=softplus_beta)
                          if smooth else nn.ReLU())
    layers = [nn.Linear(3, hidden_size), activation()]
    for _ in range(hidden_layers - 1):
        layers.extend((nn.Linear(hidden_size, hidden_size), activation()))
    layers.append(nn.Linear(hidden_size, 1))
    head = nn.Sequential(*layers)
    nn.init.constant_(head[-1].bias, -1.25)
    model.medial_head = head.to(next(model.parameters()).device)
    return model


def predict_medial(model, queries):
    """Predict a non-negative interior medial radius in mesh units."""
    return F.softplus(model.medial_head(queries)).squeeze(-1)


class PairPatchDataset(Dataset):
    """Training samples with either two local or two global point contexts."""

    def __init__(self, points, query_points, query_sdf, patch_kind,
                 points_per_patch, global_sample_size):
        if patch_kind not in ('local', 'global'):
            raise ValueError("patch_kind must be 'local' or 'global'.")
        if len(points) < 2 * points_per_patch:
            raise ValueError('Two local patches require at least twice points_per_patch points.')
        self.points = np.asarray(points, dtype=np.float32)
        self.query_points = np.asarray(query_points, dtype=np.float32)
        self.query_sdf = np.asarray(query_sdf, dtype=np.float32)
        self.patch_kind = patch_kind
        self.points_per_patch = points_per_patch
        self.global_sample_size = global_sample_size
        self.tree = spatial.cKDTree(self.points)

    def __len__(self):
        return len(self.query_points)

    def _local_patch(self, point_ids, query):
        patch = self.points[point_ids].copy()
        radius = utils.get_patch_radii(patch, query)
        return utils.model_space_to_patch_space(patch, query, radius), radius

    def __getitem__(self, index):
        query = self.query_points[index]
        target_sdf = self.query_sdf[index]
        _, point_ids = self.tree.query(query, k=2 * self.points_per_patch)
        point_ids = np.asarray(point_ids, dtype=np.int64)
        local_a, patch_radius = self._local_patch(point_ids[:self.points_per_patch], query)

        if self.patch_kind == 'local':
            branch_a = local_a
            branch_b, _ = self._local_patch(point_ids[self.points_per_patch:], query)
        else:
            rng = np.random.RandomState(np.random.randint(0, 2**32 - 1))
            branch_a = utils.get_point_cloud_sub_sample(
                self.global_sample_size, self.points, query, rng,
                uniform=False, fixed=False).astype(np.float32)
            branch_b = utils.get_point_cloud_sub_sample(
                self.global_sample_size, self.points, query, rng,
                uniform=False, fixed=False).astype(np.float32)

        return {
            'branch_a': branch_a.astype(np.float32),
            'branch_b': branch_b.astype(np.float32),
            'imp_surf_query_point_ms': query,
            'patch_radius_ms': np.array(patch_radius, dtype=np.float32),
            'imp_surf_magnitude_ms': np.array([abs(target_sdf)], dtype=np.float32),
            'imp_surf_dist_sign_ms': np.array([0.0 if target_sdf < 0.0 else 1.0], dtype=np.float32),
        }


class PairPatchPointsToSurfModel(nn.Module):
    """Points2Surf head with two same-type PointNet branches."""

    def __init__(self, patch_kind, points_per_patch=300, global_sample_size=1000,
                 net_size_max=1024, output_dim=2):
        super().__init__()
        if patch_kind not in ('local', 'global'):
            raise ValueError("patch_kind must be 'local' or 'global'.")
        self.patch_kind = patch_kind
        branch_size = points_per_patch if patch_kind == 'local' else global_sample_size
        branch_args = dict(net_size_max=net_size_max, num_points=branch_size,
                           num_scales=1, use_point_stn=True, use_feat_stn=True,
                           output_size=net_size_max, sym_op='max')
        self.feat_a = PointNetfeat(**branch_args)
        self.feat_b = PointNetfeat(**branch_args)
        self.fc1_a = nn.Linear(net_size_max, net_size_max // 2)
        self.fc1_b = nn.Linear(net_size_max, net_size_max // 2)
        self.bn1_a = nn.BatchNorm1d(net_size_max // 2)
        self.bn1_b = nn.BatchNorm1d(net_size_max // 2)
        self.fc2 = nn.Linear(net_size_max, net_size_max // 4)
        self.fc3 = nn.Linear(net_size_max // 4, net_size_max // 8)
        self.fc4 = nn.Linear(net_size_max // 8, output_dim)
        self.bn2 = nn.BatchNorm1d(net_size_max // 4)
        self.bn3 = nn.BatchNorm1d(net_size_max // 8)

    def _encode_branch(self, branch, query, encoder, projection, normalizer):
        if self.patch_kind == 'global':
            branch = branch - query.unsqueeze(1)
        features, _, _, _ = encoder(branch.transpose(1, 2))
        return F.relu(normalizer(projection(features)))

    def forward(self, batch):
        query = batch['imp_surf_query_point_ms']
        first = self._encode_branch(batch['branch_a'], query, self.feat_a, self.fc1_a, self.bn1_a)
        second = self._encode_branch(batch['branch_b'], query, self.feat_b, self.fc1_b, self.bn1_b)
        features = torch.cat((first, second), dim=1)
        features = F.relu(self.bn2(self.fc2(features)))
        return self.fc4(F.relu(self.bn3(self.fc3(features))))


def make_pair_training_loader(points, query_points, query_sdf, patch_kind,
                              patches_per_epoch, batch_size, workers, seed,
                              points_per_patch=300, global_sample_size=1000):
    """Create the two-local or two-global training loader."""
    dataset = PairPatchDataset(points, query_points, query_sdf, patch_kind,
                               points_per_patch, global_sample_size)
    generator = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(dataset, replacement=True, num_samples=patches_per_epoch,
                            generator=generator)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=workers, persistent_workers=workers > 0,
                        worker_init_fn=seed_data_loader_worker,
                        generator=generator)
    return dataset, sampler, loader


def build_pair_model(device, learning_rate, patch_kind, points_per_patch=300,
                     global_sample_size=1000):
    """Build a two-local or two-global model with the vanilla prediction heads."""
    model = PairPatchPointsToSurfModel(
        patch_kind, points_per_patch=points_per_patch,
        global_sample_size=global_sample_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    return model, optimizer


def make_evaluation_data(mesh, points, volume_padding, num_eval_points,
                         num_medial_points, medial_surface_samples,
                         medial_offset, seed, global_sample_size=1000,
                         fixed_global_context=False):
    """Prepare uniform-volume and evaluation-only GT-medial-axis targets."""
    if fixed_global_context:
        # Match utils.get_point_cloud_sub_sample(uniform=True, fixed=True).
        global_rng = np.random.RandomState(42)
        global_ids = global_rng.randint(0, len(points), size=global_sample_size)
    else:
        global_rng = np.random.RandomState(seed + 1)
        global_ids = global_rng.choice(len(points), size=global_sample_size,
                                       replace=False)
    global_sample = points[global_ids].astype(np.float32)
    eval_rng = np.random.RandomState(seed + 3)
    eval_points = eval_rng.uniform(mesh.bounds[0] - volume_padding,
                                  mesh.bounds[1] + volume_padding,
                                  size=(num_eval_points, 3)).astype(np.float32)
    if num_medial_points:
        medial_points, medial_radii = medial_field.approximate_medial_axis_voronoi_from_mesh(
            mesh, surface_sample_count=medial_surface_samples, min_radius=medial_offset,
            max_points=num_medial_points, seed=seed + 4)
    else:
        medial_points = np.empty((0, 3), dtype=np.float32)
        medial_radii = np.empty(0, dtype=np.float32)
    return EvaluationData(
        prediction_tree=spatial.cKDTree(points), global_sample=global_sample,
        eval_points=eval_points,
        eval_sdf=-sdf.get_signed_distance(mesh, eval_points).astype(np.float32),
        medial_points=medial_points, medial_sdf=-medial_radii,
    )


def make_pair_evaluation_data(mesh, points, volume_padding, num_eval_points,
                              num_medial_points, medial_surface_samples,
                              medial_offset, seed, patch_kind,
                              global_sample_size=1000):
    """Prepare common metric targets and, for globals, a second fixed context."""
    evaluation = make_evaluation_data(
        mesh, points, volume_padding, num_eval_points, num_medial_points,
        medial_surface_samples, medial_offset, seed, global_sample_size)
    if patch_kind == 'global':
        rng = np.random.RandomState(seed + 2)
        evaluation.secondary_global_sample = points[rng.choice(
            len(points), size=global_sample_size, replace=False)].astype(np.float32)
    return evaluation


def predict_sdf_queries(model, queries, points, evaluation, device, seed,
                        points_per_patch=300, batch_size=512,
                        query_conditioned_global=False):
    """Run vanilla local/global inference for arbitrary mesh-space queries."""
    queries = np.asarray(queries, dtype=np.float32)
    predictions, patch_rng = [], np.random.RandomState(seed)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(queries), batch_size):
            batch_queries = queries[start:start + batch_size]
            local_patches, patch_radii = [], []
            for query in batch_queries:
                ids = point_cloud.get_patch_kdtree(
                    evaluation.prediction_tree, patch_rng, query, patch_radius=0.0,
                    points_per_patch=points_per_patch, n_jobs=1)
                padding = np.logical_or(ids < 0, ids >= len(points))
                ids[padding] = 0
                patch = points[ids].copy()
                patch[padding] = query
                radius = utils.get_patch_radii(patch, query)
                local_patches.append(utils.model_space_to_patch_space(patch, query, radius))
                patch_radii.append(radius)
            if query_conditioned_global:
                global_samples = np.asarray([
                    utils.get_point_cloud_sub_sample(
                        len(evaluation.global_sample), points, query, patch_rng,
                        uniform=False, fixed=False)
                    for query in batch_queries], dtype=np.float32)
            else:
                global_samples = np.repeat(
                    evaluation.global_sample[None], len(batch_queries), axis=0)
            batch = {
                'patch_pts_ps': torch.from_numpy(np.asarray(local_patches, dtype=np.float32)).to(device),
                'pts_sub_sample_ms': torch.from_numpy(global_samples).to(device),
                'imp_surf_query_point_ms': torch.from_numpy(batch_queries).to(device),
            }
            output = model(batch)
            signed_distance = output[:, 0].abs() * torch.tanh(output[:, 1])
            radius = torch.as_tensor(
                patch_radii, device=device, dtype=signed_distance.dtype)
            predictions.append((signed_distance * radius).cpu().numpy())
    return np.concatenate(predictions)


def make_sdf_grid(model, points, evaluation, bounds, device, seed,
                  resolution=25, points_per_patch=300, batch_size=64):
    """Evaluate the frozen Points2Surf SDF once on a regular 3D grid."""
    bounds = np.asarray(bounds, dtype=np.float32)
    axes = [np.linspace(bounds[0, axis], bounds[1, axis], resolution,
                        dtype=np.float32) for axis in range(3)]
    grid_z, grid_y, grid_x = np.meshgrid(
        axes[2], axes[1], axes[0], indexing='ij')
    queries = np.stack(
        (grid_x.ravel(), grid_y.ravel(), grid_z.ravel()), axis=1)
    values = predict_sdf_queries(
        model, queries, points, evaluation, device, seed,
        points_per_patch=points_per_patch, batch_size=batch_size,
        query_conditioned_global=False)
    return SDFGrid(
        values=torch.from_numpy(values.reshape(
            resolution, resolution, resolution)).to(device)[None, None],
        lower=torch.from_numpy(bounds[0]).to(device),
        upper=torch.from_numpy(bounds[1]).to(device),
    )


def sample_sdf_grid(sdf_grid, queries):
    """Trilinearly sample cached SDF values at differentiable xyz queries."""
    coordinates = 2.0 * (
        queries - sdf_grid.lower) / (sdf_grid.upper - sdf_grid.lower) - 1.0
    coordinates = coordinates.reshape(1, 1, 1, -1, 3)
    return F.grid_sample(
        sdf_grid.values, coordinates, mode='bilinear',
        padding_mode='border', align_corners=True).reshape(-1)


def predict_sdf_grid_queries(sdf_grid, queries, batch_size=65536):
    """Evaluate the cached learned SDF quickly for NumPy query points."""
    queries = np.asarray(queries, dtype=np.float32)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(queries), batch_size):
            batch_queries = torch.from_numpy(
                queries[start:start + batch_size]).to(sdf_grid.values.device)
            predictions.append(
                sample_sdf_grid(sdf_grid, batch_queries).cpu().numpy())
    return np.concatenate(predictions)


def make_medial_training_loader(sdf_grid, bounds, num_points, batch_size, seed):
    """Make one shuffled pass over fixed uniform interior queries per epoch."""
    bounds = np.asarray(bounds, dtype=np.float32)
    rng = np.random.RandomState(seed)
    interior_points = []
    num_collected = 0
    empty_batches = 0
    while num_collected < num_points:
        candidates = rng.uniform(
            bounds[0], bounds[1], size=(max(4096, 3 * (num_points - num_collected)), 3)
        ).astype(np.float32)
        candidates = candidates[
            predict_sdf_grid_queries(sdf_grid, candidates) < 0.0]
        empty_batches = empty_batches + 1 if len(candidates) == 0 else 0
        if empty_batches == 10:
            raise ValueError('Frozen SDF grid has no interior samples in bounds.')
        interior_points.append(candidates)
        num_collected += len(candidates)
    dataset = MedialQueryDataset(
        np.concatenate(interior_points, axis=0)[:num_points])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator)
    return dataset, loader


def predict_medial_queries(model, queries, device, batch_size=4096):
    """Evaluate the query-only interior medial head on arbitrary xyz points."""
    queries = np.asarray(queries, dtype=np.float32)
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(queries), batch_size):
            batch_queries = torch.from_numpy(
                queries[start:start + batch_size]).to(device)
            predictions.append(predict_medial(model, batch_queries).cpu().numpy())
    return np.concatenate(predictions)


def evaluate_medial_queries(model, queries, target, device):
    """Evaluate predicted M against an evaluation-only medial-field target."""
    prediction = predict_medial_queries(model, queries, device)
    target = np.asarray(target, dtype=np.float32)
    return {
        'mse': float(np.mean((prediction - target) ** 2)),
        'mae': float(np.mean(np.abs(prediction - target))),
    }


def evaluate_box_medial_head(model, bounds, device, grid_resolution=33,
                             max_points=4096, seed=0):
    """Measure M radius error on evaluation-only box medial-axis samples."""
    axis_points, axis_radii = medial_field.approximate_box_medial_axis_from_bounds(
        bounds, grid_resolution=grid_resolution, max_points=max_points,
        seed=seed)
    prediction = predict_medial_queries(model, axis_points, device)
    return medial_field.medial_axis_field_metrics(prediction, axis_radii)


def evaluate_box_medial_volume(model, bounds, device, num_points=20000, seed=0):
    """Evaluate M on uniform box-interior points without using it for training."""
    bounds = np.asarray(bounds, dtype=np.float32)
    queries = np.random.RandomState(seed).uniform(
        bounds[0], bounds[1], size=(num_points, 3)).astype(np.float32)
    face_distances = np.concatenate(
        (queries - bounds[0], bounds[1] - queries), axis=1)
    target = np.partition(face_distances, 1, axis=1)[:, 1]
    prediction = predict_medial_queries(model, queries, device)
    return {
        'mae': float(np.mean(np.abs(prediction - target))),
        'under_fraction': float(np.mean(
            prediction < np.min(face_distances, axis=1))),
    }


def predict_pair_sdf_queries(model, queries, points, evaluation, device, seed,
                              patch_kind, points_per_patch=300, batch_size=512):
    """Run inference for a two-local or two-global model in mesh SDF units."""
    if patch_kind not in ('local', 'global'):
        raise ValueError("patch_kind must be 'local' or 'global'.")
    if patch_kind == 'global' and evaluation.secondary_global_sample is None:
        raise ValueError('Two-global inference requires a secondary global sample.')
    queries = np.asarray(queries, dtype=np.float32)
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(queries), batch_size):
            batch_queries = queries[start:start + batch_size]
            local_a, local_b, patch_radii = [], [], []
            if patch_kind == 'local':
                _, point_ids = evaluation.prediction_tree.query(
                    batch_queries, k=2 * points_per_patch)
                for query, ids in zip(batch_queries, point_ids):
                    first = points[ids[:points_per_patch]].copy()
                    second = points[ids[points_per_patch:]].copy()
                    radius = utils.get_patch_radii(first, query)
                    second_radius = utils.get_patch_radii(second, query)
                    local_a.append(utils.model_space_to_patch_space(first, query, radius))
                    local_b.append(utils.model_space_to_patch_space(second, query, second_radius))
                    patch_radii.append(radius)
                branch_a = np.asarray(local_a, dtype=np.float32)
                branch_b = np.asarray(local_b, dtype=np.float32)
            else:
                _, point_ids = evaluation.prediction_tree.query(batch_queries, k=points_per_patch)
                for query, ids in zip(batch_queries, point_ids):
                    patch_radii.append(utils.get_patch_radii(points[ids], query))
                branch_a = np.repeat(evaluation.global_sample[None], len(batch_queries), axis=0)
                branch_b = np.repeat(evaluation.secondary_global_sample[None], len(batch_queries), axis=0)
            batch = {
                'branch_a': torch.from_numpy(branch_a).to(device),
                'branch_b': torch.from_numpy(branch_b).to(device),
                'imp_surf_query_point_ms': torch.from_numpy(batch_queries).to(device),
            }
            output = model(batch)
            magnitude = output[:, 0].abs()
            sign = sdf_nn.post_process_sign(output[:, 1])
            radius = torch.as_tensor(patch_radii, device=device, dtype=magnitude.dtype)
            predictions.append((magnitude * sign * radius).cpu().numpy())
    return np.concatenate(predictions)


def find_gt_medial_ridges(mesh, volume_padding, medial_offset, candidate_count,
                          query_count, seed):
    """Find local SDF ridges using GT SDF, only for optional sampling experiments."""
    rng = np.random.RandomState(seed)
    candidates = rng.uniform(mesh.bounds[0] - volume_padding,
                             mesh.bounds[1] + volume_padding,
                             size=(candidate_count, 3)).astype(np.float32)
    center_distance = sdf.get_signed_distance(mesh, candidates)
    inside = center_distance > 0.0
    candidates, center_distance = candidates[inside], center_distance[inside]
    if not len(candidates):
        return np.empty((0, 3), np.float32), np.empty(0, np.float32), np.empty((0, 3), np.float32)
    directions = rng.normal(size=(3, 3)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    scores, best_directions = np.full(len(candidates), -np.inf, np.float32), np.zeros_like(candidates)
    for direction in directions:
        plus = sdf.get_signed_distance(mesh, candidates + medial_offset * direction)
        minus = sdf.get_signed_distance(mesh, candidates - medial_offset * direction)
        local_ridge = center_distance - np.maximum(plus, minus)
        improved = local_ridge > scores
        scores[improved], best_directions[improved] = local_ridge[improved], direction
    ridge_ids = np.flatnonzero(scores > 0.0)
    if not len(ridge_ids):
        return np.empty((0, 3), np.float32), np.empty(0, np.float32), np.empty((0, 3), np.float32)
    selected = ridge_ids[np.argsort(scores[ridge_ids])[-min(query_count, len(ridge_ids)):]]
    return candidates[selected], center_distance[selected], best_directions[selected]


def sample_medial_queries(mesh, volume_padding, medial_offset, band_offset,
                          candidate_count, query_count, seed):
    """Create center and two-sided ridge-band queries for the optional experiment."""
    centers, _, directions = find_gt_medial_ridges(
        mesh, volume_padding, medial_offset, candidate_count, query_count, seed)
    points = np.concatenate((centers - band_offset * directions, centers,
                             centers + band_offset * directions))
    return points.astype(np.float32), -sdf.get_signed_distance(mesh, points).astype(np.float32)


def train_pair_model(model, optimizer, loader, points, evaluation, device, seed,
                     patch_kind, epochs, eval_every, points_per_patch=300):
    """Train a same-context pair model and record uniform/medial SDF MSE."""
    history = {name: [] for name in (
        'total', 'magnitude', 'sign', 'eval_mse', 'eval_medial_mse')}
    progress = tqdm(range(epochs), desc=f'training two {patch_kind} patches')
    for epoch in progress:
        model.train()
        losses = {name: [] for name in ('total', 'magnitude', 'sign')}
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            prediction = model(batch)
            magnitude_target = batch['imp_surf_magnitude_ms'].squeeze() / batch['patch_radius_ms']
            magnitude_loss = sdf_nn.calc_loss_magnitude(prediction[:, 0], magnitude_target)
            sign_loss = sdf_nn.calc_loss_sign(prediction[:, 1], batch['imp_surf_dist_sign_ms'].squeeze())
            loss = magnitude_loss + sign_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses['total'].append(loss.item())
            losses['magnitude'].append(magnitude_loss.item())
            losses['sign'].append(sign_loss.item())
        for name in losses:
            history[name].append(float(np.mean(losses[name])))
        if (epoch + 1) % eval_every == 0:
            prediction = predict_pair_sdf_queries(
                model, evaluation.eval_points, points, evaluation, device, seed,
                patch_kind, points_per_patch)
            medial_prediction = predict_pair_sdf_queries(
                model, evaluation.medial_points, points, evaluation, device, seed,
                patch_kind, points_per_patch)
            history['eval_mse'].append(float(np.mean((prediction - evaluation.eval_sdf) ** 2)))
            history['eval_medial_mse'].append(float(np.mean(
                (medial_prediction - evaluation.medial_sdf) ** 2)))
        else:
            history['eval_mse'].append(np.nan)
            history['eval_medial_mse'].append(np.nan)
        postfix = {'loss': f"{history['total'][-1]:.5f}", 'lr': optimizer.param_groups[0]['lr']}
        if not np.isnan(history['eval_mse'][-1]):
            postfix['eval_mse'] = f"{history['eval_mse'][-1]:.5f}"
            if not np.isnan(history['eval_medial_mse'][-1]):
                postfix['medial_mse'] = f"{history['eval_medial_mse'][-1]:.5f}"
        progress.set_postfix(postfix)
    return history


def calc_eikonal_loss(model, batch):
    """Penalize deviations from unit distance-gradient norm."""
    eikonal_batch = batch.copy()
    query_offset = torch.zeros_like(
        batch['imp_surf_query_point_ms'], requires_grad=True)
    patch_radius = batch['patch_radius_ms'].reshape(-1)
    eikonal_batch['patch_pts_ps'] = (
        batch['patch_pts_ps']
        - query_offset.unsqueeze(1) / patch_radius[:, None, None])
    eikonal_batch['imp_surf_query_point_ms'] = (
        batch['imp_surf_query_point_ms'] + query_offset)

    was_training = model.training
    model.eval()
    try:
        output = model(eikonal_batch)
        signed_distance = output[:, 0].abs() * torch.tanh(output[:, 1])
        signed_distance = signed_distance * patch_radius
        gradient = torch.autograd.grad(
            signed_distance.sum(), query_offset, create_graph=True)[0]
    finally:
        model.train(was_training)
    return ((torch.linalg.vector_norm(gradient, dim=1) - 1.0) ** 2).mean()


def _batch_at_queries(batch, queries):
    """Recenter the current local patches at differentiable queries."""
    old_queries = batch['imp_surf_query_point_ms']
    radii = batch['patch_radius_ms'].reshape(-1, 1, 1)
    patch_points = batch['patch_pts_ps'] * radii + old_queries.unsqueeze(1)
    moved = batch.copy()
    moved['patch_pts_ps'] = (patch_points - queries.unsqueeze(1)) / radii
    moved['imp_surf_query_point_ms'] = queries
    return moved


def _batch_at_new_neighborhoods(batch, queries, points, prediction_tree,
                                points_per_patch):
    """Build local patches at projected centers for the uncached ablation."""
    query_array = queries.detach().cpu().numpy()
    _, point_ids = prediction_tree.query(query_array, k=points_per_patch)
    point_ids = np.asarray(point_ids, dtype=np.int64)
    if point_ids.ndim == 1:
        point_ids = point_ids[None, :]
    patch_points = np.asarray(points, dtype=np.float32)[point_ids]
    radii = np.asarray([
        utils.get_patch_radii(patch, query)
        for patch, query in zip(patch_points, query_array)
    ], dtype=np.float32)
    patch_points = torch.as_tensor(
        patch_points, device=queries.device, dtype=queries.dtype)
    radii = torch.as_tensor(radii, device=queries.device, dtype=queries.dtype)
    moved = batch.copy()
    moved['patch_pts_ps'] = (
        patch_points - queries.unsqueeze(1)) / radii[:, None, None]
    moved['patch_radius_ms'] = radii
    moved['imp_surf_query_point_ms'] = queries
    moved['imp_surf_query_point_ps'] = torch.zeros_like(queries)
    return moved


def _signed_sdf_from_batch(model, batch):
    """Convert Points2Surf outputs to a signed distance in mesh units."""
    output = model(batch)
    signed_distance = output[:, 0].abs() * torch.tanh(output[:, 1])
    return signed_distance * batch['patch_radius_ms'].reshape(-1)


def calc_medial_losses(model, batch, weights, sdf_grid=None,
                       normalize_sdf_gradient=True, points=None,
                       prediction_tree=None, points_per_patch=300):
    """Deep Medial Fields losses for the query-only interior M head."""
    # The existing SDF sign target only selects interior samples; M has no target.
    inside = batch['imp_surf_dist_sign_ms'].reshape(-1) < 0.5
    if not torch.any(inside):
        zero = next(model.medial_head.parameters()).sum() * 0.0
        return zero, {name: zero for name in (
            'maximality', 'inscription', 'orthogonality')}

    interior_batch = {
        name: (value[inside] if torch.is_tensor(value)
               and value.shape[:1] == inside.shape else value)
        for name, value in batch.items()
    }
    queries = interior_batch['imp_surf_query_point_ms'].detach().requires_grad_(True)
    if sdf_grid is None:
        if points is None or prediction_tree is None:
            raise ValueError('Uncached SDF sampling requires points and prediction_tree.')
        phi = _signed_sdf_from_batch(
            model, _batch_at_queries(interior_batch, queries))
    else:
        phi = sample_sdf_grid(sdf_grid, queries)
    phi_grad = torch.autograd.grad(phi.sum(), queries)[0].detach()
    phi = phi.detach()
    medial = predict_medial(model, queries)
    medial_grad = torch.autograd.grad(
        medial.sum(), queries, create_graph=True)[0]

    maximality = torch.mean(F.relu(torch.abs(phi) - medial) ** 2)

    sdf_normal = phi_grad
    if normalize_sdf_gradient:
        sdf_normal = sdf_normal / torch.linalg.vector_norm(
            sdf_normal, dim=1, keepdim=True).clamp(min=1e-8)
    projection_normal = phi_grad / torch.linalg.vector_norm(
        phi_grad, dim=1, keepdim=True).clamp(min=1e-8)
    unsigned_grad = torch.sign(phi).unsqueeze(-1) * projection_normal
    centers = queries + unsigned_grad * (medial - torch.abs(phi)).unsqueeze(-1)
    if sdf_grid is None:
        center_batch = _batch_at_new_neighborhoods(
            interior_batch, centers, points, prediction_tree, points_per_patch)
        center_phi = _signed_sdf_from_batch(model, center_batch)
    else:
        center_phi = sample_sdf_grid(sdf_grid, centers)
    inscription = torch.mean((torch.abs(center_phi) - medial) ** 2)

    orthogonality = torch.mean(
        torch.sum(medial_grad * sdf_normal, dim=1) ** 2)
    losses = {
        'maximality': maximality,
        'inscription': inscription,
        'orthogonality': orthogonality,
    }
    total = sum(weights[name] * value for name, value in losses.items())
    return total, losses


def train_medial_model(model, optimizer, loader, sdf_grid,
                       device, epochs, weights, normalize_sdf_gradient=True,
                       points=None, prediction_tree=None, points_per_patch=300,
                       description='training medial field',
                       q_mdf_evaluation=None, lr_decay_epoch=None,
                       lr_decay_factor=0.2):
    """Train only the query-based medial head against the frozen SDF."""
    training_names = (
        'medial_total', 'maximality', 'inscription', 'orthogonality')
    history = {name: [] for name in (*training_names, 'q_mdf_mse')}
    q_eval = None
    if q_mdf_evaluation is not None:
        q_queries, q_target = q_mdf_evaluation
        with torch.no_grad():
            q_queries = torch.as_tensor(
                q_queries, device=device, dtype=sdf_grid.values.dtype)
            q_target = torch.as_tensor(
                q_target, device=device, dtype=sdf_grid.values.dtype)
            q_abs_sdf = torch.abs(sample_sdf_grid(sdf_grid, q_queries))
        q_eval = q_queries, q_target, q_abs_sdf
    progress = tqdm(range(epochs), desc=description)
    model.eval()
    for epoch in progress:
        losses = {name: [] for name in training_names}
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            total, terms = calc_medial_losses(
                model, batch, weights, sdf_grid, normalize_sdf_gradient,
                points, prediction_tree, points_per_patch)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            losses['medial_total'].append(total.item())
            for name, value in terms.items():
                losses[name].append(value.item())
        for name in training_names:
            history[name].append(float(np.mean(losses[name])))
        if q_eval is None:
            history['q_mdf_mse'].append(np.nan)
        else:
            q_queries, q_target, q_abs_sdf = q_eval
            with torch.no_grad():
                q_prediction = predict_medial(model, q_queries) - q_abs_sdf
                history['q_mdf_mse'].append(float(
                    F.mse_loss(q_prediction, q_target).item()))
        if lr_decay_epoch is not None and epoch + 1 == lr_decay_epoch:
            for parameter_group in optimizer.param_groups:
                parameter_group['lr'] *= lr_decay_factor
        progress.set_postfix(
            medial=f"{history['medial_total'][-1]:.5f}",
            q_mdf_mse=f"{history['q_mdf_mse'][-1]:.5f}",
            lr=optimizer.param_groups[0]['lr'])
    return history


def train_model(model, optimizer, loader, dataset, dataset_dir,
                query_points, query_sdf, query_kinds, mesh, points, evaluation,
                device, seed, epochs, eval_every, patches_per_epoch, batch_size,
                workers, use_medial_sampling=False, medial_add_epoch=0,
                medial_sampler=None, points_per_patch=300, global_sample_size=1000,
                lr_decay_epoch=None, eikonal_weight=0.0, restore_best=False,
                direct_magnitude_loss=False, query_conditioned_global=False,
                select_best_by_mae=False, fixed_global_context=False,
                medial_weights=None, medial_sdf_grid=None):
    """Train with tqdm and optional ridge-band samples, returning history and data state."""

    loss_names = ['total', 'signed', 'magnitude', 'sign']
    if eikonal_weight > 0.0:
        loss_names.append('eikonal')
    if medial_weights is not None:
        loss_names.extend(('medial_total', 'maximality', 'inscription',
                           'orthogonality'))
    history = {name: [] for name in (
        *loss_names, 'eval_mse', 'eval_mae', 'eval_medial_mse')}
    best_score, best_epoch, best_state = np.inf, None, None
    progress = tqdm(range(epochs), desc='training')
    for epoch in progress:
        if use_medial_sampling and epoch == medial_add_epoch:
            progress.set_description('sampling GT ridges')
            medial_points, medial_sdf = medial_sampler()
            query_points = np.concatenate((query_points, medial_points))
            query_sdf = np.concatenate((query_sdf, medial_sdf))
            query_kinds = np.concatenate((query_kinds, np.full(len(medial_points), 2, dtype=np.int8)))
            write_training_dataset(dataset_dir, points, query_points, query_sdf)
            dataset, _, loader = make_training_loader(
                dataset_dir, patches_per_epoch, batch_size, workers, seed + epoch,
                points_per_patch, global_sample_size, fixed_global_context,
                do_augmentation=dataset.do_augmentation)
            progress.set_description('training')

        if getattr(model, 'prediction_head_only', False):
            model.eval()
        else:
            model.train()
        losses = {name: [] for name in loss_names}
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            prediction = model(batch)
            magnitude_target = batch['imp_surf_magnitude_ms'].squeeze() / batch['patch_radius_ms']
            if direct_magnitude_loss:
                magnitude_loss = F.smooth_l1_loss(
                    prediction[:, 0].abs(), magnitude_target, beta=0.05)
            else:
                magnitude_loss = sdf_nn.calc_loss_magnitude(
                    prediction[:, 0], magnitude_target)
            sign_target = batch['imp_surf_dist_sign_ms'].squeeze()
            sign_loss = sdf_nn.calc_loss_sign(prediction[:, 1], sign_target)
            signed_target = magnitude_target * (2.0 * sign_target - 1.0)
            signed_prediction = prediction[:, 0].abs() * torch.tanh(prediction[:, 1])
            signed_loss = F.smooth_l1_loss(
                signed_prediction, signed_target, beta=0.05)
            loss = signed_loss + magnitude_loss + sign_loss
            if eikonal_weight > 0.0:
                eikonal_loss = calc_eikonal_loss(model, batch)
                loss = loss + eikonal_weight * eikonal_loss
            if medial_weights is not None:
                if medial_sdf_grid is None:
                    raise ValueError('medial_sdf_grid is required for medial losses.')
                medial_total, medial_losses = calc_medial_losses(
                    model, batch, medial_weights, medial_sdf_grid)
                loss = loss + medial_total
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses['total'].append(loss.item())
            losses['signed'].append(signed_loss.item())
            losses['magnitude'].append(magnitude_loss.item())
            losses['sign'].append(sign_loss.item())
            if eikonal_weight > 0.0:
                losses['eikonal'].append(eikonal_loss.item())
            if medial_weights is not None:
                losses['medial_total'].append(medial_total.item())
                for name, value in medial_losses.items():
                    losses[name].append(value.item())
        for name in losses:
            history[name].append(float(np.mean(losses[name])))
        if (epoch + 1) % eval_every == 0:
            prediction = predict_sdf_queries(model, evaluation.eval_points, points, evaluation,
                                             device, seed, points_per_patch,
                                             batch_size=batch_size,
                                             query_conditioned_global=query_conditioned_global)
            error = prediction - evaluation.eval_sdf
            history['eval_mse'].append(float(np.mean(error ** 2)))
            history['eval_mae'].append(float(np.mean(np.abs(error))))
            if len(evaluation.medial_points):
                medial_prediction = predict_sdf_queries(
                    model, evaluation.medial_points, points, evaluation,
                    device, seed, points_per_patch, batch_size=batch_size,
                    query_conditioned_global=query_conditioned_global)
                medial_mse = float(np.mean(
                    (medial_prediction - evaluation.medial_sdf) ** 2))
            else:
                medial_mse = np.nan
            history['eval_medial_mse'].append(medial_mse)
            score = (history['eval_mae'][-1] if select_best_by_mae else
                     history['eval_mse'][-1] + history['eval_medial_mse'][-1])
            if restore_best and score < best_score:
                best_score, best_epoch = score, epoch + 1
                best_state = {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}
        else:
            history['eval_mse'].append(np.nan)
            history['eval_mae'].append(np.nan)
            history['eval_medial_mse'].append(np.nan)
        if lr_decay_epoch is not None and epoch + 1 == lr_decay_epoch:
            for group in optimizer.param_groups:
                group['lr'] *= 0.1
        postfix = {'loss': f"{history['total'][-1]:.5f}", 'lr': optimizer.param_groups[0]['lr']}
        if eikonal_weight > 0.0:
            postfix['eikonal'] = f"{history['eikonal'][-1]:.5f}"
        if medial_weights is not None:
            postfix['medial'] = f"{history['medial_total'][-1]:.5f}"
        if not np.isnan(history['eval_mse'][-1]):
            postfix['eval_mse'] = f"{history['eval_mse'][-1]:.5f}"
            if not np.isnan(history['eval_medial_mse'][-1]):
                postfix['medial_mse'] = f"{history['eval_medial_mse'][-1]:.5f}"
        progress.set_postfix(postfix)
    if best_state is not None:
        model.load_state_dict(best_state)
    history['selected_epoch'] = best_epoch
    return history, query_points, query_sdf, query_kinds
