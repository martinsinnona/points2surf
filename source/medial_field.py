"""
Auto-supervised medial field losses (maximality, inscription, orthogonality)
using frozen SDF predictions and autograd gradients w.r.t. query points.
"""

import numpy as np
import torch
import torch.nn.functional as F

from source import sdf_nn


def post_process_medial(raw):
    """Non-negative medial scalar."""
    if raw.dim() == 1:
        raw = raw.unsqueeze(-1)
    return F.softplus(raw).squeeze(-1)


def process_sdf_prediction(sdf_raw, train_opt, patch_radius_ms):
    """
    Convert raw network SDF outputs to signed distance in model space.

    The pretrained Points2Surf SDF head follows the dataset/trimesh convention:
    positive values are inside watertight meshes.  DMF code and visualizations use
    the common level-set convention instead: inside is negative.
    """
    fixed_radius = train_opt.patch_radius > 0.0
    if 'imp_surf' in train_opt.outputs:
        phi = sdf_nn.post_process_distance(sdf_raw[:, 0:1])
        if not fixed_radius:
            phi = phi * patch_radius_ms.unsqueeze(1)
        return -phi.squeeze(-1)
    mag = sdf_nn.post_process_magnitude(sdf_raw[:, 0:1])
    sign = sdf_nn.post_process_sign(sdf_raw[:, 1:2])
    phi = mag * sign
    if not fixed_radius:
        phi = phi * patch_radius_ms.unsqueeze(1)
    return -phi.squeeze(-1)


def make_medial_training_queries(mesh, num_query_pts, near_surface_ratio=0.5,
                                 far_query_pts_ratio=0.3, seed=42):
    """
    Generate query points for self-supervised medial-field training.

    This samples locations where the DMF constraints are evaluated; it does not
    compute or return medial-axis targets.
    """
    from source import sdf

    rng = np.random.RandomState(seed)
    num_near = int(round(num_query_pts * near_surface_ratio))
    num_uniform = max(num_query_pts - num_near, 0)
    queries = []
    if num_near > 0:
        patch_radius = (1.0 + 3) / 128
        queries.append(sdf.get_query_pts_for_mesh(
            mesh, num_near, patch_radius, far_query_pts_ratio=far_query_pts_ratio, rng=rng))
    if num_uniform > 0:
        queries.append(rng.uniform(-0.75, 0.75, size=(num_uniform, 3)).astype(np.float32))
    if not queries:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(queries, axis=0).astype(np.float32)


def get_default_weights():
    return {
        "zero_set_sdf": 0.0,
        "zero_set_grad": 0.0,
        "volume_sdf": 0.0,
        "eikonal": 1.0,
        "q_eikonal": 0.0,
        "predicted_grad": 0.0,
        "sdf_gradient": 0.0,
        "maximality": 100.0,
        "inscription": 500.0,
        "orthogonality": 0.03,
        "surface_reg": 1.0,
        "curvature": 0.0,
    }


def sdf_abs_grad(sdf, sdf_grad, eps=1e-8):
    """Gradient of the unsigned distance d(x)=|phi(x)|."""
    sdf_sign = torch.where(sdf >= 0.0, torch.ones_like(sdf), -torch.ones_like(sdf))
    sdf_abs_grad = sdf_sign.unsqueeze(-1) * sdf_grad
    sdf_abs_grad_norm = torch.linalg.norm(sdf_abs_grad, dim=-1, keepdim=True).clamp(min=eps)
    return sdf_abs_grad, sdf_abs_grad_norm


def orthogonality_loss(mf_grad, sdf_grad, eps=1e-8):
    """
    Penalize the normal component of grad M along grad phi.

    This directly optimizes the condition grad M . grad phi = 0 while avoiding
    scale drift from non-unit SDF gradients.  Unlike a pure squared-cosine loss,
    it still has a useful gradient when grad M starts exactly parallel to grad
    phi, which is the failure mode we want to fix.
    """
    sdf_norm = torch.linalg.norm(sdf_grad, dim=-1, keepdim=True)
    valid = sdf_norm.squeeze(-1) > eps
    if not torch.any(valid):
        return torch.sum(mf_grad * sdf_grad) * 0.0
    sdf_unit = sdf_grad[valid] / sdf_norm[valid].clamp(min=eps)
    normal_component = torch.sum(mf_grad[valid] * sdf_unit, dim=-1)
    return torch.mean(normal_component ** 2)


def q_mdf_direction_loss(q_grad, sdf, sdf_grad, eps=1e-8):
    """
    Q-MDF form of medial orthogonality for M = Q + |phi|.

    From grad M . grad phi = 0:
      grad Q . n = -sign(phi) ||grad phi||
    Inside the shape (phi < 0), this becomes grad Q . n = ||grad phi||.
    """
    sdf_norm = torch.linalg.norm(sdf_grad, dim=-1, keepdim=True)
    valid = sdf_norm.squeeze(-1) > eps
    if not torch.any(valid):
        return torch.sum(q_grad * sdf_grad) * 0.0
    sdf_unit = sdf_grad[valid] / sdf_norm[valid].clamp(min=eps)
    q_normal = torch.sum(q_grad[valid] * sdf_unit, dim=-1)
    sdf_sign = torch.where(sdf[valid] >= 0.0, torch.ones_like(sdf[valid]), -torch.ones_like(sdf[valid]))
    target = -sdf_sign * sdf_norm[valid].squeeze(-1)
    return torch.mean((q_normal - target) ** 2)


def q_eikonal_loss(q_grad):
    """Encourage Q to have unit gradient norm."""
    q_norm = torch.linalg.norm(q_grad, dim=-1)
    return torch.mean((q_norm - 1.0) ** 2)


def eikonal_loss_from_gradients(agrad_norm, pgrad_norm=None, pgrad=None):
    """
    Eikonal loss for the active SDF gradient signal.

    When the auxiliary gradient head is disabled, pgrad is usually just a
    detached copy of agrad.  Counting it again only adds a no-gradient constant
    to the plotted loss, so include it only when it can receive gradients.
    """
    loss = torch.mean((agrad_norm - 1.0) ** 2)
    if pgrad is not None and pgrad.requires_grad and pgrad_norm is not None:
        loss = loss + torch.mean((pgrad_norm - 1.0) ** 2)
    return loss


def project_to_medial_spoke(query_ms, sdf, sdf_grad, medial, eps=1e-8):
    """
    Deep Medial Fields Eq. 5:
      proj_M(x) = x + grad|phi(x)| * (M(x) - |phi(x)|)

    This projects along the medial spoke, not to the closest medial-axis point.
    """
    sdf_dgrad, sdf_dgrad_norm = sdf_abs_grad(sdf, sdf_grad, eps=eps)
    sdf_dgrad_unit = sdf_dgrad / sdf_dgrad_norm
    spoke_offset = (medial - torch.abs(sdf)).unsqueeze(-1)
    return query_ms + spoke_offset * sdf_dgrad_unit


def batch_at_queries(batch, queries_ms):
    """Re-use patch tensors while re-centering local patch coordinates at new queries."""
    out = {k: v for k, v in batch.items()}
    out['imp_surf_query_point_ms'] = queries_ms

    old_queries_ms = batch['imp_surf_query_point_ms'].to(device=queries_ms.device, dtype=queries_ms.dtype)
    patch_radius_ms = batch['patch_radius_ms'].to(device=queries_ms.device, dtype=queries_ms.dtype)
    patch_radius = patch_radius_ms.view(-1, 1, 1)

    patch_pts_ps = batch['patch_pts_ps'].to(device=queries_ms.device, dtype=queries_ms.dtype)
    old_patch_pts_ms = patch_pts_ps * patch_radius + old_queries_ms.unsqueeze(1)
    out['patch_pts_ps'] = (old_patch_pts_ms - queries_ms.unsqueeze(1)) / patch_radius

    out['imp_surf_query_point_ps'] = torch.zeros_like(queries_ms)
    return out


def batch_select_rows(batch, mask):
    """Select per-query batch rows while leaving scalar/shared metadata unchanged."""
    out = {}
    n_rows = mask.shape[0]
    for key, value in batch.items():
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == n_rows:
            out[key] = value[mask]
        else:
            out[key] = value
    return out


def sdf_target_from_batch(batch):
    """Return GT SDF targets in the inside-negative convention used here."""
    if 'imp_surf_ms' in batch:
        target = batch['imp_surf_ms'].to(
            device=batch['imp_surf_query_point_ms'].device,
            dtype=batch['imp_surf_query_point_ms'].dtype).view(-1)
        return -target

    if 'imp_surf_magnitude_ms' not in batch or 'imp_surf_dist_sign_ms' not in batch:
        return None

    mag = batch['imp_surf_magnitude_ms'].to(
        device=batch['imp_surf_query_point_ms'].device,
        dtype=batch['imp_surf_query_point_ms'].dtype).view(-1)
    sign01 = batch['imp_surf_dist_sign_ms'].to(device=mag.device, dtype=mag.dtype).view(-1)
    signed_positive_inside = mag * (2.0 * sign01 - 1.0)
    return -signed_positive_inside


def closest_patch_surface_queries(batch):
    """Pick one observed surface point per patch for zero-set SDF supervision."""
    if 'patch_pts_ps' not in batch or 'patch_radius_ms' not in batch:
        return None
    q = batch['imp_surf_query_point_ms']
    patch_pts_ps = batch['patch_pts_ps'].to(device=q.device, dtype=q.dtype)
    patch_radius = batch['patch_radius_ms'].to(device=q.device, dtype=q.dtype).view(-1, 1)
    closest_id = torch.argmin(torch.linalg.norm(patch_pts_ps, dim=-1), dim=1)
    closest_ps = patch_pts_ps[torch.arange(patch_pts_ps.shape[0], device=q.device), closest_id]
    return q + closest_ps * patch_radius


def q_mdf_inside_mask_from_batch(batch, sdf):
    """
    Select samples where Q-MDF constraints should be evaluated.

    SDF anchors can use mixed inside/outside points, but the Q-MDF medial
    constraints are intended for inside-shape samples.  Prefer GT SDF targets
    when available so early predicted-SDF sign mistakes do not flip the Q signal.
    """
    target = sdf_target_from_batch(batch)
    if target is not None:
        inside_mask = target < -1e-5
        if not torch.any(inside_mask):
            inside_mask = target < 0.0
    else:
        inside_mask = sdf.detach() < -1e-5
        if not torch.any(inside_mask):
            inside_mask = sdf.detach() < 0.0
    if not torch.any(inside_mask):
        inside_mask = torch.ones_like(sdf, dtype=torch.bool)
    return inside_mask


def compute_medial_losses(model, batch, train_opt, weights=None, t=0.0):
    """
    Medial + auxiliary auto-supervised losses adapted from MAT-style training.
    Uses model.with_mf_grad(batch, train_opt) -> sdf, agrad, pgrad, mf, mf_grad.
    """
    if weights is None:
        weights = get_default_weights()

    volume_x = batch['imp_surf_query_point_ms']
    total_loss = torch.tensor(0.0, device=volume_x.device)
    loss_dict = {}

    sdf, agrad, pgrad, mf, mf_grad = model.with_mf_grad(batch, train_opt)

    agrad_norm = torch.linalg.norm(agrad, dim=-1, keepdim=True).clamp(min=1e-8)
    pgrad_norm = torch.linalg.norm(pgrad, dim=-1, keepdim=True).clamp(min=1e-8)

    if weights["volume_sdf"] > 0:
        sdf_target = sdf_target_from_batch(batch)
        if sdf_target is not None:
            volume_sdf_loss = F.smooth_l1_loss(sdf, sdf_target)
            loss_dict["Volume SDF"] = volume_sdf_loss
            total_loss = total_loss + weights["volume_sdf"] * volume_sdf_loss

    if weights["zero_set_sdf"] > 0 or weights["zero_set_grad"] > 0:
        zero_x = closest_patch_surface_queries(batch)
        if zero_x is not None:
            zero_x = zero_x.detach().requires_grad_(True)
            zero_batch = batch_at_queries(batch, zero_x)
            zero_sdf, zero_agrad, _, _, _ = model.with_mf_grad(zero_batch, train_opt)
            if weights["zero_set_sdf"] > 0:
                zero_set_sdf_loss = torch.mean(zero_sdf ** 2)
                loss_dict["Zero Set SDF"] = zero_set_sdf_loss
                total_loss = total_loss + weights["zero_set_sdf"] * zero_set_sdf_loss
            if weights["zero_set_grad"] > 0:
                zero_agrad_norm = torch.linalg.norm(zero_agrad, dim=-1).clamp(min=1e-8)
                zero_set_grad_loss = torch.mean((zero_agrad_norm - 1.0) ** 2)
                loss_dict["Zero Set Gradient"] = zero_set_grad_loss
                total_loss = total_loss + weights["zero_set_grad"] * zero_set_grad_loss

    if weights["eikonal"] > 0:
        eikonal_loss = eikonal_loss_from_gradients(agrad_norm, pgrad_norm, pgrad)
        loss_dict["Eikonal"] = eikonal_loss
        total_loss = total_loss + weights["eikonal"] * eikonal_loss

    if weights["predicted_grad"] > 0:
        predicted_grad_loss = torch.mean((agrad - pgrad) ** 2)
        loss_dict["Gradient Prediction"] = predicted_grad_loss
        total_loss = total_loss + weights["predicted_grad"] * predicted_grad_loss

    if weights["surface_reg"] > 0:
        surface_reg_loss = torch.mean(torch.exp(-100.0 * torch.abs(sdf)))
        loss_dict["Surface Regularizer"] = surface_reg_loss
        total_loss = total_loss + weights["surface_reg"] * surface_reg_loss

    if weights["curvature"] > 0:
        n_curv = min(2048, volume_x.shape[0])
        curv_x = volume_x[:n_curv].clone().detach().requires_grad_(True)
        curv_batch = batch_at_queries(batch, curv_x)
        _, curv_grad, _, _, _ = model.with_mf_grad(curv_batch, train_opt)
        curvature = []
        for i in range(curv_grad.shape[-1]):
            g = torch.autograd.grad(
                curv_grad[:, i].sum(), curv_x, create_graph=True)[0][:, i]
            curvature.append(g)
        curvature = torch.stack(curvature, dim=-1)
        curvature_loss = torch.mean(torch.sum(torch.abs(curvature), dim=0))
        loss_dict["Curvature"] = curvature_loss
        sched = 10 ** (-(1.0 + 4.0 * t))
        total_loss = total_loss + weights["curvature"] * sched * curvature_loss

    if weights["maximality"] > 0:
        maximality_loss = torch.mean(F.relu(torch.abs(sdf) - mf) ** 2)
        loss_dict["Maximality"] = maximality_loss
        total_loss = total_loss + weights["maximality"] * maximality_loss

    if weights["inscription"] > 0:
        c = project_to_medial_spoke(volume_x, sdf, agrad, mf)
        batch_c = batch_at_queries(batch, c)
        c_sdf, _, _, _, _ = model.with_mf_grad(batch_c, train_opt)
        inscription_loss = torch.mean((torch.abs(c_sdf) - mf) ** 2)
        loss_dict["Inscription"] = inscription_loss
        total_loss = total_loss + weights["inscription"] * inscription_loss

    if weights["orthogonality"] > 0:
        orth_loss = orthogonality_loss(mf_grad, agrad)
        loss_dict["Orthogonality"] = orth_loss
        total_loss = total_loss + weights["orthogonality"] * orth_loss

    loss_dict["Total"] = total_loss
    return total_loss, loss_dict


def attach_sdf_gradient(phi_const, query_ms, grad):
    """Use exact sampled SDF values in forward pass and supplied gradients in backward pass."""
    if query_ms.requires_grad:
        return phi_const + torch.sum(grad * (query_ms - query_ms.detach()), dim=-1)
    return phi_const


def gt_sdf_and_gradient(mesh, query_ms, device=None, eps=1e-8):
    """
    Return inside-negative GT SDF values and approximate SDF gradients.

    trimesh.proximity.signed_distance uses positive-inside.  We flip it to the
    level-set convention used by the medial code: negative inside, positive
    outside.  The gradient is approximated from the nearest surface point.
    """
    import trimesh.proximity
    from source import sdf

    query_np = query_ms.detach().cpu().numpy().astype(np.float32)
    phi_np = -sdf.get_signed_distance(mesh, query_np).astype(np.float32)
    closest_np, _, _ = trimesh.proximity.closest_point(mesh, query_np)
    closest_np = closest_np.astype(np.float32)

    to_query = query_np - closest_np
    dist = np.linalg.norm(to_query, axis=1, keepdims=True)
    grad_np = np.zeros_like(query_np, dtype=np.float32)
    valid = dist[:, 0] > eps
    phi_sign = np.sign(phi_np).astype(np.float32)
    phi_sign[phi_sign == 0.0] = 1.0
    grad_np[valid] = phi_sign[valid, None] * to_query[valid] / dist[valid]

    if np.any(~valid):
        _, _, face_ids = trimesh.proximity.closest_point(mesh, query_np[~valid])
        normals = mesh.face_normals[face_ids].astype(np.float32)
        grad_np[~valid] = normals

    out_device = query_ms.device if device is None else device
    grad = torch.from_numpy(grad_np).to(device=out_device, dtype=query_ms.dtype)
    phi_const = torch.from_numpy(phi_np).to(device=out_device, dtype=query_ms.dtype)
    phi = attach_sdf_gradient(phi_const, query_ms, grad)
    return phi, grad


def box_sdf_and_gradient(bounds, query_ms, device=None, eps=1e-8):
    """
    Analytic inside-negative SDF and gradient for an axis-aligned box.

    This avoids noisy closest-face queries for the box notebook.  At face/edge
    ties the gradient picks one valid outward normal, which is enough for the
    sampled training/evaluation points away from exact ties.
    """
    bounds = np.asarray(bounds, dtype=np.float32)
    mins_np = bounds[0]
    maxs_np = bounds[1]
    out_device = query_ms.device if device is None else device
    mins = torch.as_tensor(mins_np, device=out_device, dtype=query_ms.dtype)
    maxs = torch.as_tensor(maxs_np, device=out_device, dtype=query_ms.dtype)
    q = query_ms.to(device=out_device)

    closest = torch.minimum(torch.maximum(q, mins), maxs)
    outside_vec = q - closest
    outside_dist = torch.linalg.norm(outside_vec, dim=-1)

    def face_distances(points):
        lower = points - mins
        upper = maxs - points
        return torch.stack((
            lower[:, 0], upper[:, 0],
            lower[:, 1], upper[:, 1],
            lower[:, 2], upper[:, 2],
        ), dim=-1)

    face_dists = face_distances(q)
    inside = torch.all((q >= mins) & (q <= maxs), dim=-1)
    inside_dist, face_ids = torch.min(face_dists, dim=-1)

    phi_const = torch.where(inside, -inside_dist, outside_dist)

    grad = torch.zeros_like(q)
    outside_valid = (~inside) & (outside_dist > eps)
    if torch.any(outside_valid):
        grad[outside_valid] = outside_vec[outside_valid] / outside_dist[outside_valid].unsqueeze(-1).clamp(min=eps)

    if torch.any(inside):
        inside_ids = torch.nonzero(inside, as_tuple=False).squeeze(-1)
        normals = torch.zeros((inside_ids.numel(), 3), device=out_device, dtype=query_ms.dtype)
        selected_faces = face_ids[inside]
        normals[selected_faces == 0, 0] = -1.0
        normals[selected_faces == 1, 0] = 1.0
        normals[selected_faces == 2, 1] = -1.0
        normals[selected_faces == 3, 1] = 1.0
        normals[selected_faces == 4, 2] = -1.0
        normals[selected_faces == 5, 2] = 1.0
        grad[inside_ids] = normals

    phi = attach_sdf_gradient(phi_const.detach(), q, grad)
    return phi, grad


def box_sdf_gradient_valid_mask(bounds, query_ms, min_face_gap=0.0, device=None):
    """
    Valid differentiable-SDF mask for box samples.

    The box SDF gradient is ambiguous where the nearest face changes.  A positive
    min_face_gap excludes points whose two nearest faces are too close in
    distance, i.e. a band around box medial/tie sets.
    """
    out_device = query_ms.device if torch.is_tensor(query_ms) else device
    q = torch.as_tensor(query_ms, dtype=torch.float32, device=out_device)
    b = torch.as_tensor(bounds, dtype=q.dtype, device=out_device)
    mins, maxs = b[0], b[1]
    face_dists = torch.stack((
        q[:, 0] - mins[0], maxs[0] - q[:, 0],
        q[:, 1] - mins[1], maxs[1] - q[:, 1],
        q[:, 2] - mins[2], maxs[2] - q[:, 2],
    ), dim=-1)
    inside = torch.all((q >= mins) & (q <= maxs), dim=-1)
    sorted_dists, _ = torch.sort(face_dists, dim=-1)
    face_gap = sorted_dists[:, 1] - sorted_dists[:, 0]
    return inside & (face_gap >= min_face_gap)


def forward_medial_with_query_grad(model, batch):
    """
    Run the medial head while keeping autograd paths from M(x) to query x.

    PointsToSurfMedialModel.forward_medial intentionally wraps the frozen
    backbone in torch.no_grad(), which is right for inference but prevents
    orthogonality/curvature losses from seeing grad M with respect to x.
    """
    bottleneck = model.backbone.encode_bottleneck(batch)
    if hasattr(model, "medial_features"):
        bottleneck = model.medial_features(bottleneck, batch)
    if hasattr(model, "medial_raw_from_features"):
        return model.medial_raw_from_features(bottleneck, batch).squeeze(-1)
    return model.medial_head(bottleneck).squeeze(-1)


def compute_medial_losses_with_sdf_sampler(model, batch, sdf_sampler, weights=None, t=0.0):
    """
    Medial losses using an external SDF sampler.

    sdf_sampler(query_ms) must return (phi, grad_phi), both in model-space
    tensors.  Gradients flow into the medial head through M(x), grad M(x), and
    the projected-center dependence on M(x).
    """
    if weights is None:
        weights = get_default_weights()

    batch = {
        k: (v.clone() if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
    volume_x = batch['imp_surf_query_point_ms']
    if not volume_x.requires_grad:
        volume_x = volume_x.detach().requires_grad_(True)
        batch['imp_surf_query_point_ms'] = volume_x

    total_loss = torch.tensor(0.0, device=volume_x.device)
    loss_dict = {}

    mf_all = post_process_medial(forward_medial_with_query_grad(model, batch))
    mf_grad_all = torch.autograd.grad(
        mf_all.sum(), volume_x, create_graph=True, allow_unused=True)[0]
    if mf_grad_all is None:
        mf_grad_all = torch.zeros_like(volume_x)

    sdf_all, agrad_all = sdf_sampler(volume_x)
    inside_mask = sdf_all < -1e-5
    if not torch.any(inside_mask):
        inside_mask = sdf_all < 0.0
    if not torch.any(inside_mask):
        inside_mask = torch.ones_like(sdf_all, dtype=torch.bool)

    mf = mf_all[inside_mask]
    mf_grad = mf_grad_all[inside_mask]
    sdf = sdf_all[inside_mask]
    agrad = agrad_all[inside_mask]
    volume_x_inside = volume_x[inside_mask]
    if mf_grad is None:
        mf_grad = torch.zeros_like(volume_x_inside)
    pgrad = agrad

    agrad_norm = torch.linalg.norm(agrad, dim=-1, keepdim=True).clamp(min=1e-8)
    pgrad_norm = torch.linalg.norm(pgrad, dim=-1, keepdim=True).clamp(min=1e-8)

    if weights["eikonal"] > 0:
        eikonal_loss = eikonal_loss_from_gradients(agrad_norm, pgrad_norm, pgrad)
        loss_dict["Eikonal"] = eikonal_loss

    if weights["predicted_grad"] > 0:
        predicted_grad_loss = torch.mean((agrad - pgrad) ** 2)
        loss_dict["Gradient Prediction"] = predicted_grad_loss

    if weights["surface_reg"] > 0:
        surface_reg_loss = torch.mean(torch.exp(-100.0 * torch.abs(sdf)))
        loss_dict["Surface Regularizer"] = surface_reg_loss

    if weights["curvature"] > 0:
        n_curv = min(2048, volume_x.shape[0])
        curv_x = volume_x[:n_curv].clone().detach().requires_grad_(True)
        curv_batch = batch_at_queries(batch, curv_x)
        curv_mf = post_process_medial(forward_medial_with_query_grad(model, curv_batch))
        curv_grad = torch.autograd.grad(
            curv_mf.sum(), curv_x, create_graph=True, allow_unused=True)[0]
        if curv_grad is None:
            curv_grad = torch.zeros_like(curv_x)
        curvature = []
        for i in range(curv_grad.shape[-1]):
            g = torch.autograd.grad(
                curv_grad[:, i].sum(), curv_x, create_graph=True)[0][:, i]
            curvature.append(g)
        curvature = torch.stack(curvature, dim=-1)
        curvature_loss = torch.mean(torch.sum(torch.abs(curvature), dim=0))
        loss_dict["Curvature"] = curvature_loss
        sched = 10 ** (-(1.0 + 4.0 * t))
        total_loss = total_loss + weights["curvature"] * sched * curvature_loss

    if weights["maximality"] > 0:
        maximality_loss = torch.mean(F.relu(torch.abs(sdf) - mf) ** 2)
        loss_dict["Maximality"] = maximality_loss
        total_loss = total_loss + weights["maximality"] * maximality_loss

    if weights["inscription"] > 0:
        c = project_to_medial_spoke(volume_x_inside, sdf, agrad, mf)
        c_sdf, _ = sdf_sampler(c)
        inscription_loss = torch.mean((torch.abs(c_sdf) - mf) ** 2)
        loss_dict["Inscription"] = inscription_loss
        total_loss = total_loss + weights["inscription"] * inscription_loss

    if weights["orthogonality"] > 0:
        orth_loss = orthogonality_loss(mf_grad, agrad)
        loss_dict["Orthogonality"] = orth_loss
        total_loss = total_loss + weights["orthogonality"] * orth_loss

    loss_dict["Total"] = total_loss
    return total_loss, loss_dict


def compute_medial_losses_gt_sdf(model, batch, train_opt, mesh_gt, weights=None, t=0.0):
    """
    Medial losses using GT mesh SDF everywhere instead of the predicted SDF.

    The GT SDF values are sampled from the mesh, with a straight-through
    nearest-surface gradient for optimization.
    """
    return compute_medial_losses_with_sdf_sampler(
        model, batch,
        lambda query_ms: gt_sdf_and_gradient(mesh_gt, query_ms, device=query_ms.device),
        weights=weights, t=t)


def q_to_medial(q_value, phi_value):
    """Recover the medial field M from Q-MDF and the SDF magnitude."""
    return q_value + torch.abs(phi_value)


def q_mdf_head_state_dict(model):
    """State for every trainable branch that contributes to Q-MDF."""
    state = {"medial_head": model.medial_head.state_dict()}
    if hasattr(model, "query_residual_head"):
        state["query_residual_head"] = model.query_residual_head.state_dict()
    return state


def load_q_mdf_head_state_dict(model, state, strict=True):
    """Load Q-MDF head state, accepting older medial-head-only checkpoints."""
    if "medial_head" not in state:
        model.medial_head.load_state_dict(state, strict=strict)
        return
    model.medial_head.load_state_dict(state["medial_head"], strict=strict)
    if "query_residual_head" in state and hasattr(model, "query_residual_head"):
        model.query_residual_head.load_state_dict(state["query_residual_head"], strict=strict)


def forward_q_with_query_grad(model, batch):
    """Run the trainable Q head while preserving query-position gradients."""
    bottleneck = model.backbone.encode_bottleneck(batch)
    if hasattr(model, "medial_features"):
        bottleneck = model.medial_features(bottleneck, batch)
    if hasattr(model, "medial_raw_from_features"):
        return model.medial_raw_from_features(bottleneck, batch).squeeze(-1)
    return model.medial_head(bottleneck).squeeze(-1)


def medial_raw_from_features(model, features, batch):
    """Evaluate the trainable medial/Q head, including optional query-only branch."""
    if hasattr(model, "medial_raw_from_features"):
        return model.medial_raw_from_features(features, batch).squeeze(-1)
    return model.medial_head(features).squeeze(-1)


def q_mdf_features_with_query_grad(model, batch):
    """Return the shared Q-MDF head features with query-position gradients."""
    bottleneck = model.backbone.encode_bottleneck(batch)
    if hasattr(model, "medial_features"):
        bottleneck = model.medial_features(bottleneck, batch)
    return bottleneck


def attach_sdf_gradient_head(model, hidden_mult=0.5):
    """Attach a small trainable head that predicts the SDF gradient as a 3D vector."""
    import torch.nn as nn

    if hasattr(model, "sdf_gradient_head"):
        return model
    first_linear = next(m for m in model.medial_head.modules() if isinstance(m, nn.Linear))
    in_dim = first_linear.in_features
    hidden_dim = max(16, int(in_dim * hidden_mult))
    model.sdf_gradient_head = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=False),
        nn.Linear(hidden_dim, 3),
    ).to(next(model.parameters()).device)
    return model


def bind_q_mdf_model(model):
    """
    Make a PointsToSurfMedialModel interpret its head output as Q-MDF.

    The public with_mf_grad API still returns M so existing projection and DMF
    losses can run unchanged: M(x) = Q(x) + |phi(x)|.
    """
    import types

    def with_q_mdf_grad(self, batch_data, train_opt):
        batch_data = {
            k: (v.clone() if torch.is_tensor(v) else v)
            for k, v in batch_data.items()
        }
        q_pts = batch_data['imp_surf_query_point_ms']
        if not q_pts.requires_grad:
            q_pts = q_pts.detach().requires_grad_(True)
            batch_data['imp_surf_query_point_ms'] = q_pts

        bottleneck = self.backbone.encode_bottleneck(batch_data)
        sdf_raw = self.backbone.fc4(bottleneck)
        phi = process_sdf_prediction(sdf_raw, train_opt, batch_data['patch_radius_ms'])
        q_pred = post_process_medial(
            medial_raw_from_features(self, self.medial_features(bottleneck, batch_data), batch_data))
        mf = q_to_medial(q_pred, phi)

        agrad = torch.autograd.grad(
            phi.sum(), q_pts, create_graph=True, retain_graph=True, allow_unused=True)[0]
        if agrad is None:
            agrad = torch.zeros_like(q_pts)
        mf_grad = torch.autograd.grad(
            mf.sum(), q_pts, create_graph=True, allow_unused=True)[0]
        if mf_grad is None:
            mf_grad = torch.zeros_like(q_pts)

        pgrad = agrad.detach()
        return phi, agrad, pgrad, mf, mf_grad

    model.predict_q_mdf = True
    model.with_mf_grad = types.MethodType(with_q_mdf_grad, model)
    return model


def compute_q_mdf_losses(model, batch, train_opt, weights=None, t=0.0):
    """
    Q-MDF losses using the model's predicted SDF.

    The trainable head predicts Q and reconstructs M = Q + |phi| for the DMF
    constraints, including the original orthogonality objective
    grad M . grad phi = 0.  SDF supervision comes from the explicit
    predicted-SDF anchors that are enabled in the weights, typically volume SDF
    and zero-set terms.
    """
    if weights is None:
        weights = get_default_weights()

    batch = {
        k: (v.clone() if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
    volume_x = batch['imp_surf_query_point_ms']
    if not volume_x.requires_grad:
        volume_x = volume_x.detach().requires_grad_(True)
        batch['imp_surf_query_point_ms'] = volume_x

    total_loss = torch.tensor(0.0, device=volume_x.device)
    loss_dict = {}

    bottleneck = model.backbone.encode_bottleneck(batch)
    sdf_raw = model.backbone.fc4(bottleneck)
    sdf = process_sdf_prediction(sdf_raw, train_opt, batch['patch_radius_ms'])
    q_features = model.medial_features(bottleneck, batch) if hasattr(model, "medial_features") else bottleneck
    q_pred = post_process_medial(medial_raw_from_features(model, q_features, batch))
    mf = q_to_medial(q_pred, sdf)

    sdf_grad_create_graph = any(weights.get(k, 0.0) > 0 for k in (
        "eikonal", "inscription", "orthogonality", "curvature"))
    agrad = torch.autograd.grad(
        sdf.sum(), volume_x, create_graph=sdf_grad_create_graph,
        retain_graph=True, allow_unused=True)[0]
    if agrad is None:
        agrad = torch.zeros_like(volume_x)
    mf_grad = None
    if weights.get("orthogonality", 0.0) > 0:
        mf_grad = torch.autograd.grad(
            mf.sum(), volume_x, create_graph=True, retain_graph=True, allow_unused=True)[0]
        if mf_grad is None:
            mf_grad = torch.zeros_like(volume_x)
    q_grad = None
    if weights.get("q_eikonal", 0.0) > 0:
        q_grad = torch.autograd.grad(
            q_pred.sum(), volume_x, create_graph=True, retain_graph=True, allow_unused=True)[0]
        if q_grad is None:
            q_grad = torch.zeros_like(volume_x)
    pgrad = agrad.detach()

    agrad_norm = torch.linalg.norm(agrad, dim=-1, keepdim=True).clamp(min=1e-8)
    pgrad_norm = torch.linalg.norm(pgrad, dim=-1, keepdim=True).clamp(min=1e-8)
    q_loss_mask = q_mdf_inside_mask_from_batch(batch, sdf)
    volume_x_q = volume_x[q_loss_mask]
    mf_q = mf[q_loss_mask]
    sdf_q = sdf[q_loss_mask]
    agrad_q = agrad[q_loss_mask]
    mf_grad_q = mf_grad[q_loss_mask] if mf_grad is not None else None
    q_grad_q = q_grad[q_loss_mask] if q_grad is not None else None
    q_batch = batch_select_rows(batch, q_loss_mask)

    if weights["volume_sdf"] > 0:
        sdf_target = sdf_target_from_batch(batch)
        if sdf_target is not None:
            volume_sdf_loss = F.smooth_l1_loss(sdf, sdf_target)
            loss_dict["Volume SDF"] = volume_sdf_loss
            total_loss = total_loss + weights["volume_sdf"] * volume_sdf_loss

    if weights["zero_set_sdf"] > 0 or weights["zero_set_grad"] > 0:
        zero_x = closest_patch_surface_queries(batch)
        if zero_x is not None:
            zero_x = zero_x.detach().requires_grad_(True)
            zero_batch = batch_at_queries(batch, zero_x)
            zero_bottleneck = model.backbone.encode_bottleneck(zero_batch)
            zero_sdf_raw = model.backbone.fc4(zero_bottleneck)
            zero_sdf = process_sdf_prediction(
                zero_sdf_raw, train_opt, zero_batch['patch_radius_ms'])
            if weights["zero_set_sdf"] > 0:
                zero_set_sdf_loss = torch.mean(zero_sdf ** 2)
                loss_dict["Zero Set SDF"] = zero_set_sdf_loss
                total_loss = total_loss + weights["zero_set_sdf"] * zero_set_sdf_loss
            if weights["zero_set_grad"] > 0:
                zero_agrad = torch.autograd.grad(
                    zero_sdf.sum(), zero_x, create_graph=True, allow_unused=True)[0]
                if zero_agrad is None:
                    zero_agrad = torch.zeros_like(zero_x)
                zero_agrad_norm = torch.linalg.norm(zero_agrad, dim=-1).clamp(min=1e-8)
                zero_set_grad_loss = torch.mean((zero_agrad_norm - 1.0) ** 2)
                loss_dict["Zero Set Gradient"] = zero_set_grad_loss
                total_loss = total_loss + weights["zero_set_grad"] * zero_set_grad_loss

    if weights["eikonal"] > 0:
        eikonal_loss = eikonal_loss_from_gradients(agrad_norm, pgrad_norm, pgrad)
        loss_dict["Eikonal"] = eikonal_loss
        total_loss = total_loss + weights["eikonal"] * eikonal_loss

    if weights.get("q_eikonal", 0.0) > 0:
        q_eik_loss = q_eikonal_loss(q_grad_q)
        loss_dict["Q Eikonal"] = q_eik_loss
        total_loss = total_loss + weights["q_eikonal"] * q_eik_loss

    if weights["predicted_grad"] > 0:
        predicted_grad_loss = torch.mean((agrad - pgrad) ** 2)
        loss_dict["Gradient Prediction"] = predicted_grad_loss
        total_loss = total_loss + weights["predicted_grad"] * predicted_grad_loss

    if weights.get("sdf_gradient", 0.0) > 0 and hasattr(model, "sdf_gradient_head"):
        sdf_gradient_loss = F.smooth_l1_loss(model.sdf_gradient_head(q_features), agrad.detach())
        loss_dict["SDF Gradient"] = sdf_gradient_loss
        total_loss = total_loss + weights["sdf_gradient"] * sdf_gradient_loss

    if weights["surface_reg"] > 0:
        surface_reg_loss = torch.mean(torch.exp(-100.0 * torch.abs(sdf)))
        loss_dict["Surface Regularizer"] = surface_reg_loss
        total_loss = total_loss + weights["surface_reg"] * surface_reg_loss

    if weights["curvature"] > 0:
        n_curv = min(2048, volume_x.shape[0])
        curv_x = volume_x[:n_curv].clone().detach().requires_grad_(True)
        curv_batch = batch_at_queries(batch, curv_x)
        curv_q = post_process_medial(forward_q_with_query_grad(model, curv_batch))
        curv_sdf_raw = model.backbone.fc4(model.backbone.encode_bottleneck(curv_batch))
        curv_sdf = process_sdf_prediction(
            curv_sdf_raw, train_opt, curv_batch['patch_radius_ms'])
        curv_mf = q_to_medial(curv_q, curv_sdf)
        curv_grad = torch.autograd.grad(
            curv_mf.sum(), curv_x, create_graph=True, allow_unused=True)[0]
        if curv_grad is None:
            curv_grad = torch.zeros_like(curv_x)
        curvature = []
        for i in range(curv_grad.shape[-1]):
            g = torch.autograd.grad(
                curv_grad[:, i].sum(), curv_x, create_graph=True)[0][:, i]
            curvature.append(g)
        curvature = torch.stack(curvature, dim=-1)
        curvature_loss = torch.mean(torch.sum(torch.abs(curvature), dim=0))
        loss_dict["Curvature"] = curvature_loss
        sched = 10 ** (-(1.0 + 4.0 * t))
        total_loss = total_loss + weights["curvature"] * sched * curvature_loss

    if weights["maximality"] > 0:
        maximality_loss = torch.mean(F.relu(torch.abs(sdf_q) - mf_q) ** 2)
        loss_dict["Maximality"] = maximality_loss
        total_loss = total_loss + weights["maximality"] * maximality_loss

    if weights["inscription"] > 0:
        c = project_to_medial_spoke(volume_x_q, sdf_q, agrad_q, mf_q)
        c_batch = batch_at_queries(q_batch, c)
        c_bottleneck = model.backbone.encode_bottleneck(c_batch)
        c_sdf_raw = model.backbone.fc4(c_bottleneck)
        c_sdf = process_sdf_prediction(c_sdf_raw, train_opt, c_batch['patch_radius_ms'])
        inscription_loss = torch.mean((torch.abs(c_sdf) - mf_q) ** 2)
        loss_dict["Inscription"] = inscription_loss
        total_loss = total_loss + weights["inscription"] * inscription_loss

    if weights["orthogonality"] > 0:
        orth_loss = orthogonality_loss(mf_grad_q, agrad_q)
        loss_dict["Orthogonality"] = orth_loss
        total_loss = total_loss + weights["orthogonality"] * orth_loss

    loss_dict["Total"] = total_loss
    return total_loss, loss_dict


def compute_q_mdf_losses_with_sdf_sampler(
        model, batch, sdf_sampler, weights=None, t=0.0,
        sdf_gradient_target_sampler=None, sdf_valid_mask_sampler=None):
    """
    Train a head that predicts Q-MDF directly.

    DMF losses operate on the reconstructed medial field M = Q + |phi|.  Q is
    not supervised with a GT Q-MDF target; it must emerge from the medial losses.
    """
    if weights is None:
        weights = get_default_weights()

    batch = {
        k: (v.clone() if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
    volume_x = batch['imp_surf_query_point_ms']
    if not volume_x.requires_grad:
        volume_x = volume_x.detach().requires_grad_(True)
        batch['imp_surf_query_point_ms'] = volume_x

    total_loss = torch.tensor(0.0, device=volume_x.device)
    loss_dict = {}

    q_features = q_mdf_features_with_query_grad(model, batch)
    q_all = post_process_medial(medial_raw_from_features(model, q_features, batch))
    sdf_all, agrad_all = sdf_sampler(volume_x)
    mf_all = q_to_medial(q_all, sdf_all)
    mf_grad_all = None
    if weights.get("orthogonality", 0.0) > 0:
        mf_grad_all = torch.autograd.grad(
            mf_all.sum(), volume_x, create_graph=True, retain_graph=True, allow_unused=True)[0]
        if mf_grad_all is None:
            mf_grad_all = torch.zeros_like(volume_x)
    q_grad_all = None
    if weights.get("q_eikonal", 0.0) > 0:
        q_grad_all = torch.autograd.grad(
            q_all.sum(), volume_x, create_graph=True, retain_graph=True, allow_unused=True)[0]
        if q_grad_all is None:
            q_grad_all = torch.zeros_like(volume_x)

    inside_mask = sdf_all < -1e-5
    if not torch.any(inside_mask):
        inside_mask = sdf_all < 0.0
    fallback_inside_mask = inside_mask
    if sdf_valid_mask_sampler is not None:
        valid_mask = sdf_valid_mask_sampler(volume_x).to(device=volume_x.device, dtype=torch.bool)
        inside_mask = inside_mask & valid_mask
    if not torch.any(inside_mask):
        inside_mask = fallback_inside_mask
    if not torch.any(inside_mask):
        inside_mask = torch.ones_like(sdf_all, dtype=torch.bool)

    q_pred = q_all[inside_mask]
    mf = mf_all[inside_mask]
    mf_grad = mf_grad_all[inside_mask] if mf_grad_all is not None else None
    q_grad = q_grad_all[inside_mask] if q_grad_all is not None else None
    sdf_vals = sdf_all[inside_mask]
    agrad = agrad_all[inside_mask]
    volume_x_inside = volume_x[inside_mask]
    pgrad = agrad

    agrad_norm = torch.linalg.norm(agrad, dim=-1, keepdim=True).clamp(min=1e-8)
    pgrad_norm = torch.linalg.norm(pgrad, dim=-1, keepdim=True).clamp(min=1e-8)

    if weights.get("sdf_gradient", 0.0) > 0 and hasattr(model, "sdf_gradient_head"):
        sdf_grad_target = agrad if sdf_gradient_target_sampler is None else sdf_gradient_target_sampler(
            volume_x_inside).detach().to(device=agrad.device, dtype=agrad.dtype)
        sdf_grad_pred = model.sdf_gradient_head(q_features)[inside_mask]
        sdf_gradient_loss = F.smooth_l1_loss(sdf_grad_pred, sdf_grad_target)
        loss_dict["SDF Gradient"] = sdf_gradient_loss
        total_loss = total_loss + weights["sdf_gradient"] * sdf_gradient_loss

    if weights["eikonal"] > 0:
        eikonal_loss = eikonal_loss_from_gradients(agrad_norm, pgrad_norm, pgrad)
        loss_dict["Eikonal"] = eikonal_loss
        total_loss = total_loss + weights["eikonal"] * eikonal_loss

    if weights.get("q_eikonal", 0.0) > 0:
        q_eik_loss = q_eikonal_loss(q_grad)
        loss_dict["Q Eikonal"] = q_eik_loss
        total_loss = total_loss + weights["q_eikonal"] * q_eik_loss

    if weights["predicted_grad"] > 0:
        predicted_grad_loss = torch.mean((agrad - pgrad) ** 2)
        loss_dict["Gradient Prediction"] = predicted_grad_loss
        total_loss = total_loss + weights["predicted_grad"] * predicted_grad_loss

    if weights["surface_reg"] > 0:
        surface_reg_loss = torch.mean(torch.exp(-100.0 * torch.abs(sdf_vals)))
        loss_dict["Surface Regularizer"] = surface_reg_loss
        total_loss = total_loss + weights["surface_reg"] * surface_reg_loss

    if weights["curvature"] > 0:
        n_curv = min(2048, volume_x.shape[0])
        curv_x = volume_x[:n_curv].clone().detach().requires_grad_(True)
        curv_batch = batch_at_queries(batch, curv_x)
        curv_q = post_process_medial(forward_q_with_query_grad(model, curv_batch))
        curv_sdf, _ = sdf_sampler(curv_x)
        curv_mf = q_to_medial(curv_q, curv_sdf)
        curv_grad = torch.autograd.grad(
            curv_mf.sum(), curv_x, create_graph=True, allow_unused=True)[0]
        if curv_grad is None:
            curv_grad = torch.zeros_like(curv_x)
        curvature = []
        for i in range(curv_grad.shape[-1]):
            g = torch.autograd.grad(
                curv_grad[:, i].sum(), curv_x, create_graph=True)[0][:, i]
            curvature.append(g)
        curvature = torch.stack(curvature, dim=-1)
        curvature_loss = torch.mean(torch.sum(torch.abs(curvature), dim=0))
        loss_dict["Curvature"] = curvature_loss
        sched = 10 ** (-(1.0 + 4.0 * t))
        total_loss = total_loss + weights["curvature"] * sched * curvature_loss

    if weights["maximality"] > 0:
        maximality_loss = torch.mean(F.relu(torch.abs(sdf_vals) - mf) ** 2)
        loss_dict["Maximality"] = maximality_loss
        total_loss = total_loss + weights["maximality"] * maximality_loss

    if weights["inscription"] > 0:
        c = project_to_medial_spoke(volume_x_inside, sdf_vals, agrad, mf)
        c_sdf, _ = sdf_sampler(c)
        inscription_loss = torch.mean((torch.abs(c_sdf) - mf) ** 2)
        loss_dict["Inscription"] = inscription_loss
        total_loss = total_loss + weights["inscription"] * inscription_loss

    if weights["orthogonality"] > 0:
        orth_loss = orthogonality_loss(mf_grad, agrad)
        loss_dict["Orthogonality"] = orth_loss
        total_loss = total_loss + weights["orthogonality"] * orth_loss

    loss_dict["Total"] = total_loss
    return total_loss, loss_dict


def compute_q_mdf_losses_gt_sdf(
        model, batch, train_opt, mesh_gt, weights=None, t=0.0,
        sdf_gradient_target_sampler=None, sdf_valid_mask_sampler=None):
    """Q-MDF loss variant using GT mesh SDF and no direct GT Q-MDF target."""
    return compute_q_mdf_losses_with_sdf_sampler(
        model, batch,
        lambda query_ms: gt_sdf_and_gradient(mesh_gt, query_ms, device=query_ms.device),
        weights=weights, t=t, sdf_gradient_target_sampler=sdf_gradient_target_sampler,
        sdf_valid_mask_sampler=sdf_valid_mask_sampler)


def _make_query_batch(query_pts_ms, pts_ms, kdtree, train_opt, rng, rng_global, device):

    """Build Points2Surf batch tensors for arbitrary model-space query points."""

    import scipy.spatial as spatial
    from source.base import point_cloud, utils

    if kdtree is None:
        kdtree = spatial.cKDTree(pts_ms)

    batch_items = []

    for q in query_pts_ms:

        patch_ids = point_cloud.get_patch_kdtree(
            kdtree=kdtree, rng=rng, query_point=q,
            patch_radius=train_opt.patch_radius,
            points_per_patch=train_opt.points_per_patch, n_jobs=1)
        
        patch_ids_pad = np.logical_or(patch_ids < 0, patch_ids >= pts_ms.shape[0])
        patch_ids_safe = patch_ids.copy()
        patch_ids_safe[patch_ids_pad] = 0
        pts_patch_ms = pts_ms[patch_ids_safe]
        pts_patch_ms[patch_ids_pad] = q
        patch_radius_ms = (
            utils.get_patch_radii(pts_patch_ms, q)
            if train_opt.patch_radius <= 0.0 else train_opt.patch_radius)
        pts_patch_ps = utils.model_space_to_patch_space(
            pts_patch_ms, q, patch_radius_ms)
        q_ps = utils.model_space_to_patch_space_single_point(q, q, patch_radius_ms)
        sub_ms = utils.get_point_cloud_sub_sample(
            train_opt.sub_sample_size, pts_ms, q, rng_global,
            uniform=getattr(train_opt, 'uniform_subsample', 0),
            fixed=getattr(train_opt, 'fixed_subsample', 0))
        batch_items.append({
            'patch_pts_ps': torch.from_numpy(pts_patch_ps.astype(np.float32)),
            'patch_radius_ms': torch.tensor(patch_radius_ms, dtype=torch.float32),
            'pts_sub_sample_ms': torch.from_numpy(sub_ms.astype(np.float32)),
            'imp_surf_query_point_ms': torch.from_numpy(q.astype(np.float32)),
            'imp_surf_query_point_ps': torch.tensor(q_ps, dtype=torch.float32),
            'imp_surf_ms': torch.tensor([0.0], dtype=torch.float32),
            'imp_surf_magnitude_ms': torch.tensor([0.0], dtype=torch.float32),
            'imp_surf_dist_sign_ms': torch.tensor([0.0], dtype=torch.float32),
        })

    return {k: torch.stack([b[k] for b in batch_items]).to(device) for k in batch_items[0]}


def predict_fields_on_queries(model, train_opt, query_pts_ms, pts_ms, device, batch_size=64):
    """Run frozen SDF + medial head on arbitrary query points (patch-based)."""
    import scipy.spatial as spatial

    query_pts_ms = np.asarray(query_pts_ms, dtype=np.float32)
    pts_ms = np.asarray(pts_ms, dtype=np.float32)
    kdtree = spatial.cKDTree(pts_ms)
    rng = np.random.RandomState(0)
    rng_global = np.random.RandomState(1)

    phi_all = np.zeros(query_pts_ms.shape[0], dtype=np.float32)
    med_all = np.zeros(query_pts_ms.shape[0], dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, query_pts_ms.shape[0], batch_size):
            end = min(start + batch_size, query_pts_ms.shape[0])
            batch = _make_query_batch(
                query_pts_ms[start:end], pts_ms, kdtree, train_opt, rng, rng_global, device)
            sdf_raw, medial_raw = model(batch)
            phi = process_sdf_prediction(sdf_raw, train_opt, batch['patch_radius_ms'])
            med = post_process_medial(medial_raw)
            phi_all[start:end] = phi.cpu().numpy()
            med_all[start:end] = med.cpu().numpy()

    return phi_all, med_all


def predict_medial_on_queries(model, train_opt, query_pts_ms, pts_ms, device, batch_size=64):
    """Run only the medial head on arbitrary query points (patch-based)."""
    import scipy.spatial as spatial

    query_pts_ms = np.asarray(query_pts_ms, dtype=np.float32)
    pts_ms = np.asarray(pts_ms, dtype=np.float32)
    kdtree = spatial.cKDTree(pts_ms)
    rng = np.random.RandomState(0)
    rng_global = np.random.RandomState(1)

    med_all = np.zeros(query_pts_ms.shape[0], dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, query_pts_ms.shape[0], batch_size):
            end = min(start + batch_size, query_pts_ms.shape[0])
            batch = _make_query_batch(
                query_pts_ms[start:end], pts_ms, kdtree, train_opt, rng, rng_global, device)
            medial_raw = model.forward_medial(batch)
            med = post_process_medial(medial_raw)
            med_all[start:end] = med.cpu().numpy()

    return med_all


def predict_sdf_gradient_on_queries(model, train_opt, query_pts_ms, pts_ms, device, batch_size=64):
    """Run the optional SDF-gradient head on arbitrary query points."""
    import scipy.spatial as spatial

    if not hasattr(model, "sdf_gradient_head"):
        raise AttributeError("model has no sdf_gradient_head; call attach_sdf_gradient_head(model) first")

    query_pts_ms = np.asarray(query_pts_ms, dtype=np.float32)
    pts_ms = np.asarray(pts_ms, dtype=np.float32)
    kdtree = spatial.cKDTree(pts_ms)
    rng = np.random.RandomState(0)
    rng_global = np.random.RandomState(1)

    grad_all = np.zeros((query_pts_ms.shape[0], 3), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, query_pts_ms.shape[0], batch_size):
            end = min(start + batch_size, query_pts_ms.shape[0])
            batch = _make_query_batch(
                query_pts_ms[start:end], pts_ms, kdtree, train_opt, rng, rng_global, device)
            bottleneck = model.backbone.encode_bottleneck(batch)
            if hasattr(model, "medial_features"):
                bottleneck = model.medial_features(bottleneck, batch)
            grad = model.sdf_gradient_head(bottleneck)
            grad_all[start:end] = grad.cpu().numpy()

    return grad_all


def predict_medial_gradient_on_queries(
        model, train_opt, query_pts_ms, pts_ms, device, batch_size=64,
        mesh_gt_sdf=None, sdf_sampler=None):
    """Return grad M and grad phi on arbitrary query points."""
    import scipy.spatial as spatial

    query_pts_ms = np.asarray(query_pts_ms, dtype=np.float32)
    pts_ms = np.asarray(pts_ms, dtype=np.float32)
    kdtree = spatial.cKDTree(pts_ms)
    rng = np.random.RandomState(0)
    rng_global = np.random.RandomState(1)

    mf_grad_all = np.zeros((query_pts_ms.shape[0], 3), dtype=np.float32)
    sdf_grad_all = np.zeros((query_pts_ms.shape[0], 3), dtype=np.float32)

    model.eval()
    for start in range(0, query_pts_ms.shape[0], batch_size):
        end = min(start + batch_size, query_pts_ms.shape[0])
        batch = _make_query_batch(
            query_pts_ms[start:end], pts_ms, kdtree, train_opt, rng, rng_global, device)
        q_pts = batch['imp_surf_query_point_ms'].detach().requires_grad_(True)
        batch['imp_surf_query_point_ms'] = q_pts

        if sdf_sampler is not None:
            q_pred = post_process_medial(forward_q_with_query_grad(model, batch))
            sdf_vals, agrad = sdf_sampler(q_pts)
            mf = q_to_medial(q_pred, sdf_vals)
            mf_grad = torch.autograd.grad(
                mf.sum(), q_pts, create_graph=False, allow_unused=True)[0]
            if mf_grad is None:
                mf_grad = torch.zeros_like(q_pts)
        elif mesh_gt_sdf is None:
            _, agrad, _, _, mf_grad = model.with_mf_grad(batch, train_opt)
        else:
            q_pred = post_process_medial(forward_q_with_query_grad(model, batch))
            sdf_vals, agrad = gt_sdf_and_gradient(mesh_gt_sdf, q_pts, device=device)
            mf = q_to_medial(q_pred, sdf_vals)
            mf_grad = torch.autograd.grad(
                mf.sum(), q_pts, create_graph=False, allow_unused=True)[0]
            if mf_grad is None:
                mf_grad = torch.zeros_like(q_pts)

        mf_grad_all[start:end] = mf_grad.detach().cpu().numpy().astype(np.float32)
        sdf_grad_all[start:end] = agrad.detach().cpu().numpy().astype(np.float32)

    return mf_grad_all, sdf_grad_all


def medial_level_score(phi, medial):
    """Implicit medial sheet score: zero where a center satisfies M(x)=|phi(x)|."""
    return np.asarray(medial) - np.abs(np.asarray(phi))


def q_mdf_field(phi, medial):
    """
    Quasi-medial distance field from Q-MDF: MF minus SDF magnitude.

    The value approximates the unsigned distance field to the medial axis and is
    zero where M(x) = |phi(x)|.
    """
    return np.maximum(medial_level_score(phi, medial), 0.0)


def select_q_mdf_level_set_points(
        query_pts_ms, phi, medial, epsilon=0.01, inside_only=True,
        max_points=20000, rng=None, require_valid=True):
    """
    Select 3D query samples close to the Q-MDF zero level set.

    Q-MDF is max(M - |phi|, 0).  For an imperfect model, the clamp can hide
    large under-predictions where M < |phi| by mapping them to exactly zero.
    For evaluation, use the unclamped residual by default and keep only the
    narrow band |M - |phi|| <= epsilon.  This avoids a flat metric dominated
    by invalid clamped-zero regions.
    """
    query_pts_ms = np.asarray(query_pts_ms, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)
    medial = np.asarray(medial, dtype=np.float32)
    residual = medial_level_score(phi, medial)
    q_mdf = np.maximum(residual, 0.0)
    mask = np.isfinite(residual) & np.isfinite(phi) & np.isfinite(medial)
    if require_valid:
        mask &= np.abs(residual) <= epsilon
    else:
        mask &= q_mdf <= epsilon
    if inside_only:
        mask &= phi < 0.0
    selected = np.flatnonzero(mask)
    if max_points is not None and selected.shape[0] > max_points:
        rng = np.random.RandomState(0) if rng is None else rng
        selected = rng.choice(selected, max_points, replace=False)
    return query_pts_ms[selected], q_mdf[selected]


def q_mdf_level_set_metrics(
        query_pts_ms, phi, medial, gt_medial_pts, epsilon=0.01,
        inside_only=True, max_points=20000, rng=None, require_valid=True):
    """Compare the Q-MDF near-zero set to a GT medial-axis point cloud in 3D."""
    q_pts, q_vals = select_q_mdf_level_set_points(
        query_pts_ms, phi, medial, epsilon=epsilon, inside_only=inside_only,
        max_points=max_points, rng=rng, require_valid=require_valid)
    metrics = chamfer_l2_points(q_pts, gt_medial_pts)
    phi = np.asarray(phi, dtype=np.float32)
    medial = np.asarray(medial, dtype=np.float32)
    residual = medial_level_score(phi, medial)
    finite = np.isfinite(residual) & np.isfinite(phi) & np.isfinite(medial)
    if inside_only:
        finite &= phi < 0.0
    metrics.update({
        "q_mdf_epsilon": float(epsilon),
        "q_mdf_level_set_count": int(q_pts.shape[0]),
        "q_mdf_level_set_mean": float(np.mean(q_vals)) if q_vals.shape[0] else None,
        "q_mdf_underpred_fraction": (
            float(np.mean(residual[finite] < -epsilon)) if np.any(finite) else None),
        "q_mdf_valid_band_fraction": (
            float(np.mean(np.abs(residual[finite]) <= epsilon)) if np.any(finite) else None),
    })
    return metrics, q_pts


def medial_axis_field_metrics(medial, gt_radii, epsilon=0.01):
    """Evaluate medial-field residuals on GT medial-axis samples."""
    medial = np.asarray(medial, dtype=np.float32)
    gt_radii = np.asarray(gt_radii, dtype=np.float32)
    finite = np.isfinite(medial) & np.isfinite(gt_radii)
    if not np.any(finite):
        return {
            "gt_axis_field_count": 0,
            "gt_axis_residual_mae": np.inf,
            "gt_axis_residual_rmse": np.inf,
            "gt_axis_residual_bias": np.inf,
            "gt_axis_underpred_fraction": None,
            "gt_axis_valid_fraction": None,
        }

    residual = medial[finite] - gt_radii[finite]
    return {
        "gt_axis_field_count": int(residual.shape[0]),
        "gt_axis_residual_mae": float(np.mean(np.abs(residual))),
        "gt_axis_residual_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "gt_axis_residual_bias": float(np.mean(residual)),
        "gt_axis_underpred_fraction": float(np.mean(residual < -epsilon)),
        "gt_axis_valid_fraction": float(np.mean(np.abs(residual) <= epsilon)),
    }


def evaluate_medial_field_on_gt_axis(
        model, train_opt, gt_medial_pts, gt_medial_radii, pts_ms, device,
        batch_size=64, max_points=None, rng=None, epsilon=0.01):
    """
    Evaluate M(x) directly on GT medial-axis samples.

    Unlike projected-center Chamfer, this metric asks whether the learned medial
    radius agrees with the GT radius on known medial-axis locations.  It is an
    evaluation-only diagnostic and is not used by the optimizer.
    """
    gt_medial_pts = np.asarray(gt_medial_pts, dtype=np.float32)
    gt_medial_radii = np.asarray(gt_medial_radii, dtype=np.float32)
    if gt_medial_pts.shape[0] == 0:
        return medial_axis_field_metrics(
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            epsilon=epsilon)

    selected = np.arange(gt_medial_pts.shape[0])
    if max_points is not None and selected.shape[0] > max_points:
        rng = np.random.RandomState(0) if rng is None else rng
        selected = rng.choice(selected, max_points, replace=False)

    med = predict_medial_on_queries(
        model, train_opt, gt_medial_pts[selected], pts_ms, device,
        batch_size=batch_size)
    return medial_axis_field_metrics(med, gt_medial_radii[selected], epsilon=epsilon)


def predict_q_and_medial_on_queries(
        model, train_opt, query_pts_ms, pts_ms, device, batch_size=64,
        mesh_gt_sdf=None):
    """Predict Q and reconstruct M = Q + |phi| on arbitrary query points."""
    q_pred = predict_medial_on_queries(
        model, train_opt, query_pts_ms, pts_ms, device, batch_size=batch_size)
    if mesh_gt_sdf is not None:
        from source import sdf

        phi_pred = -sdf.get_signed_distance(mesh_gt_sdf, query_pts_ms)
    else:
        phi_pred, q_pred = predict_fields_on_queries(
            model, train_opt, query_pts_ms, pts_ms, device, batch_size=batch_size)
    med_pred = q_pred + np.abs(phi_pred)
    return phi_pred.astype(np.float32), q_pred.astype(np.float32), med_pred.astype(np.float32)


def evaluate_medial_orthogonality_on_queries(
        model, train_opt, query_pts_ms, pts_ms, device, batch_size=64,
        mesh_gt_sdf=None, inside_only=True, sdf_sampler=None,
        sdf_valid_mask_sampler=None):
    """Measure how much grad M points along grad phi on arbitrary queries."""
    import scipy.spatial as spatial

    query_pts_ms = np.asarray(query_pts_ms, dtype=np.float32)
    pts_ms = np.asarray(pts_ms, dtype=np.float32)
    if query_pts_ms.shape[0] == 0:
        return {
            "orth_count": 0,
            "orth_normal_abs_mean": None,
            "orth_normal_rms": None,
            "orth_cos_abs_mean": None,
            "orth_cos_rms": None,
        }

    kdtree = spatial.cKDTree(pts_ms)
    rng = np.random.RandomState(0)
    rng_global = np.random.RandomState(1)

    normal_components = []
    cosines = []
    count = 0

    model.eval()
    for start in range(0, query_pts_ms.shape[0], batch_size):
        end = min(start + batch_size, query_pts_ms.shape[0])
        batch = _make_query_batch(
            query_pts_ms[start:end], pts_ms, kdtree, train_opt, rng, rng_global, device)
        q_pts = batch['imp_surf_query_point_ms'].detach().requires_grad_(True)
        batch['imp_surf_query_point_ms'] = q_pts

        if sdf_sampler is not None:
            q_pred = post_process_medial(forward_q_with_query_grad(model, batch))
            sdf_vals, agrad = sdf_sampler(q_pts)
            mf = q_to_medial(q_pred, sdf_vals)
            mf_grad = torch.autograd.grad(
                mf.sum(), q_pts, create_graph=False, allow_unused=True)[0]
            if mf_grad is None:
                mf_grad = torch.zeros_like(q_pts)
        elif mesh_gt_sdf is None:
            sdf_vals, agrad, _, mf, mf_grad = model.with_mf_grad(batch, train_opt)
        else:
            q_pred = post_process_medial(forward_q_with_query_grad(model, batch))
            sdf_vals, agrad = gt_sdf_and_gradient(mesh_gt_sdf, q_pts, device=device)
            mf = q_to_medial(q_pred, sdf_vals)
            mf_grad = torch.autograd.grad(
                mf.sum(), q_pts, create_graph=False, allow_unused=True)[0]
            if mf_grad is None:
                mf_grad = torch.zeros_like(q_pts)

        mask = torch.isfinite(sdf_vals)
        if inside_only:
            mask = mask & (sdf_vals < 0.0)
        if sdf_valid_mask_sampler is not None:
            valid_mask = sdf_valid_mask_sampler(q_pts).to(device=q_pts.device, dtype=torch.bool)
            mask = mask & valid_mask
        if not torch.any(mask):
            continue

        mf_grad_sel = mf_grad[mask]
        agrad_sel = agrad[mask]
        sdf_norm = torch.linalg.norm(agrad_sel, dim=-1, keepdim=True).clamp(min=1e-8)
        mf_norm = torch.linalg.norm(mf_grad_sel, dim=-1, keepdim=True).clamp(min=1e-8)
        normal = torch.sum(mf_grad_sel * (agrad_sel / sdf_norm), dim=-1)
        cosine = normal / mf_norm.squeeze(-1)
        normal_components.append(normal.detach().cpu())
        cosines.append(cosine.detach().cpu())
        count += int(mask.sum().detach().cpu())

    if count == 0:
        return {
            "orth_count": 0,
            "orth_normal_abs_mean": None,
            "orth_normal_rms": None,
            "orth_cos_abs_mean": None,
            "orth_cos_rms": None,
        }

    normal_all = torch.cat(normal_components)
    cosine_all = torch.cat(cosines)
    return {
        "orth_count": count,
        "orth_normal_abs_mean": float(torch.mean(torch.abs(normal_all))),
        "orth_normal_rms": float(torch.sqrt(torch.mean(normal_all ** 2))),
        "orth_cos_abs_mean": float(torch.mean(torch.abs(cosine_all))),
        "orth_cos_rms": float(torch.sqrt(torch.mean(cosine_all ** 2))),
    }


def evaluate_q_mdf_gradient_objective_on_queries(
        model, train_opt, query_pts_ms, pts_ms, device, batch_size=64,
        mesh_gt_sdf=None, inside_only=True, sdf_sampler=None,
        sdf_valid_mask_sampler=None):
    """Measure Q-MDF direction and Q-eikonal residuals on arbitrary queries."""
    import scipy.spatial as spatial

    query_pts_ms = np.asarray(query_pts_ms, dtype=np.float32)
    pts_ms = np.asarray(pts_ms, dtype=np.float32)
    if query_pts_ms.shape[0] == 0:
        return {
            "q_grad_count": 0,
            "q_direction_abs_mean": None,
            "q_direction_rms": None,
            "q_objective_rms": None,
            "q_normal_mean": None,
            "q_eikonal_abs_mean": None,
            "q_eikonal_rms": None,
            "q_grad_norm_mean": None,
        }

    kdtree = spatial.cKDTree(pts_ms)
    rng = np.random.RandomState(0)
    rng_global = np.random.RandomState(1)

    direction_residuals = []
    normal_components = []
    eikonal_residuals = []
    grad_norms = []
    count = 0

    model.eval()
    for start in range(0, query_pts_ms.shape[0], batch_size):
        end = min(start + batch_size, query_pts_ms.shape[0])
        batch = _make_query_batch(
            query_pts_ms[start:end], pts_ms, kdtree, train_opt, rng, rng_global, device)
        q_pts = batch['imp_surf_query_point_ms'].detach().requires_grad_(True)
        batch['imp_surf_query_point_ms'] = q_pts

        q_pred = post_process_medial(forward_q_with_query_grad(model, batch))
        q_grad = torch.autograd.grad(
            q_pred.sum(), q_pts, create_graph=False, allow_unused=True)[0]
        if q_grad is None:
            q_grad = torch.zeros_like(q_pts)

        if sdf_sampler is not None:
            sdf_vals, agrad = sdf_sampler(q_pts)
        elif mesh_gt_sdf is None:
            sdf_vals, agrad, _, _, _ = model.with_mf_grad(batch, train_opt)
        else:
            sdf_vals, agrad = gt_sdf_and_gradient(mesh_gt_sdf, q_pts, device=device)

        mask = torch.isfinite(sdf_vals)
        if inside_only:
            mask = mask & (sdf_vals < 0.0)
        if sdf_valid_mask_sampler is not None:
            valid_mask = sdf_valid_mask_sampler(q_pts).to(device=q_pts.device, dtype=torch.bool)
            mask = mask & valid_mask
        if not torch.any(mask):
            continue

        q_grad_sel = q_grad[mask]
        sdf_vals_sel = sdf_vals[mask]
        agrad_sel = agrad[mask]
        sdf_norm = torch.linalg.norm(agrad_sel, dim=-1, keepdim=True).clamp(min=1e-8)
        sdf_unit = agrad_sel / sdf_norm
        q_normal = torch.sum(q_grad_sel * sdf_unit, dim=-1)
        sdf_sign = torch.where(
            sdf_vals_sel >= 0.0, torch.ones_like(sdf_vals_sel), -torch.ones_like(sdf_vals_sel))
        target = -sdf_sign * sdf_norm.squeeze(-1)
        q_grad_norm = torch.linalg.norm(q_grad_sel, dim=-1)
        direction_residuals.append((q_normal - target).detach().cpu())
        normal_components.append(q_normal.detach().cpu())
        eikonal_residuals.append((q_grad_norm - 1.0).detach().cpu())
        grad_norms.append(q_grad_norm.detach().cpu())
        count += int(mask.sum().detach().cpu())

    if count == 0:
        return {
            "q_grad_count": 0,
            "q_direction_abs_mean": None,
            "q_direction_rms": None,
            "q_objective_rms": None,
            "q_normal_mean": None,
            "q_eikonal_abs_mean": None,
            "q_eikonal_rms": None,
            "q_grad_norm_mean": None,
        }

    direction_all = torch.cat(direction_residuals)
    normal_all = torch.cat(normal_components)
    eikonal_all = torch.cat(eikonal_residuals)
    grad_norm_all = torch.cat(grad_norms)
    direction_rms = torch.sqrt(torch.mean(direction_all ** 2))
    eikonal_rms = torch.sqrt(torch.mean(eikonal_all ** 2))
    return {
        "q_grad_count": count,
        "q_direction_abs_mean": float(torch.mean(torch.abs(direction_all))),
        "q_direction_rms": float(direction_rms),
        "q_objective_rms": float(torch.sqrt(direction_rms ** 2 + eikonal_rms ** 2)),
        "q_normal_mean": float(torch.mean(normal_all)),
        "q_eikonal_abs_mean": float(torch.mean(torch.abs(eikonal_all))),
        "q_eikonal_rms": float(eikonal_rms),
        "q_grad_norm_mean": float(torch.mean(grad_norm_all)),
    }


def project_queries_to_medial_surface_from_q(
        model, train_opt, query_pts_ms, pts_ms, device, batch_size=64,
        mesh_gt_sdf=None, inside_only=True, score_percentile=None,
        max_points=None):
    """Project volume samples to medial centers when the head predicts Q-MDF."""
    import scipy.spatial as spatial

    query_pts_ms = np.asarray(query_pts_ms, dtype=np.float32)
    pts_ms = np.asarray(pts_ms, dtype=np.float32)
    kdtree = spatial.cKDTree(pts_ms)
    rng = np.random.RandomState(0)
    rng_global = np.random.RandomState(1)

    centers_all = []
    phi_all = []
    medial_all = []
    model.eval()
    for start in range(0, query_pts_ms.shape[0], batch_size):
        end = min(start + batch_size, query_pts_ms.shape[0])
        batch = _make_query_batch(
            query_pts_ms[start:end], pts_ms, kdtree, train_opt, rng, rng_global, device)
        q_pts = batch['imp_surf_query_point_ms'].detach().requires_grad_(True)
        batch['imp_surf_query_point_ms'] = q_pts
        q_pred = post_process_medial(model.forward_medial(batch))
        if mesh_gt_sdf is None:
            sdf_vals, agrad, _, _, _ = model.with_mf_grad(batch, train_opt)
        else:
            sdf_vals, agrad = gt_sdf_and_gradient(mesh_gt_sdf, q_pts, device=device)
        mf = q_to_medial(q_pred, sdf_vals)
        centers = project_to_medial_spoke(q_pts, sdf_vals, agrad, mf)
        centers_all.append(centers.detach().cpu().numpy())
        phi_all.append(sdf_vals.detach().cpu().numpy())
        medial_all.append(mf.detach().cpu().numpy())

    centers = np.concatenate(centers_all, axis=0).astype(np.float32)
    phi = np.concatenate(phi_all, axis=0).astype(np.float32)
    medial = np.concatenate(medial_all, axis=0).astype(np.float32)
    mask = np.isfinite(centers).all(axis=1) & np.isfinite(phi) & np.isfinite(medial)
    if inside_only:
        mask &= phi < 0.0
    if score_percentile is not None:
        score = np.abs(medial_level_score(phi, medial))
        threshold = np.percentile(score[mask], score_percentile) if np.any(mask) else 0.0
        mask &= score <= threshold
    selected = np.flatnonzero(mask)
    if max_points is not None and selected.shape[0] > max_points:
        selected = np.random.RandomState(0).choice(selected, max_points, replace=False)
    return centers[selected], phi[selected], medial[selected]


def evaluate_predicted_medial_surface_from_q(
        model, train_opt, eval_queries_ms, pts_ms, gt_medial_pts, device,
        out_dir=None, tag='eval', batch_size=64, mesh_gt_sdf=None,
        score_percentile=None, max_points=None):
    """Evaluate projected medial centers for a Q-MDF head."""
    centers, phi, medial = project_queries_to_medial_surface_from_q(
        model, train_opt, eval_queries_ms, pts_ms, device, batch_size=batch_size,
        mesh_gt_sdf=mesh_gt_sdf, inside_only=True, score_percentile=score_percentile,
        max_points=max_points)
    ply_file = None
    if out_dir is not None:
        from source.base import point_cloud
        import os

        os.makedirs(out_dir, exist_ok=True)
        ply_file = os.path.join(out_dir, 'pred_medial_{}.ply'.format(tag))
        point_cloud.write_ply(ply_file, centers)
    metrics = chamfer_l2_points(centers, gt_medial_pts) if len(centers) and len(gt_medial_pts) else {
        'chamfer_l2': np.inf, 'pred_to_gt': np.inf, 'gt_to_pred': np.inf}
    score = medial_level_score(phi, medial) if len(phi) else np.array([])
    metrics.update({
        'count': int(len(centers)),
        'inside_eval_count': int(len(phi)),
        'mean_abs_score': float(np.mean(np.abs(score))) if len(score) else None,
        'ply_file': ply_file,
        'medial_min': float(medial.min()) if len(medial) else None,
        'medial_max': float(medial.max()) if len(medial) else None,
    })
    return metrics


def evaluate_medial_field_on_gt_axis_from_q(
        model, train_opt, gt_medial_pts, gt_medial_radii, pts_ms, device,
        batch_size=64, max_points=None, rng=None, epsilon=0.01):
    """Evaluate reconstructed M on GT-axis samples for a Q-MDF head."""
    gt_medial_pts = np.asarray(gt_medial_pts, dtype=np.float32)
    gt_medial_radii = np.asarray(gt_medial_radii, dtype=np.float32)
    if gt_medial_pts.shape[0] == 0:
        return medial_axis_field_metrics(
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            epsilon=epsilon)

    selected = np.arange(gt_medial_pts.shape[0])
    if max_points is not None and selected.shape[0] > max_points:
        rng = np.random.RandomState(0) if rng is None else rng
        selected = rng.choice(selected, max_points, replace=False)

    q_pred = predict_medial_on_queries(
        model, train_opt, gt_medial_pts[selected], pts_ms, device,
        batch_size=batch_size)
    med_pred = q_pred + gt_medial_radii[selected]
    return medial_axis_field_metrics(med_pred, gt_medial_radii[selected], epsilon=epsilon)


def select_predicted_medial_points(
        query_pts_ms, phi, medial, inside_only=True, score_percentile=2.0,
        max_points=20000, rng=None):
    """
    Select grid/query samples near the predicted medial sheet M(x)-|phi(x)|=0.

    This is a diagnostic extractor, not a loss. For denser output, use
    project_queries_to_medial_surface, which projects volume samples to DMF Eq. 5 centers.
    """
    query_pts_ms = np.asarray(query_pts_ms, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)
    medial = np.asarray(medial, dtype=np.float32)
    score = np.abs(medial_level_score(phi, medial))
    mask = np.isfinite(score)
    if inside_only:
        mask &= phi < 0.0
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    threshold = np.percentile(score[mask], score_percentile)
    selected = np.flatnonzero(mask & (score <= threshold))
    if max_points is not None and selected.shape[0] > max_points:
        rng = np.random.RandomState(0) if rng is None else rng
        selected = rng.choice(selected, max_points, replace=False)
    return query_pts_ms[selected], score[selected]


def project_queries_to_medial_surface(
        model, train_opt, query_pts_ms, pts_ms, device, batch_size=64,
        inside_only=True, score_percentile=None, max_points=None):
    """
    Project arbitrary volume samples to predicted medial centers using DMF Eq. 5.

    Returned points are predictions from the medial head and frozen SDF only. Ground-truth
    medial geometry is deliberately not used here.
    """
    import scipy.spatial as spatial

    query_pts_ms = np.asarray(query_pts_ms, dtype=np.float32)
    pts_ms = np.asarray(pts_ms, dtype=np.float32)
    kdtree = spatial.cKDTree(pts_ms)
    rng = np.random.RandomState(0)
    rng_global = np.random.RandomState(1)

    centers_all = []
    phi_all = []
    medial_all = []

    model.eval()
    for start in range(0, query_pts_ms.shape[0], batch_size):
        end = min(start + batch_size, query_pts_ms.shape[0])
        batch = _make_query_batch(
            query_pts_ms[start:end], pts_ms, kdtree, train_opt, rng, rng_global, device)
        sdf, agrad, _, mf, _ = model.with_mf_grad(batch, train_opt)
        centers = project_to_medial_spoke(
            batch['imp_surf_query_point_ms'], sdf, agrad, mf)
        centers_all.append(centers.detach().cpu().numpy())
        phi_all.append(sdf.detach().cpu().numpy())
        medial_all.append(mf.detach().cpu().numpy())

    centers = np.concatenate(centers_all, axis=0).astype(np.float32)
    phi = np.concatenate(phi_all, axis=0).astype(np.float32)
    medial = np.concatenate(medial_all, axis=0).astype(np.float32)
    mask = np.isfinite(centers).all(axis=1) & np.isfinite(phi) & np.isfinite(medial)
    if inside_only:
        mask &= phi < 0.0
    if score_percentile is not None:
        score = np.abs(medial_level_score(phi, medial))
        threshold = np.percentile(score[mask], score_percentile) if np.any(mask) else 0.0
        mask &= score <= threshold
    selected = np.flatnonzero(mask)
    if max_points is not None and selected.shape[0] > max_points:
        selected = np.random.RandomState(0).choice(selected, max_points, replace=False)
    return centers[selected], phi[selected], medial[selected]


def project_queries_to_medial_surface_gt_sdf(
        model, train_opt, mesh_gt, query_pts_ms, pts_ms, device, batch_size=64,
        inside_only=True, score_percentile=None, max_points=None):
    """
    Project volume samples to predicted medial centers using GT mesh SDF.

    This mirrors project_queries_to_medial_surface, but phi and grad phi come
    from the mesh instead of the frozen Points2Surf SDF head.
    """
    import scipy.spatial as spatial

    query_pts_ms = np.asarray(query_pts_ms, dtype=np.float32)
    pts_ms = np.asarray(pts_ms, dtype=np.float32)
    kdtree = spatial.cKDTree(pts_ms)
    rng = np.random.RandomState(0)
    rng_global = np.random.RandomState(1)

    centers_all = []
    phi_all = []
    medial_all = []

    model.eval()
    for start in range(0, query_pts_ms.shape[0], batch_size):
        end = min(start + batch_size, query_pts_ms.shape[0])
        batch = _make_query_batch(
            query_pts_ms[start:end], pts_ms, kdtree, train_opt, rng, rng_global, device)
        q = batch['imp_surf_query_point_ms'].detach().requires_grad_(True)
        batch['imp_surf_query_point_ms'] = q
        mf = post_process_medial(model.forward_medial(batch))
        sdf, agrad = gt_sdf_and_gradient(mesh_gt, q, device=device)
        centers = project_to_medial_spoke(q, sdf, agrad, mf)
        centers_all.append(centers.detach().cpu().numpy())
        phi_all.append(sdf.detach().cpu().numpy())
        medial_all.append(mf.detach().cpu().numpy())

    centers = np.concatenate(centers_all, axis=0).astype(np.float32)
    phi = np.concatenate(phi_all, axis=0).astype(np.float32)
    medial = np.concatenate(medial_all, axis=0).astype(np.float32)
    mask = np.isfinite(centers).all(axis=1) & np.isfinite(phi) & np.isfinite(medial)
    if inside_only:
        mask &= phi < 0.0
    if score_percentile is not None:
        score = np.abs(medial_level_score(phi, medial))
        threshold = np.percentile(score[mask], score_percentile) if np.any(mask) else 0.0
        mask &= score <= threshold
    selected = np.flatnonzero(mask)
    if max_points is not None and selected.shape[0] > max_points:
        selected = np.random.RandomState(0).choice(selected, max_points, replace=False)
    return centers[selected], phi[selected], medial[selected]


def approximate_gt_medial_surface_from_mesh(
        mesh, sample_count=20000, k=32, min_normal_dot=-0.75,
        min_spoke_alignment=0.65, min_radius=1e-3, max_points=20000, seed=0):
    """
    Approximate a mesh medial surface for evaluation only.

    The method samples pairs of surface points with opposing normals and keeps their
    midpoints when the connecting segment is close to both inward normals. It is a
    rough diagnostic target, not a training signal and not an exact MAT algorithm.
    """
    import scipy.spatial as spatial
    import trimesh.sample

    rng = np.random.RandomState(seed)
    surface_pts, face_ids = trimesh.sample.sample_surface(mesh, sample_count)
    surface_pts = surface_pts.astype(np.float32)
    normals = mesh.face_normals[face_ids].astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-8)

    kdtree = spatial.cKDTree(surface_pts)
    _, nn_ids = kdtree.query(surface_pts, k=min(k + 1, sample_count))

    medial_pts = []
    medial_radii = []
    for i in range(surface_pts.shape[0]):
        candidates = nn_ids[i, 1:]
        n_i = normals[i]
        p_i = surface_pts[i]
        n_j = normals[candidates]
        normal_dot = n_j @ n_i
        opposite = candidates[normal_dot < min_normal_dot]
        if opposite.shape[0] == 0:
            continue
        vec = surface_pts[opposite] - p_i
        dist = np.linalg.norm(vec, axis=1)
        valid = dist > (2.0 * min_radius)
        if not np.any(valid):
            continue
        opposite = opposite[valid]
        vec = vec[valid] / dist[valid, None]
        dist = dist[valid]
        align_i = vec @ (-n_i)
        align_j = np.sum(vec * normals[opposite], axis=1)
        valid = (align_i > min_spoke_alignment) & (align_j > min_spoke_alignment)
        if not np.any(valid):
            continue
        best = np.argmax(np.minimum(align_i[valid], align_j[valid]))
        j = opposite[valid][best]
        medial_pts.append((p_i + surface_pts[j]) * 0.5)
        medial_radii.append(np.linalg.norm(surface_pts[j] - p_i) * 0.5)

    if not medial_pts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    medial_pts = np.asarray(medial_pts, dtype=np.float32)
    medial_radii = np.asarray(medial_radii, dtype=np.float32)
    if max_points is not None and medial_pts.shape[0] > max_points:
        ids = rng.choice(medial_pts.shape[0], max_points, replace=False)
        medial_pts = medial_pts[ids]
        medial_radii = medial_radii[ids]
    return medial_pts, medial_radii


def approximate_medial_axis_voronoi_from_mesh(
        mesh, surface_sample_count=3000, min_radius=1e-4,
        max_points=30000, seed=0):
    """
    Approximate an interior medial-axis point cloud from mesh surface samples.

    Voronoi vertices of dense surface samples approximate centers of maximal
    balls.  We keep only finite vertices inside the mesh and use the GT SDF
    magnitude as an approximate radius.  This is a visualization helper, not a
    training target.
    """
    import scipy.spatial as spatial
    import trimesh.sample
    from source import sdf

    rng = np.random.RandomState(seed)
    surface_pts, _ = trimesh.sample.sample_surface(mesh, surface_sample_count)
    surface_pts = surface_pts.astype(np.float32)
    if mesh.vertices.shape[0]:
        surface_pts = np.concatenate((surface_pts, mesh.vertices[:, :3].astype(np.float32)), axis=0)

    vor = spatial.Voronoi(surface_pts)
    vertices = vor.vertices.astype(np.float32)
    finite = np.isfinite(vertices).all(axis=1)
    vertices = vertices[finite]
    if vertices.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    bounds = mesh.bounds.astype(np.float32)
    in_bounds = np.all((vertices >= bounds[0]) & (vertices <= bounds[1]), axis=1)
    vertices = vertices[in_bounds]
    if vertices.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # trimesh signed_distance is positive inside; keep interior centers.
    signed = sdf.get_signed_distance(mesh, vertices).astype(np.float32)
    keep = signed > min_radius
    vertices = vertices[keep]
    radii = signed[keep]

    if max_points is not None and vertices.shape[0] > max_points:
        ids = rng.choice(vertices.shape[0], max_points, replace=False)
        vertices = vertices[ids]
        radii = radii[ids]

    return vertices.astype(np.float32), radii.astype(np.float32)


def approximate_box_medial_axis_from_bounds(
        bounds, grid_resolution=96, tie_tol=None, min_radius=1e-4,
        max_points=30000, seed=0):
    """
    Approximate the exact medial axis of an axis-aligned box from its bounds.

    For an interior point in a box, the local feature size is the minimum
    distance to the six faces.  Points where two or more face distances tie for
    that minimum form a sampled approximation of the box medial sheets.  This
    is much less noisy than a Voronoi approximation for the box notebook.
    """
    rng = np.random.RandomState(seed)
    bounds = np.asarray(bounds, dtype=np.float32)
    mins = bounds[0]
    maxs = bounds[1]
    if tie_tol is None:
        tie_tol = float(np.max(maxs - mins) / max(grid_resolution - 1, 1) * 0.75)

    axes = [
        np.linspace(mins[d], maxs[d], grid_resolution, dtype=np.float32)
        for d in range(3)
    ]
    xx, yy, zz = np.meshgrid(*axes, indexing='ij')
    pts = np.stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)), axis=-1)

    face_dists = np.stack((
        pts[:, 0] - mins[0], maxs[0] - pts[:, 0],
        pts[:, 1] - mins[1], maxs[1] - pts[:, 1],
        pts[:, 2] - mins[2], maxs[2] - pts[:, 2],
    ), axis=-1)
    radii = np.min(face_dists, axis=1)
    interior = radii > min_radius
    tied = np.sum(np.abs(face_dists - radii[:, None]) <= tie_tol, axis=1) >= 2
    keep = interior & tied
    pts = pts[keep]
    radii = radii[keep]

    if max_points is not None and pts.shape[0] > max_points:
        ids = rng.choice(pts.shape[0], max_points, replace=False)
        pts = pts[ids]
        radii = radii[ids]

    return pts.astype(np.float32), radii.astype(np.float32)


def chamfer_l2_points(a, b):
    """Symmetric nearest-neighbor L2 distance between two point clouds."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return {
            "chamfer_l2": np.inf,
            "pred_to_gt": np.inf,
            "gt_to_pred": np.inf,
        }

    try:
        import scipy.spatial as spatial
        tree_a = spatial.cKDTree(a)
        tree_b = spatial.cKDTree(b)
        pred_to_gt = tree_b.query(a, k=1)[0]
        gt_to_pred = tree_a.query(b, k=1)[0]
    except ImportError:
        def nearest_dist(src, dst, chunk_size=4096):
            mins = []
            for start in range(0, src.shape[0], chunk_size):
                chunk = src[start:start + chunk_size]
                diff = chunk[:, None, :] - dst[None, :, :]
                mins.append(np.sqrt(np.min(np.sum(diff * diff, axis=-1), axis=1)))
            return np.concatenate(mins, axis=0)

        pred_to_gt = nearest_dist(a, b)
        gt_to_pred = nearest_dist(b, a)

    return {
        "chamfer_l2": float(np.mean(pred_to_gt ** 2) + np.mean(gt_to_pred ** 2)),
        "pred_to_gt": float(np.mean(pred_to_gt)),
        "gt_to_pred": float(np.mean(gt_to_pred)),
    }


def evaluate_predicted_medial_surface(
        model, train_opt, eval_queries_ms, pts_ms, gt_medial_pts, device,
        out_dir=None, tag='eval', batch_size=64, score_percentile=50.0,
        max_points=20000, mesh_gt_sdf=None):
    """
    Project eval samples to predicted medial centers and compare to diagnostic GT.

    This function is intentionally outside compute_medial_losses. It is for
    inspection, checkpoint selection, and plotting only; it does not provide a
    training signal.
    """
    if mesh_gt_sdf is None:
        centers, phi, medial = project_queries_to_medial_surface(
            model, train_opt, eval_queries_ms, pts_ms, device,
            batch_size=batch_size, inside_only=True,
            score_percentile=score_percentile, max_points=max_points)
    else:
        centers, phi, medial = project_queries_to_medial_surface_gt_sdf(
            model, train_opt, mesh_gt_sdf, eval_queries_ms, pts_ms, device,
            batch_size=batch_size, inside_only=True,
            score_percentile=score_percentile, max_points=max_points)
    ply_file = None
    if out_dir is not None:
        from source.base import point_cloud
        import os

        os.makedirs(out_dir, exist_ok=True)
        ply_file = os.path.join(out_dir, 'pred_medial_{}.ply'.format(tag))
        point_cloud.write_ply(ply_file, centers)

    metrics = chamfer_l2_points(centers, gt_medial_pts) if len(centers) and len(gt_medial_pts) else {
        'chamfer_l2': np.inf,
        'pred_to_gt': np.inf,
        'gt_to_pred': np.inf,
    }
    score = medial_level_score(phi, medial) if len(phi) else np.array([])
    summary = {
        'count': int(len(centers)),
        'inside_eval_count': int(len(phi)),
        'phi_min': float(phi.min()) if len(phi) else None,
        'phi_max': float(phi.max()) if len(phi) else None,
        'medial_min': float(medial.min()) if len(medial) else None,
        'medial_max': float(medial.max()) if len(medial) else None,
        'mean_abs_score': float(np.mean(np.abs(score))) if len(score) else None,
        'ply_file': ply_file,
    }
    summary.update(metrics)
    return summary


def write_medial_projection_image(surface_pts, pred_pts, gt_pts, out_file,
                                  size=900, margin=35):
    """
    Save XY/XZ/YZ orthographic projections for quick visual diagnosis.

    Colors: surface = light gray, diagnostic GT = blue, prediction = red.
    This is a visualization helper only; it is not used by any loss.
    """
    from PIL import Image, ImageDraw
    import os

    surface_pts = np.asarray(surface_pts, dtype=np.float32)
    pred_pts = np.asarray(pred_pts, dtype=np.float32)
    gt_pts = np.asarray(gt_pts, dtype=np.float32)
    all_pts = [p for p in (surface_pts, pred_pts, gt_pts) if p.size]
    if not all_pts:
        return

    pts_all = np.concatenate(all_pts, axis=0)
    mins = pts_all.min(axis=0)
    maxs = pts_all.max(axis=0)
    center = (mins + maxs) * 0.5
    scale = float(np.max(maxs - mins))
    if scale <= 0.0:
        scale = 1.0

    panels = [('XY', (0, 1)), ('XZ', (0, 2)), ('YZ', (1, 2))]
    panel_w = size // 3
    panel_h = size // 3
    image = Image.new('RGB', (panel_w * 3, panel_h), 'white')
    draw = ImageDraw.Draw(image)

    def project(points, axes, panel_index):
        if points.size == 0:
            return []
        normalized = (points[:, axes] - center[list(axes)]) / scale
        x = (normalized[:, 0] + 0.5) * (panel_w - 2 * margin) + panel_index * panel_w + margin
        y = (0.5 - normalized[:, 1]) * (panel_h - 2 * margin) + margin
        return np.stack([x, y], axis=1)

    def draw_points(points, axes, panel_index, color, radius):
        for x, y in project(points, axes, panel_index):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    for pi, (label, axes) in enumerate(panels):
        x0 = pi * panel_w
        draw.rectangle((x0, 0, x0 + panel_w - 1, panel_h - 1), outline=(220, 220, 220))
        draw.text((x0 + 8, 8), label, fill=(40, 40, 40))
        step = max(surface_pts.shape[0] // 5000, 1) if surface_pts.size else 1
        draw_points(surface_pts[::step], axes, pi, (210, 210, 210), 1)
        draw_points(gt_pts, axes, pi, (50, 100, 220), 2)
        draw_points(pred_pts, axes, pi, (220, 50, 50), 3)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    image.save(out_file)
