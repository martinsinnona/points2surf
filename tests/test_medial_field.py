import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from source import medial_field, points_to_surf_medial


class _LinearQueryBackbone(nn.Module):
    def encode_bottleneck(self, batch):
        return batch["imp_surf_query_point_ms"][:, :1]


class _LinearMedialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _LinearQueryBackbone()
        self.medial_head = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.medial_head.weight, 2.0)


class _ConstantRadiusBackbone(nn.Module):
    net_size_max = 8

    def encode_bottleneck(self, batch):
        return torch.ones(
            (batch["imp_surf_query_point_ms"].shape[0], 1),
            dtype=batch["imp_surf_query_point_ms"].dtype,
            device=batch["imp_surf_query_point_ms"].device)


class _ConstantRadiusMedialModel(nn.Module):
    def __init__(self, initial_raw=0.1):
        super().__init__()
        self.backbone = _ConstantRadiusBackbone()
        self.medial_head = nn.Linear(1, 1)
        nn.init.constant_(self.medial_head.weight, initial_raw)
        nn.init.constant_(self.medial_head.bias, initial_raw)


class _CoordFeatureMedialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _ConstantRadiusBackbone()
        self.medial_head = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(inplace=False),
            nn.Linear(4, 1),
        )

    def medial_features(self, bottleneck, batch):
        if getattr(self, "detach_backbone_for_medial_grad", False):
            bottleneck = bottleneck.detach()
        return torch.cat((bottleneck, batch["imp_surf_query_point_ms"]), dim=1)


class _QuadraticQueryBackbone(nn.Module):
    def encode_bottleneck(self, batch):
        query = batch["imp_surf_query_point_ms"]
        return query[:, :1] ** 2


class _QuadraticCoordFeatureModel(nn.Module):
    def __init__(self, detach_backbone_for_medial_grad=False):
        super().__init__()
        self.backbone = _QuadraticQueryBackbone()
        self.detach_backbone_for_medial_grad = detach_backbone_for_medial_grad
        self.medial_head = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            self.medial_head.weight.zero_()
            self.medial_head.weight[0, 0] = 1.0
            self.medial_head.weight[0, 1] = 1.0

    def medial_features(self, bottleneck, batch):
        if self.detach_backbone_for_medial_grad:
            bottleneck = bottleneck.detach()
        return torch.cat((bottleneck, batch["imp_surf_query_point_ms"]), dim=1)


class _TwoBranchQHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.medial_head = nn.Linear(2, 1)
        self.query_residual_head = nn.Linear(3, 1)


class _PredictedSdfQModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.encode_bottleneck = lambda batch: batch["imp_surf_query_point_ms"]
        self.backbone.fc4 = nn.Linear(3, 1, bias=False)
        self.medial_head = nn.Linear(3, 1)
        with torch.no_grad():
            self.backbone.fc4.weight.zero_()
            self.backbone.fc4.weight[0, 0] = 1.0
            self.medial_head.weight.zero_()
            self.medial_head.bias.zero_()

    def medial_features(self, bottleneck, batch):
        return bottleneck

    def medial_raw_from_features(self, features, batch):
        return self.medial_head(features)


def _unit_sphere_sdf_sampler(query):
    radius = torch.linalg.norm(query, dim=-1).clamp(min=1e-8)
    phi_const = radius - 1.0
    grad = query / radius.unsqueeze(-1)
    phi = medial_field.attach_sdf_gradient(phi_const, query, grad)
    return phi, grad


class MedialFieldTest(unittest.TestCase):
    def test_process_sdf_prediction_flips_trimesh_convention_to_inside_negative(self):
        train_opt = SimpleNamespace(
            outputs=["imp_surf_magnitude", "imp_surf_sign"],
            patch_radius=0.05,
        )
        patch_radius = torch.tensor([0.05, 0.05])
        # Positive sign logits are the dataset/trimesh convention: inside positive.
        sdf_raw = torch.tensor([[1.0, 2.0], [1.0, -2.0]])

        phi = medial_field.process_sdf_prediction(sdf_raw, train_opt, patch_radius)

        self.assertLess(float(phi[0]), 0.0)
        self.assertGreater(float(phi[1]), 0.0)

    def test_forward_medial_with_query_grad_keeps_query_gradient(self):
        model = _LinearMedialModel()
        query = torch.tensor([[3.0, 0.0, 0.0]], requires_grad=True)
        batch = {"imp_surf_query_point_ms": query}

        medial = medial_field.forward_medial_with_query_grad(model, batch)
        grad = torch.autograd.grad(medial.sum(), query)[0]

        self.assertTrue(torch.allclose(grad, torch.tensor([[2.0, 0.0, 0.0]])))

    def test_query_coord_medial_model_can_depend_directly_on_query(self):
        model = _CoordFeatureMedialModel()
        with torch.no_grad():
            model.medial_head[0].weight.zero_()
            model.medial_head[0].bias.fill_(1.0)
            model.medial_head[0].weight[:, -3:] = 1.0
            model.medial_head[2].weight.fill_(1.0)
            model.medial_head[2].bias.zero_()
        query = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
        batch = {"imp_surf_query_point_ms": query}

        medial = medial_field.forward_medial_with_query_grad(model, batch)
        grad = torch.autograd.grad(medial.sum(), query, allow_unused=True)[0]

        self.assertIsNotNone(grad)
        self.assertEqual(grad.shape, query.shape)
        self.assertGreater(float(torch.linalg.norm(grad)), 0.0)

    def test_detach_backbone_for_medial_grad_keeps_query_coord_gradient_only(self):
        query = torch.tensor([[2.0, 0.0, 0.0]], requires_grad=True)
        batch = {"imp_surf_query_point_ms": query}

        full_model = _QuadraticCoordFeatureModel(detach_backbone_for_medial_grad=False)
        detached_model = _QuadraticCoordFeatureModel(detach_backbone_for_medial_grad=True)

        full_q = medial_field.forward_q_with_query_grad(full_model, batch)
        detached_q = medial_field.forward_q_with_query_grad(detached_model, batch)
        full_grad = torch.autograd.grad(full_q.sum(), query, retain_graph=True)[0]
        detached_grad = torch.autograd.grad(detached_q.sum(), query)[0]

        self.assertTrue(torch.allclose(full_grad, torch.tensor([[5.0, 0.0, 0.0]])))
        self.assertTrue(torch.allclose(detached_grad, torch.tensor([[1.0, 0.0, 0.0]])))

    def test_q_mdf_head_state_dict_preserves_query_residual_branch(self):
        model = _TwoBranchQHeadModel()
        with torch.no_grad():
            model.medial_head.weight.fill_(1.0)
            model.query_residual_head.weight.fill_(2.0)
        state = medial_field.q_mdf_head_state_dict(model)

        restored = _TwoBranchQHeadModel()
        medial_field.load_q_mdf_head_state_dict(restored, state)

        self.assertTrue(torch.allclose(restored.medial_head.weight, model.medial_head.weight))
        self.assertTrue(torch.allclose(
            restored.query_residual_head.weight, model.query_residual_head.weight))

    def test_fourier_query_residual_head_keeps_query_gradient(self):
        model = points_to_surf_medial.PointsToSurfMedialModel(
            _ConstantRadiusBackbone(),
            hidden_mult=4.0,
            use_query_coords=True,
            detach_backbone_for_medial_grad=True,
            use_query_residual_head=True,
            query_residual_encoding="fourier",
            query_fourier_num_freqs=3,
        )
        query = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
        batch = {"imp_surf_query_point_ms": query}

        medial = medial_field.forward_q_with_query_grad(model, batch)
        grad = torch.autograd.grad(medial.sum(), query, allow_unused=True)[0]

        self.assertIsNotNone(grad)
        self.assertEqual(grad.shape, query.shape)
        self.assertGreater(float(torch.linalg.norm(grad)), 0.0)

    def test_attach_sdf_gradient_uses_sampled_value_and_supplied_backward_gradient(self):
        query = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        phi_const = torch.tensor([7.0])
        supplied_grad = torch.tensor([[0.5, -1.0, 2.0]])

        phi = medial_field.attach_sdf_gradient(phi_const, query, supplied_grad)
        grad = torch.autograd.grad(phi.sum(), query)[0]

        self.assertTrue(torch.allclose(phi.detach(), phi_const))
        self.assertTrue(torch.allclose(grad, supplied_grad))

    def test_box_sdf_gradient_valid_mask_excludes_near_face_ties(self):
        bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        query = torch.tensor([
            [0.1, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.1, 0.12, 0.5],
        ])

        mask = medial_field.box_sdf_gradient_valid_mask(bounds, query, min_face_gap=0.1)

        self.assertTrue(bool(mask[0]))
        self.assertFalse(bool(mask[1]))
        self.assertFalse(bool(mask[2]))

    def test_project_to_medial_spoke_uses_unsigned_sdf_gradient(self):
        query = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        sdf = torch.tensor([1.0, -1.0])
        sdf_grad = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        medial = torch.tensor([2.0, 2.0])

        projected = medial_field.project_to_medial_spoke(query, sdf, sdf_grad, medial)

        expected = torch.tensor([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        self.assertTrue(torch.allclose(projected, expected))

    def test_batch_at_queries_recenters_patch_coordinates_with_gradients(self):
        old_query = torch.tensor([[1.0, 0.0, 0.0]])
        patch_radius = torch.tensor([2.0])
        patch_pts_ps = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        new_query = torch.tensor([[2.0, 0.0, 0.0]], requires_grad=True)
        batch = {
            "imp_surf_query_point_ms": old_query,
            "imp_surf_query_point_ps": torch.zeros_like(old_query),
            "patch_pts_ps": patch_pts_ps,
            "patch_radius_ms": patch_radius,
        }

        shifted = medial_field.batch_at_queries(batch, new_query)

        expected_patch = torch.tensor([[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]])
        self.assertTrue(torch.allclose(shifted["patch_pts_ps"], expected_patch))
        self.assertTrue(torch.allclose(shifted["imp_surf_query_point_ps"], torch.zeros_like(new_query)))

        shifted["patch_pts_ps"].sum().backward()
        self.assertTrue(torch.allclose(new_query.grad, torch.tensor([[-1.0, -1.0, -1.0]])))

    def test_select_predicted_medial_points_prefers_low_inside_score(self):
        query = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]).numpy()
        phi = torch.tensor([-0.5, -0.5, -0.5, 0.5]).numpy()
        medial = torch.tensor([0.50, 0.55, 1.50, 0.50]).numpy()

        points, score = medial_field.select_predicted_medial_points(
            query, phi, medial, inside_only=True, score_percentile=50.0)

        self.assertEqual(points.shape[0], 2)
        self.assertTrue((score <= 0.05 + 1e-6).all())

    def test_medial_level_score_is_m_minus_unsigned_sdf(self):
        phi = torch.tensor([-2.0, 0.5, 1.0]).numpy()
        medial = torch.tensor([2.0, 1.5, 0.25]).numpy()

        score = medial_field.medial_level_score(phi, medial)

        self.assertTrue(torch.allclose(torch.from_numpy(score), torch.tensor([0.0, 1.0, -0.75])))

    def test_q_mdf_field_is_nonnegative_medial_minus_unsigned_sdf(self):
        phi = torch.tensor([-1.5, 0.25, 2.0]).numpy()
        medial = torch.tensor([2.0, 1.0, 1.5]).numpy()

        q_mdf = medial_field.q_mdf_field(phi, medial)

        self.assertTrue(torch.allclose(torch.from_numpy(q_mdf), torch.tensor([0.5, 0.75, 0.0])))

    def test_q_mdf_level_set_metrics_compares_selected_points_to_gt(self):
        query = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]).numpy()
        phi = torch.tensor([-0.5, -0.5, -0.5]).numpy()
        medial = torch.tensor([0.50, 0.52, 0.80]).numpy()
        gt = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).numpy()

        metrics, q_pts = medial_field.q_mdf_level_set_metrics(
            query, phi, medial, gt, epsilon=0.03)

        self.assertEqual(q_pts.shape[0], 2)
        self.assertEqual(metrics["q_mdf_level_set_count"], 2)
        self.assertAlmostEqual(metrics["chamfer_l2"], 0.0)
        self.assertAlmostEqual(metrics["q_mdf_valid_band_fraction"], 2.0 / 3.0)

    def test_q_mdf_level_set_metrics_rejects_clamped_underpredictions(self):
        query = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]).numpy()
        phi = torch.tensor([-1.0, -1.0]).numpy()
        medial = torch.tensor([0.1, 1.0]).numpy()
        gt = torch.tensor([[1.0, 0.0, 0.0]]).numpy()

        metrics, q_pts = medial_field.q_mdf_level_set_metrics(
            query, phi, medial, gt, epsilon=0.03)

        self.assertEqual(q_pts.shape[0], 1)
        self.assertAlmostEqual(float(q_pts[0, 0]), 1.0)
        self.assertAlmostEqual(metrics["q_mdf_underpred_fraction"], 0.5)

    def test_box_medial_axis_from_bounds_returns_interior_tie_points(self):
        bounds = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=np.float32)

        axis_pts, radii = medial_field.approximate_box_medial_axis_from_bounds(
            bounds, grid_resolution=9, min_radius=0.05)

        self.assertGreater(axis_pts.shape[0], 0)
        self.assertTrue(np.all(radii > 0.05))
        face_dists = np.stack((
            axis_pts[:, 0] - bounds[0, 0], bounds[1, 0] - axis_pts[:, 0],
            axis_pts[:, 1] - bounds[0, 1], bounds[1, 1] - axis_pts[:, 1],
            axis_pts[:, 2] - bounds[0, 2], bounds[1, 2] - axis_pts[:, 2],
        ), axis=-1)
        min_dist = face_dists.min(axis=1)
        tied = np.sum(np.abs(face_dists - min_dist[:, None]) <= (1.0 / 8.0), axis=1)
        self.assertTrue(np.all(tied >= 2))

    def test_box_sdf_and_gradient_inside_negative_convention(self):
        bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        query = torch.tensor([
            [0.2, 0.5, 0.5],
            [0.8, 0.5, 0.5],
            [0.5, 0.3, 0.5],
        ], dtype=torch.float32, requires_grad=True)

        phi, grad = medial_field.box_sdf_and_gradient(bounds, query)
        auto_grad = torch.autograd.grad(phi.sum(), query)[0]

        self.assertTrue(torch.allclose(phi.detach(), torch.tensor([-0.2, -0.2, -0.3])))
        self.assertTrue(torch.allclose(grad[0], torch.tensor([-1.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(grad[1], torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(grad[2], torch.tensor([0.0, -1.0, 0.0])))
        self.assertTrue(torch.allclose(auto_grad, grad))

    def test_box_sdf_and_gradient_outside_positive_convention(self):
        bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        query = torch.tensor([
            [-0.2, 0.5, 0.5],
            [1.2, 0.5, 0.5],
            [1.0, 1.3, 0.5],
        ], dtype=torch.float32, requires_grad=True)

        phi, grad = medial_field.box_sdf_and_gradient(bounds, query)
        auto_grad = torch.autograd.grad(phi.sum(), query)[0]

        self.assertTrue(torch.allclose(phi.detach(), torch.tensor([0.2, 0.2, 0.3])))
        self.assertTrue(torch.allclose(grad[0], torch.tensor([-1.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(grad[1], torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(grad[2], torch.tensor([0.0, 1.0, 0.0])))
        self.assertTrue(torch.allclose(auto_grad, grad))

    def test_medial_axis_field_metrics_reports_radius_residuals(self):
        medial = np.array([0.45, 0.55, 0.70], dtype=np.float32)
        radii = np.array([0.50, 0.50, 0.50], dtype=np.float32)

        metrics = medial_field.medial_axis_field_metrics(medial, radii, epsilon=0.06)

        self.assertEqual(metrics["gt_axis_field_count"], 3)
        self.assertAlmostEqual(metrics["gt_axis_residual_mae"], 0.1, places=6)
        self.assertAlmostEqual(metrics["gt_axis_residual_bias"], 0.06666667, places=6)
        self.assertAlmostEqual(metrics["gt_axis_underpred_fraction"], 0.0)
        self.assertAlmostEqual(metrics["gt_axis_valid_fraction"], 2.0 / 3.0)

    def test_orthogonality_uses_signed_sdf_gradient(self):
        # DMF Eq. 8 uses grad phi, while Eq. 5 projection uses grad |phi|.
        mf_grad = torch.tensor([[1.0, 0.0, 0.0]])
        sdf_grad = torch.tensor([[1.0, 0.0, 0.0]])
        sdf = torch.tensor([-1.0])

        sdf_abs_grad, _ = medial_field.sdf_abs_grad(sdf, sdf_grad)
        signed_dot = torch.sum(mf_grad * sdf_grad, dim=-1)
        unsigned_dot = torch.sum(mf_grad * sdf_abs_grad, dim=-1)

        self.assertTrue(torch.allclose(signed_dot, torch.tensor([1.0])))
        self.assertTrue(torch.allclose(unsigned_dot, torch.tensor([-1.0])))

    def test_orthogonality_loss_penalizes_normal_component(self):
        sdf_grad = torch.tensor([[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        aligned_mf_grad = torch.tensor([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
        orthogonal_mf_grad = torch.tensor([[0.0, 2.0, 0.0], [0.0, -3.0, 0.0]])
        zero_mf_grad = torch.zeros_like(aligned_mf_grad)

        aligned_loss = medial_field.orthogonality_loss(aligned_mf_grad, sdf_grad)
        orthogonal_loss = medial_field.orthogonality_loss(orthogonal_mf_grad, sdf_grad)
        zero_loss = medial_field.orthogonality_loss(zero_mf_grad, sdf_grad)

        self.assertTrue(torch.allclose(aligned_loss, torch.tensor(0.025)))
        self.assertTrue(torch.allclose(orthogonal_loss, torch.tensor(0.0)))
        self.assertTrue(torch.allclose(zero_loss, torch.tensor(0.0)))

    def test_orthogonality_loss_has_gradient_when_aligned(self):
        sdf_grad = torch.tensor([[1.0, 0.0, 0.0]])
        mf_grad = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)

        loss = medial_field.orthogonality_loss(mf_grad, sdf_grad)
        loss.backward()

        self.assertGreater(float(mf_grad.grad[0, 0]), 0.0)
        self.assertTrue(torch.allclose(mf_grad.grad[0, 1:], torch.zeros(2)))

    def test_q_mdf_direction_loss_learns_outside_negative_sdf_sign(self):
        slope = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.SGD([slope], lr=0.2)
        query_base = torch.tensor([[0.4, 0.1, 0.0]], dtype=torch.float32)
        sdf_grad = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        losses = []

        for _ in range(40):
            optimizer.zero_grad()
            query = query_base.detach().clone().requires_grad_(True)
            q = slope * query[:, 0]
            phi = medial_field.attach_sdf_gradient(query[:, 0].detach(), query, sdf_grad)
            q_grad = torch.autograd.grad(q.sum(), query, create_graph=True)[0]
            loss = medial_field.q_mdf_direction_loss(q_grad, phi, sdf_grad)
            losses.append(float(loss.detach()))
            loss.backward()
            optimizer.step()

        self.assertLess(losses[-1], losses[0] * 0.01)
        self.assertAlmostEqual(float(slope.detach()), -1.0, places=3)

    def test_q_mdf_direction_loss_learns_inside_negative_sdf_sign(self):
        slope = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.SGD([slope], lr=0.2)
        query_base = torch.tensor([[0.4, 0.1, 0.0]], dtype=torch.float32)
        sdf_grad = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        losses = []

        for _ in range(40):
            optimizer.zero_grad()
            query = query_base.detach().clone().requires_grad_(True)
            q = slope * query[:, 0]
            phi = medial_field.attach_sdf_gradient(-query[:, 0].detach(), query, sdf_grad)
            q_grad = torch.autograd.grad(q.sum(), query, create_graph=True)[0]
            loss = medial_field.q_mdf_direction_loss(q_grad, phi, sdf_grad)
            losses.append(float(loss.detach()))
            loss.backward()
            optimizer.step()

        self.assertLess(losses[-1], losses[0] * 0.01)
        self.assertAlmostEqual(float(slope.detach()), 1.0, places=3)

    def test_q_eikonal_loss_penalizes_non_unit_q_gradient(self):
        unit_q_grad = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        non_unit_q_grad = torch.tensor([[2.0, 0.0, 0.0], [0.0, -0.5, 0.0]])
        zero_q_grad = torch.zeros_like(unit_q_grad)

        unit_loss = medial_field.q_eikonal_loss(unit_q_grad)
        non_unit_loss = medial_field.q_eikonal_loss(non_unit_q_grad)
        zero_loss = medial_field.q_eikonal_loss(zero_q_grad)

        self.assertTrue(torch.allclose(unit_loss, torch.tensor(0.0)))
        self.assertTrue(torch.allclose(non_unit_loss, torch.tensor(0.625)))
        self.assertTrue(torch.allclose(zero_loss, torch.tensor(1.0)))

    def test_eikonal_loss_skips_detached_duplicate_gradient(self):
        agrad_norm = torch.tensor([[2.0], [0.5]], requires_grad=True)
        pgrad = torch.zeros((2, 3))
        pgrad_norm = agrad_norm.detach()

        loss = medial_field.eikonal_loss_from_gradients(agrad_norm, pgrad_norm, pgrad)

        self.assertTrue(torch.allclose(loss, torch.tensor(0.625)))

    def test_predicted_sdf_q_mdf_loss_uses_grad_m_orthogonality(self):
        model = _PredictedSdfQModel()
        train_opt = SimpleNamespace(outputs=["imp_surf"], patch_radius=1.0)
        batch = {
            "imp_surf_query_point_ms": torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32),
            "patch_radius_ms": torch.tensor([1.0], dtype=torch.float32),
            "patch_pts_ps": torch.zeros((1, 3, 3), dtype=torch.float32),
            "imp_surf_ms": torch.tensor([[0.2]], dtype=torch.float32),
        }
        weights = medial_field.get_default_weights()
        for key in weights:
            weights[key] = 0.0
        weights.update({
            "volume_sdf": 1.0,
            "zero_set_sdf": 1.0,
            "zero_set_grad": 1.0,
            "orthogonality": 1.0,
        })

        loss, parts = medial_field.compute_q_mdf_losses(model, batch, train_opt, weights=weights)
        loss.backward()

        self.assertIn("Volume SDF", parts)
        self.assertIn("Zero Set SDF", parts)
        self.assertIn("Zero Set Gradient", parts)
        self.assertIn("Orthogonality", parts)
        self.assertNotIn("Surface Regularizer", parts)
        self.assertGreater(float(model.medial_head.weight.grad.norm()), 0.0)

    def test_predicted_sdf_q_mdf_loss_applies_q_terms_inside_only(self):
        model = _PredictedSdfQModel()
        with torch.no_grad():
            model.medial_head.weight.zero_()
            model.medial_head.weight[0, 0] = 1.0
            model.medial_head.bias.zero_()
        train_opt = SimpleNamespace(outputs=["imp_surf"], patch_radius=1.0)
        weights = medial_field.get_default_weights()
        for key in weights:
            weights[key] = 0.0
        weights["orthogonality"] = 1.0

        inside_batch = {
            "imp_surf_query_point_ms": torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32),
            "patch_radius_ms": torch.tensor([1.0], dtype=torch.float32),
            "patch_pts_ps": torch.zeros((1, 3, 3), dtype=torch.float32),
            "imp_surf_ms": torch.tensor([[0.2]], dtype=torch.float32),
        }
        mixed_batch = {
            "imp_surf_query_point_ms": torch.tensor([
                [0.5, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
            ], dtype=torch.float32),
            "patch_radius_ms": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "patch_pts_ps": torch.zeros((2, 3, 3), dtype=torch.float32),
            "imp_surf_ms": torch.tensor([[0.2], [-0.2]], dtype=torch.float32),
        }

        _, inside_parts = medial_field.compute_q_mdf_losses(
            model, inside_batch, train_opt, weights=weights)
        _, mixed_parts = medial_field.compute_q_mdf_losses(
            model, mixed_batch, train_opt, weights=weights)

        self.assertTrue(torch.allclose(
            inside_parts["Orthogonality"], mixed_parts["Orthogonality"]))

    def test_predicted_sdf_q_mdf_inscription_uses_matching_inside_batch_rows(self):
        model = _PredictedSdfQModel()
        train_opt = SimpleNamespace(outputs=["imp_surf"], patch_radius=1.0)
        weights = medial_field.get_default_weights()
        for key in weights:
            weights[key] = 0.0
        weights["inscription"] = 1.0
        batch = {
            "imp_surf_query_point_ms": torch.tensor([
                [0.5, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
            ], dtype=torch.float32),
            "patch_radius_ms": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "patch_pts_ps": torch.zeros((2, 3, 3), dtype=torch.float32),
            "imp_surf_ms": torch.tensor([[0.2], [-0.2]], dtype=torch.float32),
        }

        _, parts = medial_field.compute_q_mdf_losses(
            model, batch, train_opt, weights=weights)

        self.assertIn("Inscription", parts)
        self.assertTrue(torch.isfinite(parts["Inscription"]))

    def test_q_mdf_orthogonality_only_skips_inscription_sdf_sample(self):
        model = _CoordFeatureMedialModel()
        batch = {
            "imp_surf_query_point_ms": torch.tensor([[0.4, 0.1, 0.0]], dtype=torch.float32),
        }
        calls = []

        def sdf_sampler(query):
            calls.append(query.detach().clone())
            grad = torch.zeros_like(query)
            grad[:, 0] = 1.0
            phi = medial_field.attach_sdf_gradient(-query[:, 0].detach(), query, grad)
            return phi, grad

        weights = medial_field.get_default_weights()
        for key in weights:
            weights[key] = 0.0
        weights["orthogonality"] = 1.0

        medial_field.compute_q_mdf_losses_with_sdf_sampler(
            model, batch, sdf_sampler, weights=weights)

        self.assertEqual(len(calls), 1)

    def test_external_sdf_medial_loss_converges_on_unit_sphere_center_samples(self):
        model = _ConstantRadiusMedialModel(initial_raw=0.05)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        batch = {
            "imp_surf_query_point_ms": torch.tensor([
                [0.00, 0.00, 0.00],
                [0.20, 0.00, 0.00],
                [-0.20, 0.00, 0.00],
                [0.00, 0.20, 0.00],
                [0.00, -0.20, 0.00],
            ], dtype=torch.float32)
        }
        weights = medial_field.get_default_weights()
        weights.update({
            "maximality": 1.0,
            "inscription": 10.0,
            "orthogonality": 0.0,
            "eikonal": 0.0,
            "surface_reg": 0.0,
        })

        losses = []
        for _ in range(80):
            optimizer.zero_grad()
            loss, _ = medial_field.compute_medial_losses_with_sdf_sampler(
                model, batch, _unit_sphere_sdf_sampler, weights=weights)
            losses.append(float(loss.detach()))
            loss.backward()
            optimizer.step()

        self.assertLess(losses[-1], losses[0] * 0.2)


if __name__ == "__main__":
    unittest.main()
