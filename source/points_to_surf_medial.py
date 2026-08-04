import torch
import torch.nn as nn
from typing import Any


class FourierQueryResidualHead(nn.Module):
    """Coordinate branch with fixed Fourier features for sharper Q gradients."""

    def __init__(self, hidden_dim, num_freqs=2):
        super().__init__()
        freqs = (2.0 ** torch.arange(num_freqs, dtype=torch.float32)) * torch.pi
        self.register_buffer("freqs", freqs)
        in_dim = 3 + 3 * 2 * num_freqs
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_ms):
        encoded = query_ms.unsqueeze(-1) * self.freqs.view(1, 1, -1)
        features = torch.cat(
            (query_ms, torch.sin(encoded).flatten(1), torch.cos(encoded).flatten(1)),
            dim=1)
        return self.net(features)


class PointsToSurfMedialModel(nn.Module):
    """
    Frozen Points2Surf backbone + trainable medial field head on the shared bottleneck.
    """

    def __init__(self, backbone: Any, hidden_mult=0.5, use_query_coords=False,
                 detach_backbone_for_medial_grad=False, use_query_residual_head=False,
                 query_residual_encoding="mlp", query_fourier_num_freqs=2):
        super().__init__()
        self.backbone = backbone
        self.use_query_coords = use_query_coords
        self.detach_backbone_for_medial_grad = detach_backbone_for_medial_grad
        self.use_query_residual_head = use_query_residual_head
        self.query_residual_encoding = query_residual_encoding
        bottleneck_dim = int(backbone.net_size_max / 8)
        medial_in_dim = bottleneck_dim + (3 if use_query_coords else 0)
        hidden_dim = max(16, int(bottleneck_dim * hidden_mult))
        self.medial_head = nn.Sequential(
            nn.Linear(medial_in_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, 1),
        )
        if use_query_residual_head:
            coord_hidden_dim = max(32, hidden_dim)
            if query_residual_encoding == "fourier":
                self.query_residual_head = FourierQueryResidualHead(
                    coord_hidden_dim, num_freqs=query_fourier_num_freqs)
            elif query_residual_encoding == "mlp":
                self.query_residual_head = nn.Sequential(
                    nn.Linear(3, coord_hidden_dim),
                    nn.ReLU(inplace=False),
                    nn.Linear(coord_hidden_dim, coord_hidden_dim),
                    nn.ReLU(inplace=False),
                    nn.Linear(coord_hidden_dim, 1),
                )
            else:
                raise ValueError(f"Unknown query_residual_encoding: {query_residual_encoding}")
        self._freeze_backbone()

    def medial_features(self, bottleneck, batch_data):
        if self.detach_backbone_for_medial_grad:
            bottleneck = bottleneck.detach()
        if not self.use_query_coords:
            return bottleneck
        return torch.cat((bottleneck, batch_data['imp_surf_query_point_ms']), dim=1)

    def medial_raw_from_features(self, features, batch_data):
        raw = self.medial_head(features)
        if self.use_query_residual_head:
            raw = raw + self.query_residual_head(batch_data['imp_surf_query_point_ms'])
        return raw

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, batch_data):
        with torch.no_grad():
            bottleneck = self.backbone.encode_bottleneck(batch_data)
            sdf_pred = self.backbone.fc4(bottleneck)
        medial_raw = self.medial_raw_from_features(self.medial_features(bottleneck, batch_data), batch_data)
        return sdf_pred, medial_raw.squeeze(-1)

    def forward_medial(self, batch_data):
        with torch.no_grad():
            bottleneck = self.backbone.encode_bottleneck(batch_data)
        return self.medial_raw_from_features(self.medial_features(bottleneck, batch_data), batch_data).squeeze(-1)

    def with_mf_grad(self, batch_data, train_opt):
        """
        Differentiable w.r.t. query position (backbone frozen, medial head trainable).
        Returns sdf, agrad, pgrad, mf, mf_grad — same layout as MAT-style models.
        """
        from source import medial_field

        batch_data = {
            k: (v.clone() if torch.is_tensor(v) else v)
            for k, v in batch_data.items()
        }
        q = batch_data['imp_surf_query_point_ms']
        if not q.requires_grad:
            q = q.detach().requires_grad_(True)
            batch_data['imp_surf_query_point_ms'] = q

        bottleneck = self.backbone.encode_bottleneck(batch_data)
        sdf_raw = self.backbone.fc4(bottleneck)
        sdf = medial_field.process_sdf_prediction(sdf_raw, train_opt, batch_data['patch_radius_ms'])
        mf = medial_field.post_process_medial(
            self.medial_raw_from_features(self.medial_features(bottleneck, batch_data), batch_data))

        agrad = torch.autograd.grad(
            sdf.sum(), q, create_graph=True, retain_graph=True, allow_unused=True)[0]
        if agrad is None:
            agrad = torch.zeros_like(q)
        mf_grad = torch.autograd.grad(
            mf.sum(), q, create_graph=True, allow_unused=True)[0]
        if mf_grad is None:
            mf_grad = torch.zeros_like(q)

        pgrad = agrad.detach()
        return sdf, agrad, pgrad, mf, mf_grad


def load_pretrained_backbone(model_filename, param_filename, device):
    from source.points_to_surf_model import PointsToSurfModel
    from source import points_to_surf_eval
    from source.base.utils import torch_load

    train_opt = torch_load(param_filename)
    if not hasattr(train_opt, 'single_transformer'):
        train_opt.single_transformer = 0
    if not hasattr(train_opt, 'shared_transformer'):
        train_opt.shared_transformer = False

    pred_dim, _ = points_to_surf_eval.get_output_dimensions(train_opt)
    use_query_point = any(f in train_opt.outputs for f in ['imp_surf', 'imp_surf_magnitude', 'imp_surf_sign'])

    model = PointsToSurfModel(
        net_size_max=train_opt.net_size if hasattr(train_opt, 'net_size') else 1024,
        num_points=train_opt.points_per_patch,
        output_dim=pred_dim,
        use_point_stn=train_opt.use_point_stn,
        use_feat_stn=train_opt.use_feat_stn,
        sym_op=train_opt.sym_op,
        use_query_point=use_query_point,
        sub_sample_size=train_opt.sub_sample_size,
        do_augmentation=False,
        single_transformer=train_opt.single_transformer,
        shared_transformation=train_opt.shared_transformer,
    )
    state = torch_load(model_filename, map_location=device)
    if state and next(iter(state.keys())).startswith('module.'):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, train_opt
