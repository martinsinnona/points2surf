"""Plot helpers for the minimal Points2Surf experiment."""

import numpy as np
import matplotlib.pyplot as plt
import trimesh
from matplotlib.colors import FuncNorm

from source import sdf


def _draw_sampling_slice(ax, mesh, points, kinds, volume_padding, slice_z,
                         slice_half_thickness, title):
    """Draw one XY slice of query categories and the mesh cross-section."""
    in_slice = np.abs(points[:, 2] - slice_z) <= slice_half_thickness
    for kind, label, color in ((0, 'uniform volume', 'tab:blue'),
                               (1, 'near surface', 'tab:orange'),
                               (2, 'GT medial ridge band', 'tab:green')):
        mask = in_slice & (kinds == kind)
        if mask.any():
            ax.scatter(points[mask, 0], points[mask, 1], s=18, alpha=0.75,
                       color=color, label=label)
    section = trimesh.intersections.mesh_plane(
        mesh, plane_normal=[0, 0, 1], plane_origin=[0, 0, slice_z])
    for segment in section:
        ax.plot(segment[:, 0], segment[:, 1], color='black', linewidth=1.4)
    plot_min, plot_max = mesh.bounds[0, :2] - volume_padding, mesh.bounds[1, :2] + volume_padding
    ax.set(xlim=(plot_min[0], plot_max[0]), ylim=(plot_min[1], plot_max[1]), aspect='equal')
    ax.set_title(title, fontsize=10, pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=9, loc='upper right')


def plot_active_sampling(mesh, points, kinds, volume_padding, medial_enabled,
                         slice_z=0.0, slice_half_thickness=0.025):
    """Draw the sampling distribution currently used for training."""
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    title = 'Sampling at thin slice (z=0)'
    _draw_sampling_slice(ax, mesh, points, kinds, volume_padding,
                         slice_z, slice_half_thickness, title)
    fig.tight_layout(pad=1.0)
    return fig


def plot_training_history(history):
    """Plot training losses and the two evaluation SDF MSE curves."""
    fig, (loss_ax, mse_ax) = plt.subplots(1, 2, figsize=(12.5, 4.2))
    epochs = np.arange(1, len(history['total']) + 1)
    loss_names = ['total', 'signed', 'magnitude', 'sign']
    if 'eikonal' in history:
        loss_names.append('eikonal')
    for name in loss_names:
        loss_ax.plot(epochs, history[name], label=name)
    loss_ax.set(xlabel='epoch', ylabel='training loss', yscale='log')
    loss_ax.set_title('Points2Surf training loss', fontsize=10, pad=12)
    loss_ax.grid(alpha=0.25)
    loss_ax.legend()
    evaluated = np.isfinite(history['eval_mse'])
    mse_ax.plot(epochs[evaluated], np.asarray(history['eval_mse'])[evaluated],
                marker='o', color='tab:purple', label='uniform volume')
    mse_ax.plot(epochs[evaluated], np.asarray(history['eval_medial_mse'])[evaluated],
                marker='o', color='tab:green', label='GT medial axis')
    mse_ax.set(xlabel='epoch', ylabel='MSE (log)', yscale='log')
    mse_ax.set_title('SDF MSE', fontsize=10, pad=12)
    mse_ax.grid(alpha=0.25)
    mse_ax.legend()
    if history.get('selected_epoch') is not None:
        for ax in (loss_ax, mse_ax):
            ax.axvline(history['selected_epoch'], color='black', linestyle=':',
                       alpha=0.6)
    fig.tight_layout(pad=2.0, w_pad=3.0)
    return fig


def plot_medial_training_history(history):
    """Plot the three raw DMF losses and their weighted sum separately."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    epochs = np.arange(1, len(history['medial_total']) + 1)
    for name in ('maximality', 'inscription', 'orthogonality'):
        ax.plot(epochs, history[name], label=name)
    ax.plot(epochs, history['medial_total'], color='black', linewidth=2,
            label='weighted total')
    ax.set(xlabel='epoch', ylabel='medial loss', yscale='log')
    ax.set_title('Interior medial-field losses', fontsize=10, pad=12)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout(pad=1.5)
    return fig


def plot_medial_ablation(results):
    """Compare leave-one-change-out medial-field ablations."""
    labels = [result['name'] for result in results]
    positions = np.arange(len(labels))
    fig, (mae_ax, orthogonal_ax) = plt.subplots(
        1, 2, figsize=(11.5, 4.4), sharey=True)
    mae = [result['medial_mae'] for result in results]
    orthogonality = [result['orthogonality'] for result in results]
    colors = ['tab:blue'] * (len(results) - 1) + ['tab:green']
    mae_ax.barh(positions, mae, color=colors)
    orthogonal_ax.barh(positions, orthogonality, color=colors)
    mae_ax.set_yticks(positions, labels)
    mae_ax.invert_yaxis()
    mae_ax.set(xlabel='GT-axis radius MAE', title='Medial-field accuracy')
    orthogonal_ax.set(xlabel='final raw loss', title='Orthogonality')
    for ax, values in ((mae_ax, mae), (orthogonal_ax, orthogonality)):
        ax.grid(axis='x', alpha=0.25)
        for position, value in zip(positions, values):
            ax.text(value, position, f' {value:.4f}', va='center', fontsize=9)
    fig.suptitle('Leave-one-change-out ablation (lower is better)', fontsize=11)
    fig.tight_layout(pad=1.5, w_pad=3.0)
    return fig


def plot_mse_comparison(histories):
    """Compare uniform-volume and medial-axis SDF MSE across experiments."""
    fig, (volume_ax, medial_ax) = plt.subplots(1, 2, figsize=(11.5, 4.0),
                                               sharex=True, sharey=True)
    for label, history in histories.items():
        epochs = np.arange(1, len(history['eval_mse']) + 1)
        evaluated = np.isfinite(history['eval_mse'])
        style = {'color': 'black', 'linestyle': '--', 'linewidth': 2.2} \
            if label == '1 local + 1 global' else {}
        volume_ax.plot(epochs[evaluated], np.asarray(history['eval_mse'])[evaluated],
                       marker='o', label=label, **style)
        medial_ax.plot(epochs[evaluated], np.asarray(history['eval_medial_mse'])[evaluated],
                       marker='o', label=label, **style)
    for ax, title in ((volume_ax, 'Uniform volume'), (medial_ax, 'Medial axis')):
        ax.set(xlabel='epoch', ylabel='SDF MSE', yscale='log')
        ax.set_title(title, fontsize=10, pad=12)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout(pad=2.0, w_pad=3.0)
    return fig


def plot_sdf_slices(mesh, predict_sdf, predict_medial=None,
                     slice_zs=(0.0, 0.25), resolution=128,
                     axis_limit=0.65, isoline_spacing=0.05,
                     interior_gamma=2.0):
    """Compare predicted/GT SDF by row and optionally show interior M at z=0."""
    if isoline_spacing <= 0:
        raise ValueError('isoline_spacing must be positive.')
    if interior_gamma <= 0:
        raise ValueError('interior_gamma must be positive.')
    if len(slice_zs) != 2:
        raise ValueError('This comparison expects exactly two slice_zs.')
    axis_values = np.linspace(-axis_limit, axis_limit, resolution, dtype=np.float32)
    xx, yy = np.meshgrid(axis_values, axis_values)
    fields = []
    for z in slice_zs:
        queries = np.column_stack((xx.ravel(), yy.ravel(),
                                   np.full(xx.size, z, dtype=np.float32)))
        fields.append((z, predict_sdf(queries).reshape(resolution, resolution),
                       -sdf.get_signed_distance(mesh, queries).reshape(resolution, resolution)))
    all_fields = [field for _, predicted, gt in fields
                  for field in (predicted, gt)]
    negative_limit = max(
        max(-float(field.min()) for field in all_fields), isoline_spacing)
    positive_limit = max(
        max(float(field.max()) for field in all_fields), isoline_spacing)

    def color_forward(values):
        values = np.asarray(values)
        mapped = np.empty_like(values, dtype=float)
        inside = values < 0.0
        inside_depth = np.clip(-values[inside] / negative_limit, 0.0, 1.0)
        mapped[inside] = 0.5 * (1.0 - inside_depth ** interior_gamma)
        mapped[~inside] = 0.5 + 0.5 * np.clip(
            values[~inside] / positive_limit, 0.0, 1.0)
        return mapped

    def color_inverse(values):
        values = np.asarray(values)
        distances = np.empty_like(values, dtype=float)
        inside = values < 0.5
        distances[inside] = -negative_limit * (
            1.0 - 2.0 * values[inside]) ** (1.0 / interior_gamma)
        distances[~inside] = positive_limit * (2.0 * values[~inside] - 1.0)
        return distances

    color_norm = FuncNorm(
        (color_forward, color_inverse), vmin=-negative_limit,
        vmax=positive_limit, clip=True)
    level_limit = max(negative_limit, positive_limit)
    level_count = int(np.ceil(level_limit / isoline_spacing))
    levels = np.arange(-level_count, level_count + 1) * isoline_spacing
    levels = levels[~np.isclose(levels, 0.0)]
    column_count = 3 if predict_medial is not None else 2
    fig = plt.figure(figsize=(10.4 if predict_medial is not None else 7.0, 6.4),
                     constrained_layout=True)
    grid = fig.add_gridspec(2, column_count, wspace=0.04, hspace=0.04)
    axes = np.empty((2, 2), dtype=object)
    sdf_image = None
    for column, (z, predicted, gt) in enumerate(fields):
        for row, (field, row_label) in enumerate(
                ((predicted, 'Predicted SDF'), (gt, 'GT SDF'))):
            ax = fig.add_subplot(grid[row, column])
            axes[row, column] = ax
            sdf_image = ax.imshow(
                field, origin='lower', extent=[axis_values[0], axis_values[-1]] * 2,
                cmap='coolwarm', norm=color_norm)
            visible_levels = levels[(levels > field.min()) & (levels < field.max())]
            if len(visible_levels):
                ax.contour(xx, yy, field, levels=visible_levels,
                           colors='black', linewidths=0.5, alpha=0.4,
                           linestyles='solid')
            if field.min() < 0 < field.max():
                ax.contour(xx, yy, field, levels=[0.0], colors='black',
                           linewidths=1.1, alpha=0.9, linestyles='solid')
            ax.set(aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(f'z = {z:g}', fontsize=10, pad=10)
            if column == 0:
                ax.set_ylabel(row_label, fontsize=10, labelpad=12)

    sdf_ticks = [-negative_limit, -negative_limit / 2.0, 0.0,
                 positive_limit / 2.0, positive_limit]
    sdf_bar = fig.colorbar(
        sdf_image, ax=axes.ravel().tolist(), orientation='horizontal',
        fraction=0.055, pad=0.06, aspect=32, label='signed distance')
    sdf_bar.set_ticks(sdf_ticks, labels=[f'{tick:.2f}' for tick in sdf_ticks])

    if predict_medial is not None:
        z = slice_zs[0]
        queries = np.column_stack((xx.ravel(), yy.ravel(),
                                   np.full(xx.size, z, dtype=np.float32)))
        medial = predict_medial(queries).reshape(resolution, resolution)
        gt_sdf = fields[0][2]
        interior_medial = np.ma.masked_where(gt_sdf >= 0.0, medial)
        visible_medial = medial[gt_sdf < 0.0]
        medial_limit = max(float(np.percentile(visible_medial, 99.0)), 1e-6)
        medial_ax = fig.add_subplot(grid[:, 2])
        medial_cmap = plt.get_cmap('viridis').copy()
        medial_cmap.set_bad('#eeeeee')
        medial_image = medial_ax.imshow(
            interior_medial, origin='lower',
            extent=[axis_values[0], axis_values[-1]] * 2,
            cmap=medial_cmap, vmin=0.0, vmax=medial_limit)
        medial_ax.contour(xx, yy, gt_sdf, levels=[0.0], colors='black',
                          linewidths=1.1)
        medial_ax.set(aspect='equal')
        medial_ax.set_title(f'Interior medial field M\nz = {z:g}',
                            fontsize=10, pad=10)
        medial_ax.set_xticks([])
        medial_ax.set_yticks([])
        for spine in medial_ax.spines.values():
            spine.set_visible(False)
        medial_ticks = np.linspace(0.0, medial_limit, 4)
        medial_bar = fig.colorbar(
            medial_image, ax=medial_ax, orientation='horizontal',
            fraction=0.055, pad=0.06, aspect=16, label='medial radius M')
        medial_bar.set_ticks(
            medial_ticks, labels=[f'{tick:.2f}' for tick in medial_ticks])
    return fig


def compute_sdf_error_slices(mesh, predict_sdf, slice_zs=(0.0, 0.25),
                             resolution=128, axis_limit=0.65):
    """Evaluate absolute SDF error on XY slices for one trained model."""
    axis_values = np.linspace(-axis_limit, axis_limit, resolution, dtype=np.float32)
    xx, yy = np.meshgrid(axis_values, axis_values)
    errors = []
    for z in slice_zs:
        queries = np.column_stack((xx.ravel(), yy.ravel(),
                                   np.full(xx.size, z, dtype=np.float32)))
        predicted = predict_sdf(queries).reshape(resolution, resolution)
        gt = -sdf.get_signed_distance(mesh, queries).reshape(resolution, resolution)
        errors.append(np.abs(predicted - gt))
    return errors


def plot_sdf_error_slices(mesh, errors_by_label, slice_zs=(0.0, 0.25),
                          axis_limit=0.65):
    """Compare absolute SDF error fields using one color scale across models."""
    labels = list(errors_by_label)
    if not labels:
        raise ValueError('errors_by_label must contain at least one model.')
    if any(len(errors_by_label[label]) != len(slice_zs) for label in labels):
        raise ValueError('Each model must provide one error field per slice.')

    axis_values = np.linspace(-axis_limit, axis_limit, errors_by_label[labels[0]][0].shape[0])
    xx, yy = np.meshgrid(axis_values, axis_values)
    color_limit = max(float(np.max(error))
                      for errors in errors_by_label.values() for error in errors)
    color_limit = max(color_limit, 1e-6)
    fig, axes = plt.subplots(len(slice_zs), len(labels), figsize=(3.3 * len(labels),
                                                                    3.1 * len(slice_zs)),
                             squeeze=False, constrained_layout=True)
    for row, z in enumerate(slice_zs):
        section = trimesh.intersections.mesh_plane(
            mesh, plane_normal=[0, 0, 1], plane_origin=[0, 0, z])
        for column, label in enumerate(labels):
            ax = axes[row, column]
            image = ax.imshow(errors_by_label[label][row], origin='lower',
                              extent=[axis_values[0], axis_values[-1]] * 2,
                              cmap='magma', vmin=0.0, vmax=color_limit)
            for segment in section:
                ax.plot(segment[:, 0], segment[:, 1], color='cyan', linewidth=1.1)
            ax.set(aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(label, fontsize=10, pad=12)
        axes[row, 0].set_ylabel(f'z = {z:g}', rotation=90, fontsize=11, labelpad=18)
    fig.suptitle('Absolute SDF error (cyan: surface)', fontsize=11)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02,
                 label='absolute SDF error')
    return fig
