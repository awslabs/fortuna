from typing import Callable, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from fortuna.data.loader import InputsLoader
from fortuna.prob_model.regression import ProbRegressor
from fortuna.typing import Array, Batch
from matplotlib.collections import Collection
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def plot_2d_classification_predictions_and_uncertainty(
    inputs: Optional[np.ndarray] = None,
    preds: Optional[np.ndarray] = None,
    grid_uncertainty: Optional[np.ndarray] = None,
    uncertainty_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    grid: Optional[np.ndarray] = None,
    xx1: Optional[np.ndarray] = None,
    xx2: Optional[np.ndarray] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    figsize: tuple = (5, 5),
    title: Optional[str] = None,
    uncertainty_cmap: str = "Blues",
    preds_color: Tuple[str, str] = ("C2", "C3"),
    base_inputs_color: str = "black",
    marker_size: int = 10,
    legend: bool = True,
    uncertainty_label: str = "uncertainty",
    fontsize: int = 12,
    colorbar: bool = False,
    remove_ticks: bool = False,
    show: bool = False,
    return_pcolor: bool = False,
    **save_options,
) -> Optional[Collection]:
    """
    Plot inputs, predictions and uncertainty.

    :param inputs: Optional[np.ndarray]
        Input data. This must be a two-dimensional array, with different inputs over the first axis. The number of
        dimensions over the second axis must be 2.
    :param preds: Optional[np.ndarray]
        Predictions. This must be a one-dimensional array of zeros and ones, with different inputs over the first
        axis.
    :param grid_uncertainty: Optional[np.ndarray]
        Scalar uncertainty value for each element of the grid. This must have the same dimensionality as `grid`.
    :param uncertainty_fn: Optional[Callable[[np.ndarray], np.ndarray]]
        A function taking as argument a grid of inputs, and returning a scalar uncertainty value for each element
        in the grid.
    :param grid: Optional[np.ndarray]
        The grid to display the plot over.
    :param xx1: Optional[np.ndarray]
        Mesh over the first axis.
    :param xx2: Optional[np.ndarray]
        Mesh over the second axis.
    :param ax: Optional[matplotlib.axes.Axes]
        Axis matplotlib object.
    :param figsize: tuple
        Figure size.
    :param title: Optional[str]
        Figure title.
    :param uncertainty_cmap: str
        Uncertainty color map.
    :param preds_color: Tuple[str, str]
        Color of predictions. The first element of the tuple is associated to the prediction 1; the second to 0.
    :param base_inputs_color: str
        Color of inputs if `preds` is not available.
    :param marker_size: int
        Size of scattered inputs.
    :param legend: bool
        Whether to include a legend.
    :param uncertainty_label: str
        Uncertainty label in the legend.
    :param fontsize: int
        Font size in the legend.
    :param colorbar: bool
        Whether to include a color bar.
    :param remove_ticks: bool
        Whether to remove ticks from axes.
    :param save_options: dict
        Options to save the file with `matplotlib.pyplot.savefig`. If no option is given, the file is not saved.
    :param show: bool
        Whether to show the plot.
    :param return_pcolor: bool
        Whether to return the output of `matplotlib.pyplot.pcolor`.

    :return Optional[Collection]
        If `return_pcolor`, it returns the output of `matplotlib.pyplot.pcolor`.
    """
    if (xx1 is None or xx2 is None) and (inputs is None):
        raise ValueError(
            """If `inputs` is not passed, please pass either both `xx1` and `xx2`, or `grid`.
        """
        )
    if grid_uncertainty is None and uncertainty_fn is None:
        raise ValueError(
            """ One between `grid_uncertainty` and `uncertainty_fn` must be passed. If both are passed,
        `uncertainty_fn` is ignored."""
        )
    if grid is None and uncertainty_fn is None:
        raise ValueError(
            """If `grid` is not passed, then `uncertainty_fn` must be passed."""
        )
    if grid is not None and (uncertainty_fn is None and grid_uncertainty is None):
        raise ValueError(
            """If `grid` is not passed, then at least one of `uncertainty_fn` and `grid_uncertainty`
        must be passed."""
        )
    if grid_uncertainty is not None and grid is None:
        raise ValueError(
            """If `grid_uncertainty` is passed, then `grid` must also be passed."""
        )
    if grid is None:
        if xx1 is None or xx2 is None:
            mins, maxs = np.min(inputs, 0), np.max(inputs, 0)
            if xx1 is None:
                xx1 = np.linspace(
                    mins[0] - 1.25 * (maxs[0] - mins[0]),
                    maxs[0] + 1.25 * (maxs[0] - mins[0]),
                    100 if xx2 is None else len(xx2),
                )
            if xx2 is None:
                xx2 = np.linspace(
                    mins[1] - 1.25 * (maxs[1] - mins[1]),
                    maxs[1] + 1.25 * (maxs[1] - mins[1]),
                    100 if xx1 is None else len(xx1),
                )
        grid = np.array([[_xx1, _xx2] for _xx1 in xx1 for _xx2 in xx2])
    if grid_uncertainty is None:
        grid_uncertainty = uncertainty_fn(grid)
    size1 = len(np.unique(grid[:, 0])) if xx1 is None else len(xx1)
    size2 = len(np.unique(grid[:, 1])) if xx2 is None else len(xx2)
    grid_uncertainty = grid_uncertainty.reshape(size1, size2)
    grid = grid.reshape(size1, size2, 2)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    if title:
        ax.set_title(title, fontsize=fontsize + 2)
    im = ax.pcolor(
        grid[:, :, 0],
        grid[:, :, 1],
        grid_uncertainty,
        cmap=uncertainty_cmap,
        label=uncertainty_label,
    )
    if inputs is not None:
        ax.scatter(
            inputs[:, 0],
            inputs[:, 1],
            s=marker_size,
            c=[preds_color[0] if i == 1 else preds_color[1] for i in preds]
            if preds is not None
            else base_inputs_color,
        )
    if colorbar:
        plt.colorbar(im, ax=ax.ravel().tolist() if hasattr(ax, "ravel") else ax)
    if legend:
        ax.legend(fontsize=fontsize)
    if remove_ticks:
        ax.xticks([], [])
        ax.yticks([], [])
    if len(save_options) > 0:
        plt.savefig(**save_options)
    if show:
        plt.show()

    if return_pcolor:
        return im


def plot_reliability_diagram(
    accs: Union[Array, List[Array]],
    confs: Union[Array, List[Array]],
    ax: Optional[matplotlib.axes.Axes] = None,
    labels: Optional[List[str]] = None,
    figsize: tuple = (6, 4),
    legend_loc: Optional[str] = None,
    fontsize: int = 10,
    title: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show: bool = False,
    **save_options,
) -> None:
    """
    Plot a reliability diagram.

    Parameters
    ----------
    accs: Union[Array, List[Array, ...]]
        An accuracy score for each bin. A list of multiple accuracy scores is also accepted.
    confs: Union[Array, List[Array, ...]]
        A confidence score for each bin. A list of multiple confidence scores is also accepted.
    labels: Optional[List[str, ...]]
        Labels used in the legend presumably describing the method used to generate accuracy and confidence scores. A
        list of labels, corresponding to lists of accuracies and confidences, is also accepted.
    ax: Optional[matplotlib.axes.Axes]
        Axis matplotlib object.
    figsize: Optional[tuple]
        Figure size.
    legend_loc: Optional[str]
        Legend location, as in :code:`matplotlib`.
        See `here <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html>`__
    fontsize: Optional[int]
        Font size.
    title: Optional[str] = None
        Plot title.
    ylim: Optional[Tuple[float, float]]
        Bottom and top limits on the y-axis.
    show: bool
        Whether to show the plot.
    save_options: dict
        Options to save the file with `matplotlib.pyplot.savefig`. If no option is given, the file is not saved.
    """
    if figsize is None:
        figsize = (6, 4)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    if type(confs) != list:
        confs = [confs]
    if type(accs) != list:
        accs = [accs]
    if labels:
        if type(labels) != list:
            labels = [labels]
    if len(accs) != len(confs):
        raise ValueError("`accs` and `confs` must contain the same number of entries.")
    if labels:
        if len(accs) != len(labels):
            raise ValueError(
                "`accs` and `labels` must contain the same number of entries."
            )
    ax.grid()
    ax.plot([0, 1], [0, 0], color="grey", linestyle="--", alpha=0.3)
    for i, (a, c) in enumerate(zip(accs, confs)):
        ax.plot(
            c, c - a, marker=".", linestyle="-", label=labels[i] if labels else None
        )
    if labels:
        ax.legend(fontsize=fontsize, loc=legend_loc if legend_loc else None)
    ax.set_xlabel("confidence", fontsize=fontsize)
    ax.set_ylabel("confidence - accuracy", fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize + 2)
    if ylim:
        ax.set_ylim(ylim)
    if len(save_options) > 0:
        plt.savefig(**save_options)
    if show:
        plt.show()
    if ax:
        return ax


def plot_1d_regression_preds_and_std(
    prob_model: ProbRegressor,
    mesh: np.ndarray,
    data: Optional[Batch] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    n_std: float = 2.0,
    title: Optional[str] = None,
    figsize: Tuple = (5, 2),
    fontsize: int = 12,
    marker_size_data: int = 1,
    legend: bool = True,
    show: bool = False,
    **saving_options,
) -> Optional[Collection]:
    mesh_loader = InputsLoader.from_array_inputs(mesh)
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    if title:
        ax.set_title(title, fontsize=fontsize + 2)

    mean = prob_model.predictive.mean(mesh_loader)
    std = prob_model.predictive.std(mesh_loader)
    if data:
        ax.scatter(*data, s=marker_size_data, color="C0", label="data")
    ax.plot(mesh, mean, color="C1", label="mean")
    ax.fill_between(
        mesh,
        (mean - n_std * std).squeeze(1),
        (mean + n_std * std).squeeze(1),
        alpha=0.3,
        color="C0",
        label=f"+/- {n_std} std",
    )
    if legend:
        ax.legend(fontsize=fontsize - 2)
    plt.tight_layout()
    if show:
        plt.show()
    if len(saving_options) > 0:
        plt.savefig(**saving_options)
    return ax


def radar_factory(num_vars, frame="circle"):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(num_vars),
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta
