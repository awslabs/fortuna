import tempfile

from flax import linen as nn
import jax.numpy as jnp
import numpy as np
import pytest

from fortuna.data.loader import DataLoader
from fortuna.metric.classification import accuracy
from fortuna.metric.regression import rmse
from fortuna.partitioner.base import Partitioner
from fortuna.prob_model import (
    CalibConfig,
    CalibOptimizer,
    CalibProcessor,
    FitConfig,
    FitMonitor,
    SNGPPosteriorApproximator,
)
from fortuna.prob_model.classification import ProbClassifier
from fortuna.prob_model.fit_config.checkpointer import FitCheckpointer
from fortuna.prob_model.fit_config.optimizer import FitOptimizer
from fortuna.prob_model.posterior.deep_ensemble.deep_ensemble_posterior import (
    DeepEnsemblePosteriorApproximator,
)
from fortuna.prob_model.posterior.laplace.laplace_posterior import (
    LaplacePosteriorApproximator,
)
from fortuna.prob_model.posterior.map.map_approximator import MAPPosteriorApproximator
from fortuna.prob_model.posterior.normalizing_flow.advi.advi_posterior import (
    ADVIPosteriorApproximator,
)
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld.cyclical_sgld_posterior import (
    CyclicalSGLDPosteriorApproximator,
)
from fortuna.prob_model.posterior.sgmcmc.sghmc.sghmc_posterior import (
    SGHMCPosteriorApproximator,
)
from fortuna.prob_model.posterior.swag.swag_posterior import SWAGPosteriorApproximator
from fortuna.prob_model.regression import ProbRegressor
from tests.make_data import make_array_random_data
from tests.make_model import (
    MyModel,
    MyModelWithSpectralNorm,
)

OUTPUT_DIM = 2
BATCH_SIZE = 8
INPUT_SHAPE = (3,)
N_DATA = 16

METHODS = {
    "map": MAPPosteriorApproximator(),
    "advi": ADVIPosteriorApproximator(),
    "laplace": LaplacePosteriorApproximator(),
    "swag": SWAGPosteriorApproximator(rank=2),
    "deep_ensemble": DeepEnsemblePosteriorApproximator(ensemble_size=2),
    "sngp": SNGPPosteriorApproximator(output_dim=OUTPUT_DIM, gp_hidden_features=2),
    "sghmc": SGHMCPosteriorApproximator(n_samples=3, n_thinning=1, burnin_length=1),
    "cyclical_sgld": CyclicalSGLDPosteriorApproximator(
        n_samples=3, n_thinning=1, cycle_length=4
    ),
}


def make_data_loader(
    task,
    n_data=N_DATA,
    input_shape=INPUT_SHAPE,
    output_dim=OUTPUT_DIM,
    batch_size=BATCH_SIZE,
):
    x_train, y_train = make_array_random_data(
        n_data=n_data,
        shape_inputs=input_shape,
        output_dim=output_dim,
        output_type="continuous" if task == "regression" else "discrete",
    )
    x_train /= np.max(x_train)
    if task == "regression":
        y_train /= np.max(y_train)
    return DataLoader.from_array_data((x_train, y_train), batch_size=batch_size)


def fit_config(
    task, restore_dir, start_current, save_dir, dump_state, save_n_steps, freeze
):
    return FitConfig(
        optimizer=FitOptimizer(n_epochs=3, freeze_fun=freeze),
        monitor=FitMonitor(metrics=(accuracy if task == "classification" else rmse,)),
        checkpointer=FitCheckpointer(
            start_from_current_state=start_current,
            restore_checkpoint_dir=restore_dir,
            save_checkpoint_dir=save_dir,
            dump_state=dump_state,
            save_every_n_steps=save_n_steps,
        ),
    )


calib_config = CalibConfig(
    optimizer=CalibOptimizer(n_epochs=3),
    processor=CalibProcessor(n_posterior_samples=2),
)


def train(
    task,
    model,
    train_data_loader,
    val_data_loader,
    calib_data_loader,
    restore_dir=None,
    start_current=False,
    save_dir=None,
    dump_state=False,
    save_n_steps=None,
    freeze=None,
    map_fit_config=None,
):
    model.train(
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        calib_data_loader=calib_data_loader,
        fit_config=fit_config(
            task,
            restore_dir,
            start_current,
            save_dir,
            dump_state,
            save_n_steps,
            freeze,
        ),
        calib_config=calib_config,
        map_fit_config=map_fit_config,
    )


def sample(method, model, train_data_loader):
    if method in ["swag"]:
        sample = model.posterior.sample(
            inputs_loader=train_data_loader.to_inputs_loader()
        )
    else:
        sample = model.posterior.sample()


def train_and_sample(
    task,
    method,
    model,
    train_data_loader,
    val_data_loader,
    calib_data_loader,
    restore_dir=None,
    start_current=False,
    save_dir=None,
    dump_state=False,
    save_n_steps=None,
    freeze=None,
    map_fit_config=None,
):
    train(
        task,
        model,
        train_data_loader,
        val_data_loader,
        calib_data_loader,
        restore_dir,
        start_current,
        save_dir,
        dump_state,
        save_n_steps,
        freeze,
        map_fit_config,
    )
    sample(method, model, train_data_loader)


def define_prob_model(task, method, model_editor=None):
    partitioner = Partitioner(
        axes_dims={"mp": 1, "fsdp": 1, "dp": 1},
        rules={"l1/kernel": (None, "mp"), "bn1": ("mp",)},
    )

    if task == "regression":
        return ProbRegressor(
            model=MyModel(OUTPUT_DIM),
            likelihood_log_variance_model=MyModel(OUTPUT_DIM),
            posterior_approximator=METHODS[method],
            model_editor=model_editor,
            partitioner=partitioner,
        )
    else:
        return ProbClassifier(
            model=MyModel(OUTPUT_DIM)
            if method != "sngp"
            else MyModelWithSpectralNorm(OUTPUT_DIM),
            posterior_approximator=METHODS[method],
            model_editor=model_editor,
            partitioner=partitioner,
        )


class ModelEditor(nn.Module):
    @nn.compact
    def __call__(self, apply_fn, model_params, x, has_aux: bool):
        log_temp = self.param("log_temp", nn.initializers.zeros, (1,))
        f = apply_fn(model_params, x)
        if has_aux:
            f, aux = f
        f += log_temp
        if has_aux:
            return f, aux
        return f


def dryrun_task(task, method):
    freeze_fun = lambda p, v: "trainable" if "l2" in p and "model" in p else "frozen"

    train_data_loader = make_data_loader(task)
    val_data_loader = make_data_loader(task)
    calib_data_loader = make_data_loader(task)

    prob_model = define_prob_model(task, method)
    map_fit_config = fit_config(
        task,
        restore_dir=None,
        start_current=None,
        save_dir=None,
        dump_state=False,
        save_n_steps=None,
        freeze=None,
    )
    train_and_sample(
        task,
        method,
        prob_model,
        train_data_loader,
        val_data_loader,
        calib_data_loader,
        map_fit_config=map_fit_config,
    )
    train_and_sample(
        task,
        method,
        prob_model,
        train_data_loader,
        val_data_loader,
        calib_data_loader,
        start_current=True,
    )

    if method not in ["laplace", "swag"]:
        train_and_sample(
            task,
            method,
            prob_model,
            train_data_loader,
            val_data_loader,
            calib_data_loader,
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        map_fit_config = fit_config(
            task,
            restore_dir=None,
            start_current=None,
            save_dir=None,
            dump_state=False,
            save_n_steps=None,
            freeze=None,
        )
        train_and_sample(
            task,
            method,
            prob_model,
            train_data_loader,
            val_data_loader,
            calib_data_loader,
            map_fit_config=map_fit_config,
            save_dir=tmp_dir,
            dump_state=True,
        )
        train_and_sample(
            task,
            method,
            prob_model,
            train_data_loader,
            val_data_loader,
            calib_data_loader,
            restore_dir=tmp_dir,
        )

        prob_model = define_prob_model(task, method)
        prob_model.load_state(tmp_dir + "/last")
        sample(method, prob_model, train_data_loader)
        prob_model.predictive.log_prob(train_data_loader)

        prob_model = define_prob_model(task, method)
        if method not in ["laplace", "swag"]:
            train_and_sample(
                task,
                method,
                prob_model,
                train_data_loader,
                val_data_loader,
                calib_data_loader,
                freeze=freeze_fun,
            )

        train_and_sample(
            task,
            method,
            prob_model,
            train_data_loader,
            val_data_loader,
            calib_data_loader,
            start_current=True,
            freeze=freeze_fun,
        )
        train_and_sample(
            task,
            method,
            prob_model,
            train_data_loader,
            val_data_loader,
            calib_data_loader,
            save_dir=tmp_dir + "2",
            dump_state=True,
            restore_dir=tmp_dir,
            freeze=freeze_fun,
        )
        train_and_sample(
            task,
            method,
            prob_model,
            train_data_loader,
            val_data_loader,
            calib_data_loader,
            save_dir=tmp_dir + "3",
            dump_state=True,
            restore_dir=tmp_dir + "2",
            freeze=freeze_fun,
        )
        train_and_sample(
            task,
            method,
            prob_model,
            train_data_loader,
            val_data_loader,
            calib_data_loader,
            map_fit_config=fit_config(
                task,
                restore_dir=None,
                start_current=None,
                save_dir=None,
                dump_state=False,
                save_n_steps=None,
                freeze=None,
            ),
            save_dir=tmp_dir + "4",
            dump_state=True,
            freeze=freeze_fun,
        )

        train_and_sample(
            task,
            method,
            prob_model,
            train_data_loader,
            val_data_loader,
            calib_data_loader,
            start_current=True,
            save_dir=tmp_dir + "5",
            save_n_steps=1,
            freeze=freeze_fun,
        )
        prob_model = define_prob_model(task, method)
        prob_model.load_state(tmp_dir + "5")
        sample(method, prob_model, train_data_loader)
        prob_model.predictive.log_prob(train_data_loader)

    prob_model = define_prob_model(task, method, model_editor=ModelEditor())
    train_and_sample(
        task,
        method,
        prob_model,
        train_data_loader,
        calib_data_loader,
        val_data_loader,
        map_fit_config=map_fit_config,
    )
    state = (
        prob_model.posterior.state.get()
        if method not in ["deep_ensemble", "sghmc", "cyclical_sgld"]
        else prob_model.posterior.state.get(-1)
    )
    model_editor_params = state.params["model_editor"]["params"].unfreeze()
    if method in ["advi", "laplace"]:
        model_editor_params = model_editor_params["mean"]
    assert not jnp.allclose(model_editor_params["log_temp"], jnp.array([0.0]))


@pytest.mark.parametrize("method", METHODS.keys())
def test_dryrun_classification(method):
    dryrun_task(task="classification", method=method)


@pytest.mark.parametrize("method", [m for m in METHODS.keys() if m != "sngp"])
def test_dryrun_regression(method):
    dryrun_task(task="regression", method=method)
