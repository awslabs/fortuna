import tempfile

import numpy as np
import pytest

from fortuna.data.loader import DataLoader
from fortuna.metric.classification import accuracy
from fortuna.metric.regression import rmse
from fortuna.prob_model import (
    CalibConfig,
    CalibOptimizer,
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
from fortuna.prob_model.posterior.swag.swag_posterior import SWAGPosteriorApproximator
from fortuna.prob_model.posterior.sgmcmc.sghmc.sghmc_posterior import (
    SGHMCPosteriorApproximator,
)
from fortuna.prob_model.posterior.sgmcmc.cyclical_sgld.cyclical_sgld_posterior import (
    CyclicalSGLDPosteriorApproximator,
)
from fortuna.prob_model.regression import ProbRegressor
from tests.make_data import make_array_random_data
from tests.make_model import (
    MyModel,
    MyModelWithSpectralNorm,
)


OUTPUT_DIM = 2

TASKS = ["regression", "classification"]
METHODS = {
    "map": MAPPosteriorApproximator(),
    "advi": ADVIPosteriorApproximator(),
    "laplace": LaplacePosteriorApproximator(),
    "swag": SWAGPosteriorApproximator(rank=2),
    "deep_ensemble": DeepEnsemblePosteriorApproximator(ensemble_size=2),
    "sngp": SNGPPosteriorApproximator(output_dim=OUTPUT_DIM),
    "sghmc": SGHMCPosteriorApproximator(n_samples=3,
                                        n_thinning=1,
                                        burnin_length=1),
    "cyclical_sgld": CyclicalSGLDPosteriorApproximator(n_samples=3,
                                                       n_thinning=1,
                                                       cycle_length=4),
}
TASKS_METHODS = [
    (task, method)
    for task in TASKS
    for method in METHODS
    if (task, method) != ("regression", "sngp")
]
TASKS_IDS = [t + "-" + m for t, m in TASKS_METHODS]


def make_data_loader(task, n_data, input_shape, output_dim, batch_size):
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


@pytest.mark.parametrize("task, method", TASKS_METHODS, ids=TASKS_IDS)
def test_dryrun(task, method):
    batch_size = 32
    input_shape = (3,)
    n_data = 100

    train_data_loader = make_data_loader(
        task, n_data, input_shape, OUTPUT_DIM, batch_size
    )
    val_data_loader = make_data_loader(
        task, n_data, input_shape, OUTPUT_DIM, batch_size
    )
    calib_data_loader = make_data_loader(
        task, n_data, input_shape, OUTPUT_DIM, batch_size
    )

    freeze_fun = lambda p, v: "trainable" if "l2" in p and "model" in p else "frozen"

    fit_config = lambda restore_path, start_current, save_dir, dump_state, save_n_steps, freeze: FitConfig(
        optimizer=FitOptimizer(n_epochs=3, freeze_fun=freeze),
        monitor=FitMonitor(metrics=(accuracy if task == "classification" else rmse,)),
        checkpointer=FitCheckpointer(
            start_from_current_state=start_current,
            restore_checkpoint_path=restore_path,
            save_checkpoint_dir=save_dir,
            dump_state=dump_state,
            save_every_n_steps=save_n_steps,
        ),
    )

    calib_config = CalibConfig(optimizer=CalibOptimizer(n_epochs=3))

    def train(
        restore_path=None,
        start_current=False,
        save_dir=None,
        dump_state=False,
        save_n_steps=None,
        freeze=None,
        map_fit_config=None,
    ):
        prob_model.train(
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            calib_data_loader=calib_data_loader,
            fit_config=fit_config(
                restore_path, start_current, save_dir, dump_state, save_n_steps, freeze
            ),
            calib_config=calib_config,
            map_fit_config=map_fit_config,
        )

    def sample():
        if method in ["swag"]:
            sample = prob_model.posterior.sample(
                inputs_loader=train_data_loader.to_inputs_loader()
            )
        else:
            sample = prob_model.posterior.sample()

    def train_and_sample(
        restore_path=None,
        start_current=False,
        save_dir=None,
        dump_state=False,
        save_n_steps=None,
        freeze=None,
        map_fit_config=None,
    ):
        train(
            restore_path,
            start_current,
            save_dir,
            dump_state,
            save_n_steps,
            freeze,
            map_fit_config,
        )
        sample()

    def define_prob_model():
        if task == "regression":
            return ProbRegressor(
                model=MyModel(OUTPUT_DIM),
                likelihood_log_variance_model=MyModel(OUTPUT_DIM),
                posterior_approximator=METHODS[method],
            )
        else:
            return ProbClassifier(
                model=MyModel(OUTPUT_DIM)
                if method != "sngp"
                else MyModelWithSpectralNorm(OUTPUT_DIM),
                posterior_approximator=METHODS[method],
            )

    prob_model = define_prob_model()
    train_and_sample(
        map_fit_config=fit_config(
            restore_path=None,
            start_current=None,
            save_dir=None,
            dump_state=False,
            save_n_steps=None,
            freeze=None,
        )
    )
    train_and_sample(start_current=True)
    if method not in ["laplace", "swag"]:
        train_and_sample()

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_and_sample(
            map_fit_config=fit_config(
                restore_path=None,
                start_current=None,
                save_dir=None,
                dump_state=False,
                save_n_steps=None,
                freeze=None,
            ),
            save_dir=tmp_dir,
            dump_state=True,
        )
        train_and_sample(restore_path=tmp_dir)

        prob_model = define_prob_model()
        prob_model.load_state(tmp_dir)
        sample()
        prob_model.predictive.log_prob(train_data_loader)

        if method not in ["laplace", "swag"]:
            train_and_sample(freeze=freeze_fun)
        train_and_sample(start_current=True, freeze=freeze_fun)
        train_and_sample(
            save_dir=tmp_dir, dump_state=True, restore_path=tmp_dir, freeze=freeze_fun
        )
        train_and_sample(
            save_dir=tmp_dir, dump_state=True, restore_path=tmp_dir, freeze=freeze_fun
        )
        train_and_sample(
            map_fit_config=fit_config(
                restore_path=None,
                start_current=None,
                save_dir=None,
                dump_state=False,
                save_n_steps=None,
                freeze=None,
            ),
            save_dir=tmp_dir,
            dump_state=True,
            freeze=freeze_fun,
        )

        train_and_sample(
            start_current=True,
            save_dir=tmp_dir + "/tmp",
            save_n_steps=1,
            freeze=freeze_fun,
        )
        prob_model = define_prob_model()
        prob_model.load_state(tmp_dir + "/tmp")
        sample()
        prob_model.predictive.log_prob(train_data_loader)
