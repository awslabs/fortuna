import tempfile

from flax import linen as nn
import numpy as np

from fortuna.data.loader import DataLoader
from fortuna.metric.classification import accuracy
from fortuna.metric.regression import rmse
from fortuna.partitioner.base import Partitioner
from fortuna.calib_model import (
    Config,
    Monitor,
    Checkpointer,
    Optimizer
)
from fortuna.calib_model.classification import CalibClassifier
from fortuna.calib_model.regression import CalibRegressor
from tests.make_data import make_array_random_data
from tests.make_model import MyModel

OUTPUT_DIM = 2
BATCH_SIZE = 8
INPUT_SHAPE = (3,)
N_DATA = 16


def accuracy2(preds, probs, targets):
    return accuracy(preds, targets)


def rmse2(preds, probs, targets):
    return rmse(preds, targets)


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


def config(
    task, restore_dir, start_current, save_dir, dump_state, save_n_steps, freeze
):
    return Config(
        optimizer=Optimizer(n_epochs=3, freeze_fun=freeze),
        monitor=Monitor(metrics=(accuracy2 if task == "classification" else rmse2,)),
        checkpointer=Checkpointer(
            start_from_current_state=start_current,
            restore_checkpoint_dir=restore_dir,
            save_checkpoint_dir=save_dir,
            dump_state=dump_state,
            save_every_n_steps=save_n_steps,
        ),
    )


def calibrate(
    task,
    model,
    calib_data_loader,
    val_data_loader,
    restore_dir=None,
    start_current=False,
    save_dir=None,
    dump_state=False,
    save_n_steps=None,
    freeze=None,
):
    model.calibrate(
        calib_data_loader=calib_data_loader,
        val_data_loader=val_data_loader,
        config=config(
            task,
            restore_dir,
            start_current,
            save_dir,
            dump_state,
            save_n_steps,
            freeze,
        ),
    )


def define_calib_model(task, model_editor=None):
    partitioner = Partitioner(
        axes_dims={"mp": 2, "fsdp": 2, "dp": 2},
        rules={"l1/kernel": (None, "mp"), "bn1": ("mp",)},
    )

    if task == "regression":
        return CalibRegressor(
            model=MyModel(OUTPUT_DIM),
            likelihood_log_variance_model=MyModel(OUTPUT_DIM),
            model_editor=model_editor,
            partitioner=partitioner,
        )
    else:
        return CalibClassifier(
            model=MyModel(OUTPUT_DIM),
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


def dryrun_task(task):
    freeze_fun = lambda p, v: "trainable" if "l2" in p and "model" in p else "frozen"

    calib_data_loader = make_data_loader(task)
    val_data_loader = make_data_loader(task)

    calib_model = define_calib_model(task)
    calibrate(
        task,
        calib_model,
        calib_data_loader,
        val_data_loader,
    )
    calibrate(
        task,
        calib_model,
        calib_data_loader,
        val_data_loader,
        start_current=True,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        calibrate(
            task,
            calib_model,
            calib_data_loader,
            val_data_loader,
            save_dir=tmp_dir,
            dump_state=True,
        )
        calibrate(
            task,
            calib_model,
            calib_data_loader,
            val_data_loader,
            restore_dir=tmp_dir,
        )

        calib_model = define_calib_model(task)
        calib_model.load_state(tmp_dir + "/last")
        calib_model.predictive.log_prob(calib_data_loader)

        calib_model = define_calib_model(task)
        calibrate(
            task,
            calib_model,
            calib_data_loader,
            val_data_loader,
            freeze=freeze_fun,
        )

        calibrate(
            task,
            calib_model,
            calib_data_loader,
            val_data_loader,
            start_current=True,
            freeze=freeze_fun,
        )

        calibrate(
            task,
            calib_model,
            calib_data_loader,
            val_data_loader,
            save_dir=tmp_dir + "/frozen",
            dump_state=True,
            freeze=freeze_fun,
        )

        calibrate(
            task,
            calib_model,
            calib_data_loader,
            val_data_loader,
            restore_dir=tmp_dir + "/frozen",
            freeze=freeze_fun,
        )

        calibrate(
            task,
            calib_model,
            calib_data_loader,
            val_data_loader,
            start_current=True,
            save_dir=tmp_dir + "/frozen/tmp",
            save_n_steps=1,
            freeze=freeze_fun,
        )
        calib_model = define_calib_model(task)
        calib_model.load_state(tmp_dir + "/frozen/tmp")
        calib_model.predictive.log_prob(calib_data_loader)

    calib_model = define_calib_model(task, model_editor=ModelEditor())
    calibrate(
        task,
        calib_model,
        calib_data_loader,
        val_data_loader,
    )


def test_dryrun_classification():
    dryrun_task(task="classification")


def test_dryrun_regression():
    dryrun_task(task="regression")
