# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
# ---

# # How to calibrate sub-networks of pre-trained models

# Fortuna's calibration model offer a simple interface to train or fine-tune a deep learning model. The user is free to choose a custom calibration loss, select an optimizer, decide which parameters to train or fine-tune, monitor calibration metrics, and more. In this example, we look at some of its main functionalities.

# ## Download, split and process the data

# First, we download the data from TensorFlow, and split them into training, validation and test set.

# +
import tensorflow as tf
import tensorflow_datasets as tfds


def download(split_range, shuffle=False):
    ds = tfds.load(
        name="mnist",
        split=f"train[{split_range}]",
        as_supervised=True,
        shuffle_files=True,
    ).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    if shuffle:
        ds = ds.shuffle(10, reshuffle_each_iteration=True)
    return ds.batch(128).prefetch(1)


train_data_loader, val_data_loader, test_data_loader = (
    download(":80%", shuffle=True),
    download("80%:90%"),
    download("90%:"),
)
# -

# We then convert the data loaders into something that Fortuna can work with.

# +
from fortuna.data import DataLoader

train_data_loader = DataLoader.from_tensorflow_data_loader(train_data_loader)
val_data_loader = DataLoader.from_tensorflow_data_loader(val_data_loader)
test_data_loader = DataLoader.from_tensorflow_data_loader(test_data_loader)
# -

# ## Define and calibrate the calibration model

# We now introduce `CalibClassifier`, i.e. Fortuna's calibration classifier purposed to obtain calibrated predictions. 

from fortuna.model import LeNet5
from fortuna.calib_model import CalibClassifier
calib_model = CalibClassifier(model=LeNet5(output_dim=10))

# Let's calibrate this model! At first, we will run the calibration from scratch, thus this can just be seen as training the model. By default, the calibration exploits a focal loss [Mukhoti et al., 2020](https://proceedings.neurips.cc/paper/2020/file/aeb7b30ef1d024a76f21a1d40e30c302-Paper.pdf) with `gamma=2.`, but other custom losses may be used. During the calibration, we will enable early stopping and monitor accuracy and Brier score - we will just have to adust the signature to make sure it is compatible with one that the `CalibClassifier` expects. 

# +
from fortuna.calib_model import Config, Monitor
from fortuna.metric.classification import brier_score, accuracy

def brier(preds, uncertainties, targets): 
    return brier_score(uncertainties, targets)

def acc(preds, uncertainties, targets): 
    return accuracy(preds, targets)

status = calib_model.calibrate(
    train_data_loader, 
    val_data_loader=val_data_loader,
    config=Config(monitor=Monitor(early_stopping_patience=2, metrics=(brier, acc)))
)
# -

# ## Expected calibration error and reliability plot

# In one go, let's compute the Expected Calibration Error (ECE) and draw a reliability plot! To obtain this, we need to first obtain predictions and their probabilities over the test data set.

test_inputs_loader = test_data_loader.to_inputs_loader()
preds = calib_model.predictive.mode(test_inputs_loader)
probs = calib_model.predictive.mean(test_inputs_loader)

from fortuna.metric.classification import expected_calibration_error
test_targets = test_data_loader.to_array_targets()
ece = expected_calibration_error(preds, probs, test_targets, plot=True, plot_options=dict(figsize=(6, 2)))
print(f"ECE: {ece}.")

# Expect for very low confidence, where usually we do not have enough information to obtain a reliable ECE, the model seems well calibrated, since the difference between confidence and accuracy is close to 0 for most confidence bins.

# ## Calibrate only a subset of model parameters

# With the only purpose of demonstrating the functionality, let us now show how you can start from a pre-trained model and fine-tune only a subset of model parameters, perhaps with the purpose of achieving better calibration. 
#
# All you need to do is pass `freeze_map` to the `Optimizer` in the `Config` object, and declare which parameters you want to be `trainable` and which `frozen`. In this example, the parameters of the LeNet-5 model in use are internally organized in a deep feature extractor sub-network (`dfe_subnet`) and an output sub-network. Then we simply freeze `dfe_subnet` and let the model fine-tune only the output layer.
#
# In order to start from the pre-trained state, we simply enable the flag `start_from_current_state` in the `Checkpointer`.

from fortuna.calib_model import Optimizer, Checkpointer
status = calib_model.calibrate(
    train_data_loader, 
    val_data_loader=val_data_loader,
    config=Config(
        monitor=Monitor(early_stopping_patience=2, metrics=(brier, acc)),
        checkpointer=Checkpointer(start_from_current_state=True),
        optimizer=Optimizer(freeze_map=lambda path, v: 'frozen' if "dfe_subnet" in path else 'trainable')
    )
)
