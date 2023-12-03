import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from fortuna.conformal import (
    OneShotTopLabelMulticalibrator,
    TopLabelMulticalibrator,
)
from fortuna.data import DataLoader
from fortuna.metric.classification import (
    accuracy,
    expected_calibration_error,
)
from fortuna.model import LeNet5
from fortuna.prob_model import (
    FitConfig,
    FitMonitor,
    FitOptimizer,
    MAPPosteriorApproximator,
    ProbClassifier,
)


def download(split_range, shuffle=False):
    ds = tfds.load(
        name="MNIST",
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

train_data_loader = DataLoader.from_tensorflow_data_loader(train_data_loader)
val_data_loader = DataLoader.from_tensorflow_data_loader(val_data_loader)
test_data_loader = DataLoader.from_tensorflow_data_loader(test_data_loader)
val_inputs_loader = val_data_loader.to_inputs_loader()
test_inputs_loader = test_data_loader.to_inputs_loader()
val_targets = val_data_loader.to_array_targets()
test_targets = test_data_loader.to_array_targets()

output_dim = 10
prob_model = ProbClassifier(
    model=LeNet5(output_dim=output_dim),
    posterior_approximator=MAPPosteriorApproximator(),
)

status = prob_model.train(
    train_data_loader=train_data_loader,
    val_data_loader=val_data_loader,
    fit_config=FitConfig(
        optimizer=FitOptimizer(), monitor=FitMonitor(early_stopping_patience=2)
    ),
)

test_means = prob_model.predictive.mean(inputs_loader=test_inputs_loader)
test_modes = prob_model.predictive.mode(
    inputs_loader=test_inputs_loader, means=test_means
)

val_means = prob_model.predictive.mean(inputs_loader=val_inputs_loader)
val_modes = prob_model.predictive.mode(inputs_loader=val_inputs_loader, means=val_means)

groups = []
for i in range(10):
    groups.append((val_means.max(1) > 0.1 * i) * (val_means.max(1) <= 0.1 * (i + 1)))
groups = np.stack(groups, axis=1)
test_groups = []
for i in range(10):
    test_groups.append(
        (test_means.max(1) > 0.1 * i) * (test_means.max(1) <= 0.1 * (i + 1))
    )
test_groups = np.stack(test_groups, axis=1)

mc = TopLabelMulticalibrator(n_classes=10)
mc_calib_test_probs, status = mc.calibrate(
    targets=val_targets,
    probs=val_means,
    groups=groups,
    test_probs=test_means,
    test_groups=test_groups,
    patch_type="multiplicative",
)
mc_calib_val_probs = mc.apply_patches(probs=val_means, groups=groups)

print(
    f"MSE on calibration data before calibration: {mc.mean_squared_error(val_means, val_targets)}"
)
print(
    f"MSE on calibration data after top-label multicalibration: {mc.mean_squared_error(mc_calib_val_probs, val_targets)}"
)
print()
print(
    f"MSE on test data before calibration: {mc.mean_squared_error(test_means, test_targets)}"
)
print(
    f"MSE on test data after top-label multicalibration: {mc.mean_squared_error(mc_calib_test_probs, test_targets)}"
)
print()
print(
    f"ECE on calibration data before calibration: {expected_calibration_error(preds=val_modes, probs=val_means, targets=val_targets)}"
)
print(
    f"ECE on calibration data after top-label multicalibration: {expected_calibration_error(preds=mc_calib_val_probs.argmax(1), probs=mc_calib_val_probs, targets=val_targets)}"
)
print()
print(
    f"ECE on test data before calibration: {expected_calibration_error(preds=test_modes, probs=test_means, targets=test_targets)}"
)
print(
    f"ECE on test data after top-label multicalibration: {expected_calibration_error(preds=mc_calib_test_probs.argmax(1), probs=mc_calib_test_probs, targets=test_targets)}"
)
print()
print(
    f"Accuracy on calibration data before top-label calibration: {accuracy(val_modes, val_targets)}"
)
print(
    f"Accuracy on calibration data after top-label multicalibration: {accuracy(mc_calib_val_probs.argmax(1), val_targets)}"
)
print()
print(f"Accuracy on test data before calibration: {accuracy(test_modes, test_targets)}")
print(
    f"Accuracy on test data after top-label multicalibration: {accuracy(mc_calib_test_probs.argmax(1), test_targets)}"
)
