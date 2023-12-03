from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from fortuna.conformal import BinaryClassificationMulticalibrator
from fortuna.metric.classification import accuracy

data = load_breast_cancer()
inputs = data.data
targets = data.target
train_inputs, test_inputs, train_targets, test_targets = train_test_split(
    inputs, targets, test_size=0.3, random_state=1
)
train_size = int(len(train_inputs) * 0.5)
train_inputs, calib_inputs = train_inputs[:train_size], train_inputs[train_size:]
train_targets, calib_targets = train_targets[:train_size], train_targets[train_size:]

calib_size = calib_targets.shape[0]

model = MLPClassifier(random_state=42)
model.fit(train_inputs, train_targets)

calib_preds = model.predict(calib_inputs)
calib_probs = (
    model.predict_proba(calib_inputs)
    if hasattr(model, "predict_proba")
    else model._predict_proba_lr(calib_inputs)
)
test_preds = model.predict(test_inputs)
test_probs = (
    model.predict_proba(test_inputs)
    if hasattr(model, "predict_proba")
    else model._predict_proba_lr(test_inputs)
)

mc = BinaryClassificationMulticalibrator()
mc_test_probs1, status = mc.calibrate(
    targets=calib_targets,
    probs=calib_probs[:, 1],
    test_probs=test_probs[:, 1],
    patch_type="multiplicative",
)

mc_calib_probs1 = mc.apply_patches(probs=calib_probs[:, 1])

print(
    f"Calib accuracy pre/post calibration: {float(accuracy(calib_preds, calib_targets)), float(accuracy(mc_calib_probs1 > 0.5, calib_targets))}"
)
print(
    f"Test accuracy pre/post calibration: {float(accuracy(test_preds, test_targets)), float(accuracy(mc_test_probs1 > 0.5, test_targets))}"
)
print()
print(
    f"Calib MSE pre/post calibration: {float(mc.mean_squared_error(calib_probs[:, 1], calib_targets)), float(mc.mean_squared_error(mc_calib_probs1, calib_targets))}"
)
print(
    f"Test MSE pre/post calibration: {float(mc.mean_squared_error(test_probs[:, 1], test_targets)), float(mc.mean_squared_error(mc_test_probs1, test_targets))}"
)
print()

from fortuna.conformal import OneShotBinaryClassificationMulticalibrator

osmc = OneShotBinaryClassificationMulticalibrator()
osmc_test_probs1 = osmc.calibrate(
    targets=calib_targets, probs=calib_probs[:, 1], test_probs=test_probs[:, 1]
)

osmc_calib_probs1 = osmc.apply_patches(probs=calib_probs[:, 1])

print(
    f"Calib accuracy pre/post one-shot calibration: {float(accuracy(calib_preds, calib_targets)), float(accuracy(osmc_calib_probs1 > 0.5, calib_targets))}"
)
print(
    f"Test accuracy pre/post one-shot calibration: {float(accuracy(test_preds, test_targets)), float(accuracy(osmc_test_probs1 > 0.5, test_targets))}"
)
print()
print(
    f"Calib MSE pre/post one-shot calibration: {float(mc.mean_squared_error(calib_probs[:, 1], calib_targets)), float(mc.mean_squared_error(osmc_calib_probs1, calib_targets))}"
)
print(
    f"Test MSE pre/post one-shot calibration: {float(mc.mean_squared_error(test_probs[:, 1], test_targets)), float(mc.mean_squared_error(osmc_test_probs1, test_targets))}"
)
