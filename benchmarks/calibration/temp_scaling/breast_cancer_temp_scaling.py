import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tabulate import tabulate

from fortuna.calibration import (
    BiasBinaryClassificationTemperatureScaling,
    ClassificationTemperatureScaling,
    CrossEntropyBinaryClassificationTemperatureScaling,
    F1BinaryClassificationTemperatureScaling,
    MSEBinaryClassificationTemperatureScaling,
)
from fortuna.metric.classification import (
    brier_score,
    expected_calibration_error,
)


def precision(preds: np.ndarray, targets: np.ndarray) -> float:
    n_pos_preds = np.sum(preds)
    return float(np.sum(preds * targets) / n_pos_preds)


def recall(preds: np.ndarray, targets: np.ndarray) -> float:
    n_pos_targets = np.sum(targets)
    return float(np.sum(preds * targets) / n_pos_targets) if n_pos_targets > 0 else 0.0


def f1(preds: np.ndarray, targets: np.ndarray) -> float:
    prec = precision(preds, targets)
    rec = recall(preds, targets)
    return float(2 * prec * rec / (prec + rec))


def binary_cross_entropy(probs: np.array, targets: np.ndarray) -> float:
    new_probs = (1 - 1e-6) * (1e-6 + probs)
    return float(
        -np.mean(targets * np.log(new_probs) + (1 - targets) * np.log(1 - new_probs))
    )


if __name__ == "__main__":
    data = load_breast_cancer()
    inputs = data.data
    targets = data.target
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(
        inputs, targets, test_size=0.3, random_state=1
    )
    train_size = int(len(train_inputs) * 0.5)
    train_inputs, calib_inputs = train_inputs[:train_size], train_inputs[train_size:]
    train_targets, calib_targets = (
        train_targets[:train_size],
        train_targets[train_size:],
    )

    calib_size = calib_targets.shape[0]

    model = MLPClassifier(random_state=42)
    model.fit(train_inputs, train_targets)

    calib_probs = model.predict_proba(calib_inputs)[:, 1]
    test_preds = model.predict(test_inputs)
    test_probs = model.predict_proba(test_inputs)[:, 1]
    before_brier_score = brier_score(test_probs, test_targets)
    before_ece = expected_calibration_error(
        probs=np.stack((1 - test_probs, test_probs), axis=1),
        preds=test_preds,
        targets=test_targets,
    )
    before_prec = precision(test_preds, test_targets)
    before_rec = recall(test_preds, test_targets)
    before_f1 = f1(test_probs, test_targets)
    before_ce = binary_cross_entropy(test_probs, test_targets)

    mse_temp_scaler = MSEBinaryClassificationTemperatureScaling()
    mse_temp_scaler.fit(probs=calib_probs, targets=calib_targets)
    mse_temp_scaled_test_probs = mse_temp_scaler.predict_proba(probs=test_probs)
    mse_temp_scaled_test_preds = mse_temp_scaler.predict(probs=test_probs)
    mse_temp_scaled_brier_score = brier_score(mse_temp_scaled_test_probs, test_targets)
    mse_temp_scaled_ece = expected_calibration_error(
        probs=np.stack(
            (1 - mse_temp_scaled_test_probs, mse_temp_scaled_test_probs), axis=1
        ),
        preds=mse_temp_scaled_test_preds,
        targets=test_targets,
    )
    mse_temp_scaled_prec = precision(mse_temp_scaled_test_preds, test_targets)
    mse_temp_scaled_rec = recall(mse_temp_scaled_test_preds, test_targets)
    mse_temp_scaled_f1 = f1(mse_temp_scaled_test_preds, test_targets)
    mse_temp_scaled_ce = binary_cross_entropy(mse_temp_scaled_test_probs, test_targets)

    ce_temp_scaler = CrossEntropyBinaryClassificationTemperatureScaling()
    ce_temp_scaler.fit(probs=calib_probs, targets=calib_targets)
    ce_temp_scaled_test_probs = ce_temp_scaler.predict_proba(probs=test_probs)
    ce_temp_scaled_test_preds = ce_temp_scaler.predict(probs=test_probs)
    ce_temp_scaled_brier_score = brier_score(ce_temp_scaled_test_probs, test_targets)
    ce_temp_scaled_ece = expected_calibration_error(
        probs=np.stack(
            (1 - ce_temp_scaled_test_probs, ce_temp_scaled_test_probs), axis=1
        ),
        preds=ce_temp_scaled_test_preds,
        targets=test_targets,
    )
    ce_temp_scaled_prec = precision(ce_temp_scaled_test_preds, test_targets)
    ce_temp_scaled_rec = recall(ce_temp_scaled_test_preds, test_targets)
    ce_temp_scaled_f1 = f1(ce_temp_scaled_test_preds, test_targets)
    ce_temp_scaled_ce = binary_cross_entropy(ce_temp_scaled_test_probs, test_targets)

    bias_temp_scaler = BiasBinaryClassificationTemperatureScaling()
    bias_temp_scaler.fit(probs=calib_probs, targets=calib_targets)
    bias_temp_scaled_test_probs = bias_temp_scaler.predict_proba(probs=test_probs)
    bias_temp_scaled_test_preds = bias_temp_scaler.predict(probs=test_probs)
    bias_temp_scaled_brier_score = brier_score(
        bias_temp_scaled_test_probs, test_targets
    )
    bias_temp_scaled_ece = expected_calibration_error(
        probs=np.stack(
            (1 - bias_temp_scaled_test_probs, bias_temp_scaled_test_probs), axis=1
        ),
        preds=bias_temp_scaled_test_preds,
        targets=test_targets,
    )
    bias_temp_scaled_prec = precision(bias_temp_scaled_test_preds, test_targets)
    bias_temp_scaled_rec = recall(bias_temp_scaled_test_preds, test_targets)
    bias_temp_scaled_f1 = f1(bias_temp_scaled_test_preds, test_targets)
    bias_temp_scaled_ce = binary_cross_entropy(
        bias_temp_scaled_test_probs, test_targets
    )

    f1_temp_scaler = F1BinaryClassificationTemperatureScaling()
    f1_temp_scaler.fit(probs=calib_probs, targets=calib_targets, threshold=0.5)
    f1_temp_scaled_test_probs = f1_temp_scaler.predict_proba(probs=test_probs)
    f1_temp_scaled_test_preds = f1_temp_scaler.predict(probs=test_probs)
    f1_temp_scaled_brier_score = brier_score(f1_temp_scaled_test_probs, test_targets)
    f1_temp_scaled_ece = expected_calibration_error(
        probs=np.stack(
            (1 - f1_temp_scaled_test_probs, f1_temp_scaled_test_probs), axis=1
        ),
        preds=f1_temp_scaled_test_preds,
        targets=test_targets,
    )
    f1_temp_scaled_prec = precision(f1_temp_scaled_test_preds, test_targets)
    f1_temp_scaled_rec = recall(f1_temp_scaled_test_preds, test_targets)
    f1_temp_scaled_f1 = f1(f1_temp_scaled_test_preds, test_targets)
    f1_temp_scaled_ce = binary_cross_entropy(f1_temp_scaled_test_probs, test_targets)

    logits_temp_scaler = ClassificationTemperatureScaling()
    status = logits_temp_scaler.fit(
        probs=np.stack((1 - calib_probs, calib_probs), axis=1), targets=calib_targets
    )
    logits_temp_scaled_test_probs = logits_temp_scaler.predict_proba(
        probs=np.stack((1 - test_probs, test_probs), axis=1)
    )[:, 1]
    logits_temp_scaled_test_preds = logits_temp_scaler.predict(
        probs=np.stack((1 - test_probs, test_probs), axis=1)
    )
    logits_temp_scaled_brier_score = brier_score(
        logits_temp_scaled_test_probs, test_targets
    )
    logits_temp_scaled_ece = expected_calibration_error(
        probs=np.stack(
            (1 - logits_temp_scaled_test_probs, logits_temp_scaled_test_probs), axis=1
        ),
        preds=logits_temp_scaled_test_preds,
        targets=test_targets,
    )
    logits_temp_scaled_prec = precision(logits_temp_scaled_test_preds, test_targets)
    logits_temp_scaled_rec = recall(logits_temp_scaled_test_preds, test_targets)
    logits_temp_scaled_f1 = f1(logits_temp_scaled_test_preds, test_targets)
    logits_temp_scaled_ce = binary_cross_entropy(
        logits_temp_scaled_test_probs, test_targets
    )

    print(
        tabulate(
            [
                [
                    "Before calibration",
                    before_brier_score,
                    before_ce,
                    before_ece,
                    before_prec,
                    before_rec,
                    before_f1,
                ],
                [
                    "MSE binary temperature scaling",
                    mse_temp_scaled_brier_score,
                    mse_temp_scaled_ce,
                    mse_temp_scaled_ece,
                    mse_temp_scaled_prec,
                    mse_temp_scaled_rec,
                    mse_temp_scaled_f1,
                ],
                [
                    "Cross-Entropy binary temperature scaling",
                    ce_temp_scaled_brier_score,
                    ce_temp_scaled_ce,
                    ce_temp_scaled_ece,
                    ce_temp_scaled_prec,
                    ce_temp_scaled_rec,
                    ce_temp_scaled_f1,
                ],
                [
                    "Bias binary temperature scaling",
                    bias_temp_scaled_brier_score,
                    bias_temp_scaled_ce,
                    bias_temp_scaled_ece,
                    bias_temp_scaled_prec,
                    bias_temp_scaled_rec,
                    bias_temp_scaled_f1,
                ],
                [
                    "F1 binary temperature scaling",
                    f1_temp_scaled_brier_score,
                    f1_temp_scaled_ce,
                    f1_temp_scaled_ece,
                    f1_temp_scaled_prec,
                    f1_temp_scaled_rec,
                    f1_temp_scaled_f1,
                ],
                [
                    "Logits temperature scaling",
                    logits_temp_scaled_brier_score,
                    logits_temp_scaled_ce,
                    logits_temp_scaled_ece,
                    logits_temp_scaled_prec,
                    logits_temp_scaled_rec,
                    logits_temp_scaled_f1,
                ],
            ],
            headers=[
                "",
                "Brier score",
                "Binary cross-entropy",
                "ECE",
                "Precision",
                "Recall",
                "F1",
            ],
            tablefmt="rounded_outline",
        )
    )

    print(
        tabulate(
            [
                ["MSE binary temperature scaling", mse_temp_scaler.temperature],
                [
                    "Cross-Entropy binary temperature scaling",
                    ce_temp_scaler.temperature,
                ],
                ["Bias binary temperature scaling", bias_temp_scaler.temperature],
                [
                    "Precision-recall binary temperature scaling",
                    f1_temp_scaler.temperature,
                ],
                ["Logits temperature scaling", logits_temp_scaler.temperature],
            ],
            headers=["", "temperature"],
            tablefmt="rounded_outline",
        )
    )
