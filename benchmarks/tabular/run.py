import logging
from fortuna.prob_model import ProbRegressor, FitConfig, FitMonitor, FitOptimizer, MAPPosteriorApproximator, ProbClassifier, CalibConfig, CalibOptimizer
from fortuna.model import MLP, ScalarHyperparameterModel
from fortuna.conformal import QuantileConformalRegressor, BinaryClassificationMulticalibrator, TopLabelMulticalibrator
from fortuna.metric.regression import rmse, picp
from fortuna.metric.classification import brier_score, expected_calibration_error, accuracy, maximum_calibration_error
from fortuna.data import DataLoader
import optax
import json
import numpy as np
from benchmarks.tabular.dataset import regression_datasets, classification_datasets, download_regression_dataset, download_classification_dataset, load_regression_dataset, load_classification_dataset
from time import time
from typing import Union
from sklearn.metrics import roc_auc_score, average_precision_score
import os
from tqdm import tqdm
from copy import deepcopy
from jax.nn import one_hot

TASKS = ["regression", "classification"]
DATA_DIR = "./datasets/"
COVERAGE_ERROR = 0.05
N_EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 2
TRAIN_LR = 1e-2
N_CALIB_EPOCHS = 300
CALIB_LR = 1e-1
BATCH_SIZE = 512
PROP_TRAIN = 0.5
PROP_VAL = 0.3
PROP_TEST = 0.2

assert PROP_TRAIN + PROP_VAL + PROP_TEST == 1


if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)


def get_nll(one_hot_targets, probs):
    _probs = one_hot_targets * probs
    return -np.sum(np.log(_probs[_probs > 0]))


def get_prob_model(task: str, data_loader: DataLoader) -> Union[ProbRegressor, ProbClassifier]:
    if task == "regression":
        prob_model = prob_model_cls(
            model=MLP(output_dim=output_dim),
            likelihood_log_variance_model=ScalarHyperparameterModel(output_dim, value=float(np.log(np.var(data_loader.to_array_targets())))),
            posterior_approximator=MAPPosteriorApproximator()
        )
    else:
        prob_model = prob_model_cls(
            model=MLP(output_dim=output_dim),
            posterior_approximator=MAPPosteriorApproximator()
        )
    return prob_model


all_status = {task: dict() for task in TASKS}
all_metrics = {task: dict() for task in TASKS}

for task in TASKS:
    if task == "regression":
        download_fn = download_regression_dataset
        load_fn = load_regression_dataset
        datasets = regression_datasets
        prob_model_cls = ProbRegressor
    elif task == "classification":
        download_fn = download_classification_dataset
        load_fn = load_classification_dataset
        datasets = classification_datasets
        prob_model_cls = ProbClassifier
    else:
        raise ValueError(f"`task={task}` not supported.")

    for dataset_name in tqdm(datasets, desc="Dataset"):
        print(dataset_name)
        # download and load data
        download_fn(dataset_name, DATA_DIR)
        train_data_loader, val_data_loader, test_data_loader = load_fn(
            dataset_name,
            DATA_DIR,
            shuffle_train=True,
            batch_size=BATCH_SIZE,
            prop_train=PROP_TRAIN,
            prop_val=PROP_VAL,
            prop_test=PROP_TEST
        )
        train_targets, val_targets, test_targets = train_data_loader.to_array_targets(), val_data_loader.to_array_targets(), test_data_loader.to_array_targets()

        if task == "classification":
            train_unique_targets, val_unique_targets, test_unique_targets = np.unique(train_targets), np.unique(val_targets), np.unique(test_targets)
            if (len(train_unique_targets) != len(val_unique_targets)) or (len(val_unique_targets) != len(test_unique_targets)):
                logging.warning(f"Skipping dataset {dataset_name} because the number of labels in train/val/test splits "
                                f"don't match.")
                continue
            if not (np.allclose(train_unique_targets, val_unique_targets) and np.allclose(val_unique_targets, test_unique_targets)):
                logging.warning(f"Skipping dataset {dataset_name} because the labels in train/val/test splits "
                                f"don't match.")
                continue
            if not np.allclose(train_unique_targets, np.arange(len(train_unique_targets))):
                logging.warning(f"Skipping dataset {dataset_name} because targets do not follow canonical indices.")
                continue
            if train_data_loader.num_unique_labels == 1:
                logging.warning(f"Skipping dataset {dataset_name} because the number of unique labels per split is 1.")
                continue

        # find output dimension
        if task == "regression":
            for batch_inputs, batch_targets in train_data_loader:
                output_dim = batch_targets.shape[-1]
                break
        else:
            output_dim = train_data_loader.num_unique_labels

        if task == "regression" and output_dim > 1:
            logging.warning(f"The dimension of the target variables in the {dataset_name} regression dataset "
                            f"is greater than 1. Skipped.")
            continue

        all_metrics[task][dataset_name] = dict(map=dict(), temp_scaling=dict())
        if task == "regression":
            all_metrics[task][dataset_name]["cqr"] = dict()
            all_metrics[task][dataset_name]["multicalibrate"] = dict()
            all_metrics[task][dataset_name]["temp_cqr"] = dict()
        else:
            all_metrics[task][dataset_name]["mc_conf"] = dict()
            all_metrics[task][dataset_name]["mc_prob"] = dict()
            all_metrics[task][dataset_name]["temp_mc_conf"] = dict()

        # define probabilistic model
        prob_model = get_prob_model(task, train_data_loader)

        # train the probabilistic regression
        time_init = time()
        all_status[task][dataset_name] = prob_model.train(
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            fit_config=FitConfig(
                monitor=FitMonitor(early_stopping_patience=EARLY_STOPPING_PATIENCE),
                optimizer=FitOptimizer(method=optax.adam(TRAIN_LR), n_epochs=N_EPOCHS),
            )
        )
        all_metrics[task][dataset_name]["map"]["time"] = time() - time_init

        # compute predictive statistics
        test_inputs_loader = test_data_loader.to_inputs_loader()

        if task == "classification":
            one_hot_test_targets = np.array(one_hot(test_targets, test_data_loader.num_unique_labels))

        test_means = prob_model.predictive.mean(inputs_loader=test_inputs_loader, n_posterior_samples=1)
        test_modes = prob_model.predictive.mode(inputs_loader=test_inputs_loader, n_posterior_samples=1)

        # compute metrics
        all_metrics[task][dataset_name]["map"]["nll"] = float(-prob_model.predictive.log_prob(data_loader=test_data_loader, n_posterior_samples=1).sum())

        if task == "regression":
            test_cred_intervals = prob_model.predictive.credible_interval(
                inputs_loader=test_inputs_loader, error=COVERAGE_ERROR
            )

            all_metrics[task][dataset_name]["temp_scaling"]["rmse"] = float(rmse(preds=test_modes, targets=test_targets))
            all_metrics[task][dataset_name]["map"]["picp"] = float(picp(lower_bounds=test_cred_intervals[:, 0], upper_bounds=test_cred_intervals[:, 1], targets=test_targets))
        else:
            all_metrics[task][dataset_name]["map"]["accuracy"] = float(accuracy(preds=test_modes, targets=test_targets))
            all_metrics[task][dataset_name]["map"]["mse"] = float(brier_score(probs=test_means, targets=test_targets))
            all_metrics[task][dataset_name]["map"]["ece"] = float(expected_calibration_error(preds=test_modes, probs=test_means, targets=test_targets))
            all_metrics[task][dataset_name]["map"]["mce"] = float(maximum_calibration_error(preds=test_modes, probs=test_means, targets=test_targets))
            all_metrics[task][dataset_name]["map"]["rocauc"] = roc_auc_score(y_true=one_hot_test_targets, y_score=test_means, average="macro")
            all_metrics[task][dataset_name]["map"]["prauc"] = average_precision_score(y_true=one_hot_test_targets, y_score=test_means, average="macro")

        val_inputs_loader = val_data_loader.to_inputs_loader()

        if task == "regression":
            # calibrate the credibility intervals
            val_cred_intervals = prob_model.predictive.credible_interval(
                inputs_loader=val_inputs_loader
            )

            time_init = time()
            test_conformal_intervals = QuantileConformalRegressor().conformal_interval(
                val_lower_bounds=val_cred_intervals[:, 0],
                val_upper_bounds=val_cred_intervals[:, 1],
                test_lower_bounds=test_cred_intervals[:, 0],
                test_upper_bounds=test_cred_intervals[:, 1],
                val_targets=val_targets,
                error=COVERAGE_ERROR,
            )
            all_metrics[task][dataset_name]["cqr"]["time"] = time() - time_init
            all_metrics[task][dataset_name]["cqr"]["picp"] = float(picp(lower_bounds=test_conformal_intervals[:, 0], upper_bounds=test_conformal_intervals[:, 1], targets=test_targets))
        else:
            val_modes = prob_model.predictive.mode(inputs_loader=val_inputs_loader, n_posterior_samples=1)
            val_means = prob_model.predictive.mean(inputs_loader=val_inputs_loader, n_posterior_samples=1)

            time_init = time()
            mc_conf = TopLabelMulticalibrator(n_classes=train_data_loader.num_unique_labels)
            test_mc_conf, mc_status = mc_conf.calibrate(
                targets=val_targets,
                probs=val_means,
                test_probs=test_means,
            )
            all_metrics[task][dataset_name]["mc_conf"]["time"] = time() - time_init

            all_metrics[task][dataset_name]["mc_conf"]["accuracy"] = float(accuracy(test_mc_conf.argmax(1), test_targets))
            all_metrics[task][dataset_name]["mc_conf"]["nll"] = float(get_nll(one_hot_test_targets, test_mc_conf))
            all_metrics[task][dataset_name]["mc_conf"]["mse"] = float(mc_conf.mean_squared_error(probs=test_mc_conf, targets=test_targets))
            all_metrics[task][dataset_name]["mc_conf"]["ece"] = float(expected_calibration_error(preds=test_modes, probs=test_mc_conf, targets=test_targets))
            all_metrics[task][dataset_name]["mc_conf"]["mce"] = float(maximum_calibration_error(preds=test_modes, probs=test_mc_conf, targets=test_targets))
            all_metrics[task][dataset_name]["mc_conf"]["rocauc"] = roc_auc_score(y_true=one_hot_test_targets, y_score=test_mc_conf, average="macro")
            all_metrics[task][dataset_name]["mc_conf"]["prauc"] = average_precision_score(y_true=one_hot_test_targets, y_score=test_mc_conf, average="macro")

            if output_dim == 2:
                time_init = time()
                mc_prob = BinaryClassificationMulticalibrator()
                test_mc_prob, mc_status = mc_prob.calibrate(
                    targets=val_targets,
                    probs=val_means[:, 1],
                    test_probs=test_means[:, 1],
                )
                probs = np.stack([1 - test_mc_prob, test_mc_prob], axis=1)
                all_metrics[task][dataset_name]["mc_prob"]["time"] = time() - time_init

                all_metrics[task][dataset_name]["mc_prob"]["accuracy"] = float(accuracy(probs.argmax(1), test_targets))
                all_metrics[task][dataset_name]["mc_prob"]["nll"] = float(get_nll(one_hot_test_targets, probs))
                all_metrics[task][dataset_name]["mc_prob"]["mse"] = float(mc_prob.mean_squared_error(probs=test_mc_prob, targets=test_targets))
                all_metrics[task][dataset_name]["mc_prob"]["ece"] = float(expected_calibration_error(preds=probs.argmax(1), probs=probs, targets=test_targets))
                all_metrics[task][dataset_name]["mc_prob"]["mce"] = float(maximum_calibration_error(preds=probs.argmax(1), probs=probs, targets=test_targets))
                all_metrics[task][dataset_name]["mc_prob"]["rocauc"] = roc_auc_score(y_true=one_hot_test_targets, y_score=probs, average="macro")
                all_metrics[task][dataset_name]["mc_prob"]["prauc"] = average_precision_score(y_true=one_hot_test_targets, y_score=probs, average="macro")

        temp_scaling_prob_model = get_prob_model(task, train_data_loader)
        temp_scaling_prob_model.posterior.state = deepcopy(prob_model.posterior.state)

        time_init = time()
        temp_scaling_status = temp_scaling_prob_model.calibrate(
            calib_data_loader=val_data_loader,
            calib_config=CalibConfig(
                optimizer=CalibOptimizer(n_epochs=N_CALIB_EPOCHS, method=optax.adam(CALIB_LR))
            )
        )
        all_metrics[task][dataset_name]["temp_scaling"]["time"] = time() - time_init

        test_temp_means = temp_scaling_prob_model.predictive.mean(inputs_loader=test_inputs_loader, n_posterior_samples=1)
        test_temp_modes = temp_scaling_prob_model.predictive.mode(inputs_loader=test_inputs_loader, n_posterior_samples=1)

        # compute metrics
        all_metrics[task][dataset_name]["temp_scaling"]["nll"] = float(-temp_scaling_prob_model.predictive.log_prob(data_loader=test_data_loader, n_posterior_samples=1).sum())

        if task == "regression":
            test_temp_cred_intervals = temp_scaling_prob_model.predictive.credible_interval(
                inputs_loader=test_inputs_loader, error=COVERAGE_ERROR
            )

            all_metrics[task][dataset_name]["temp_scaling"]["rmse"] = float(rmse(preds=test_temp_modes, targets=test_targets))
            all_metrics[task][dataset_name]["temp_scaling"]["picp"] = float(picp(lower_bounds=test_temp_cred_intervals[:, 0], upper_bounds=test_temp_cred_intervals[:, 1], targets=test_targets))
        else:
            all_metrics[task][dataset_name]["temp_scaling"]["accuracy"] = float(accuracy(preds=test_temp_modes, targets=test_targets))
            all_metrics[task][dataset_name]["temp_scaling"]["mse"] = float(brier_score(probs=test_temp_means, targets=test_targets))
            all_metrics[task][dataset_name]["temp_scaling"]["ece"] = float(expected_calibration_error(preds=test_temp_modes, probs=test_temp_means, targets=test_targets))
            all_metrics[task][dataset_name]["temp_scaling"]["mce"] = float(maximum_calibration_error(preds=test_temp_modes, probs=test_temp_means, targets=test_targets))
            all_metrics[task][dataset_name]["temp_scaling"]["rocauc"] = roc_auc_score(y_true=one_hot_test_targets, y_score=test_temp_means, average="macro")
            all_metrics[task][dataset_name]["temp_scaling"]["prauc"] = average_precision_score(y_true=one_hot_test_targets, y_score=test_temp_means, average="macro")

        if task == "regression":
            # calibrate the credibility intervals
            temp_scaling_val_cred_intervals = temp_scaling_prob_model.predictive.credible_interval(
                inputs_loader=val_inputs_loader
            )

            time_init = time()
            temp_scaling_test_conformal_intervals = QuantileConformalRegressor().conformal_interval(
                val_lower_bounds=temp_scaling_val_cred_intervals[:, 0],
                val_upper_bounds=temp_scaling_val_cred_intervals[:, 1],
                test_lower_bounds=test_temp_cred_intervals[:, 0],
                test_upper_bounds=test_temp_cred_intervals[:, 1],
                val_targets=val_targets,
                error=COVERAGE_ERROR,
            )
            all_metrics[task][dataset_name]["temp_cqr"]["time"] = time() - time_init
            all_metrics[task][dataset_name]["temp_cqr"]["picp"] = float(picp(lower_bounds=temp_scaling_test_conformal_intervals[:, 0], upper_bounds=temp_scaling_test_conformal_intervals[:, 1], targets=test_targets))
        else:
            temp_val_modes = temp_scaling_prob_model.predictive.mode(inputs_loader=val_inputs_loader, n_posterior_samples=1)
            temp_val_means = temp_scaling_prob_model.predictive.mean(inputs_loader=val_inputs_loader, n_posterior_samples=1)

            time_init = time()
            temp_mc_conf = TopLabelMulticalibrator(n_classes=train_data_loader.num_unique_labels)
            temp_test_mc_conf, mc_status = temp_mc_conf.calibrate(
                targets=val_targets,
                probs=temp_val_means,
                test_probs=test_means,
            )
            all_metrics[task][dataset_name]["temp_mc_conf"]["time"] = time() - time_init

            all_metrics[task][dataset_name]["temp_mc_conf"]["accuracy"] = float(accuracy(temp_test_mc_conf.argmax(1), test_targets))
            all_metrics[task][dataset_name]["temp_mc_conf"]["nll"] = float(get_nll(one_hot_test_targets, temp_test_mc_conf))
            all_metrics[task][dataset_name]["temp_mc_conf"]["mse"] = float(temp_mc_conf.mean_squared_error(probs=temp_test_mc_conf, targets=test_targets))
            all_metrics[task][dataset_name]["temp_mc_conf"]["ece"] = float(expected_calibration_error(preds=test_temp_modes, probs=temp_test_mc_conf, targets=test_targets))
            all_metrics[task][dataset_name]["temp_mc_conf"]["mce"] = float(maximum_calibration_error(preds=test_temp_modes, probs=temp_test_mc_conf, targets=test_targets))
            all_metrics[task][dataset_name]["temp_mc_conf"]["rocauc"] = roc_auc_score(y_true=one_hot_test_targets, y_score=temp_test_mc_conf, average="macro")
            all_metrics[task][dataset_name]["temp_mc_conf"]["prauc"] = average_precision_score(y_true=one_hot_test_targets, y_score=temp_test_mc_conf, average="macro")

with open('tabular_results.json', 'w') as fp:
    json.dump(all_metrics, fp)
