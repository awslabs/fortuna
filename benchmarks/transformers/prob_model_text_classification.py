# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
# import tensorflow as tf
# tf.config.experimental.set_visible_devices([], "GPU")
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import copy
import json
import logging
import os
import pathlib
import tarfile
from typing import (
    List,
    Optional,
    Union,
)

from datasets import (
    DatasetDict,
    load_dataset,
)
import jax.numpy as jnp
import jax.random
import optax
from transformers import (
    AutoTokenizer,
    FlaxAutoModelForSequenceClassification,
)

from fortuna.data.dataset.huggingface_datasets import (
    HuggingFaceSequenceClassificationDataset,
)
from fortuna.metric.classification import (
    accuracy,
    expected_calibration_error,
)
from fortuna.model_editor import ProbitModelEditor
from fortuna.prob_model import (
    ADVIPosteriorApproximator,
    DeepEnsemblePosteriorApproximator,
    FitCheckpointer,
    FitConfig,
    FitMonitor,
    FitOptimizer,
    FitProcessor,
    LaplacePosteriorApproximator,
    MAPPosteriorApproximator,
    ProbClassifier,
    SGHMCPosteriorApproximator,
    SNGPPosteriorApproximator,
    SWAGPosteriorApproximator,
)
from fortuna.prob_model.fit_config.hyperparameters import FitHyperparameters
from fortuna.prob_model.posterior.posterior_approximations import (
    ADVI_NAME,
    DEEP_ENSEMBLE_NAME,
    LAPLACE_NAME,
    MAP_NAME,
    SGHMC_NAME,
    SNGP_NAME,
    SWAG_NAME,
)
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_preconditioner import (
    identity_preconditioner,
    rmsprop_preconditioner,
)
from fortuna.prob_model.posterior.sgmcmc.sgmcmc_step_schedule import (
    constant_schedule,
    cosine_schedule,
    polynomial_schedule,
)
from fortuna.prob_model.posterior.sngp.sngp_callback import ResetCovarianceCallback
from fortuna.prob_model.posterior.sngp.transformers import (
    FlaxAutoSNGPModelForSequenceClassification,
)
from fortuna.prob_model.prior import IsotropicGaussianPrior
from fortuna.utils.optimizer import (
    decay_mask_without_layer_norm_fn,
    linear_scheduler_with_warmup,
)

LAST_LAYER_APPROXIMATORS = [LAPLACE_NAME, ADVI_NAME, SGHMC_NAME]

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def strbool(s: Union[str, int, bool]) -> bool:
    if isinstance(s, (int, bool)):
        return s
    return s.lower() not in ("false", "f", "n", "0")


def str2list(s: str) -> Optional[List[str]]:
    if s is None:
        return None
    return s.split(",")


def unpack_model_tar(model_ckpt_path: pathlib.Path) -> pathlib.Path:
    """
    Untar a tar.gz object

    Args:
        model_data_dir (str): a local file system path that points to the `tar.gz` archive
        containing the model ckpts.
    Returns:
        model_dir (str): the directory that contains the uncompress model checkpoint files
    """
    model_dir = model_ckpt_path.parent
    # untar the model
    tar = tarfile.open(str(model_ckpt_path))
    tar.extractall(model_dir)
    tar.close()
    return model_dir


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description="Transformers fine tuning")
    # channels
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")
    )
    parser.add_argument(
        "--load-model-dir", type=str, default=os.environ.get("SM_CHANNEL_LOADMODEL")
    )
    parser.add_argument(
        "--train-data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN")
    )
    parser.add_argument(
        "--validation-data-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_VALIDATION"),
    )
    parser.add_argument(
        "--test-data-dir", type=str, default=os.environ.get("SM_CHANNEL_TEST")
    )

    # general
    parser.add_argument("--job_type", type=str, default="train_and_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_jit", type=strbool, default=False)
    parser.add_argument("--devices", type=int, default=-1)
    # data
    parser.add_argument("--dataset_name", type=str, default="imdb")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--text_columns", type=str2list, default=("text",))
    parser.add_argument("--target_column", type=str, nargs="+", default="label")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument(
        "--validation_split", type=str, default="test[:25%]+test[-25%:]"
    )
    parser.add_argument("--test_split", type=str, default="test[25%:75%]")
    parser.add_argument("--ood_dataset_name", type=str, default="yelp_polarity")
    parser.add_argument("--ood_task_name", type=str, default=None)
    parser.add_argument("--ood_text_columns", type=str2list, default=None)
    parser.add_argument("--ood_target_column", type=str, default=None)
    parser.add_argument("--ood_test_split", type=str, default="test[:50%]")
    # trainer/model
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-cased")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--tokenizer_max_length", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", type=strbool, default=False)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--half_precision", type=strbool, default=False)
    # posterior approximation
    parser.add_argument("--posterior_approximator_name", type=str, default="map")
    parser.add_argument("--last_layer_only", type=strbool, default=False)
    parser.add_argument("--prior_log_var", type=float, default=20.0)
    parser.add_argument("--n_posterior_samples", type=int, default=30)
    # posterior approximation - sngp only
    parser.add_argument(
        "--sngp_mean_field_factor",
        type=float,
        default=1.0,
        help="The scale factor for mean-field approximation, used to adjust (at inference time) the influence of posterior variance in posterior mean approximation.",
    )
    parser.add_argument(
        "--sngp_spectral_norm_bound",
        type=float,
        default=0.95,
        help="Multiplicative constant to threshold the normalization.Usually under normalization, the singular value will converge to this value.",
    )
    # posterior approximation - swag only
    parser.add_argument(
        "--swag_rank",
        type=float,
        default=2,
        help="SWAG approximates the posterior with a Gaussian distribution. The Gaussian's covariance matrix is formed by"
        " a diagonal matrix, and a low-rank empirical approximation. This argument defines the rank of the low-rank"
        "empirical covariance approximation. It must be at least 2.",
    )
    # posterior approximation - sgmcmc only
    parser.add_argument("--sgmcmc_n_thinning", type=int, default=100)
    parser.add_argument("--sgmcmc_burnin_length", type=int, default=0)
    parser.add_argument("--sgmcmc_step_schedule", type=str, default="constant")
    parser.add_argument("--sgmcmc_init_step_size", type=float, default=1e-5)
    parser.add_argument("--sgmcmc_polynomial_schedule_a", type=float, default=1)
    parser.add_argument("--sgmcmc_polynomial_schedule_b", type=float, default=1)
    parser.add_argument("--sgmcmc_polynomial_schedule_gamma", type=float, default=0.55)
    parser.add_argument("--sgmcmc_preconditioner", type=strbool, default=False)
    parser.add_argument("--sghmc_momentum_decay", type=float, default=0.01)
    # model editor
    parser.add_argument("--enable_probit_model_editor", type=strbool, default=False)
    parser.add_argument("--probit_init_log_var", type=float, default=-5)
    parser.add_argument("--probit_stop_gradient", type=strbool, default=False)
    parser.add_argument("--probit_last_layer_only", type=strbool, default=False)
    # optimizer
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--adam_b1", type=float, default=0.9)
    parser.add_argument("--adam_b2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    # checkpoint
    parser.add_argument("--keep_top_n_checkpoints", type=int, default=1)
    parser.add_argument("--save_every_n_steps", type=int, default=2000)
    parser.add_argument("--early_stopping_mode", type=str, default="min")
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument(
        "--early_stopping_monitor", type=str, default="validation_loss_epoch"
    )

    args = parser.parse_args()
    rng = jax.random.PRNGKey(args.seed)

    try:
        logger.info(list(pathlib.Path(args.load_model_dir).rglob("*")))
        restore_checkpoint_path = unpack_model_tar(
            list(pathlib.Path(args.load_model_dir).rglob("*"))[0]
        )
        logger.info(list(pathlib.Path(restore_checkpoint_path).rglob("*")))
    except:
        logger.info("No checkpoint to restore")
        restore_checkpoint_path = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    ### DATASET
    datasets = DatasetDict(
        {
            "train": load_dataset(
                args.dataset_name, name=args.task_name, split=args.train_split
            ),
            "validation": load_dataset(
                args.dataset_name, name=args.task_name, split=args.validation_split
            ),
            "test": load_dataset(
                args.dataset_name, name=args.task_name, split=args.test_split
            ),
        }
    )
    for k, v in datasets.items():
        unique_labels_set = len(v.unique("label"))
        if unique_labels_set != args.num_labels:
            raise ValueError(
                f"the {k} set has {unique_labels_set} unique labels but was expecting {args.num_labels}."
            )

    hf_dataset = HuggingFaceSequenceClassificationDataset(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=args.tokenizer_max_length,
        num_unique_labels=args.num_labels,
    )
    datasets = hf_dataset.get_tokenized_datasets(
        datasets,
        text_columns=args.text_columns
        if not isinstance(args.text_columns, str)
        else (args.text_columns,),
        target_column=args.target_column,
    )
    train_data_loader = hf_dataset.get_data_loader(
        datasets["train"],
        per_device_batch_size=args.per_device_train_batch_size,
        shuffle=True,
        drop_last=True,
        rng=rng,
    )
    val_data_loader = hf_dataset.get_data_loader(
        datasets["validation"],
        per_device_batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        drop_last=False,
        rng=rng,
    )
    test_data_loader = hf_dataset.get_data_loader(
        datasets["test"],
        per_device_batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        drop_last=False,
        rng=rng,
        verbose=True,
    )

    total_train_batch_size = args.per_device_train_batch_size * jax.local_device_count()
    logger.info(
        f"Dataset stats: \n"
        f"total_train_batch_size={total_train_batch_size}, "
        f"n_samples={train_data_loader.size}, "
        f"steps_per_epoch={train_data_loader.size // total_train_batch_size}"
    )

    ### MODEL
    model_kwargs = {"num_labels": args.num_labels}
    if args.half_precision:
        model_kwargs.update({"dtype": jnp.bfloat16})
    try:
        if args.posterior_approximator_name == SNGP_NAME:
            model = FlaxAutoSNGPModelForSequenceClassification.from_pretrained(
                args.model_name_or_path,
                **model_kwargs,
                spectral_norm_bound=args.sngp_spectral_norm_bound,
            )
        else:
            model = FlaxAutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path, **model_kwargs
            )
    except OSError:
        model = FlaxAutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path, from_pt=True, **model_kwargs
        )
    if args.gradient_checkpointing:
        try:
            model.enable_gradient_checkpointing()
        except NotImplementedError as e:
            logger.warning(e.args[0])

    optimizer = optax.adamw(
        learning_rate=linear_scheduler_with_warmup(
            learning_rate=args.learning_rate,
            num_inputs_train=len(datasets["train"]),
            train_total_batch_size=total_train_batch_size,
            num_train_epochs=args.num_train_epochs,
            num_warmup_steps=args.num_warmup_steps,
        ),
        b1=args.adam_b1,
        b2=args.adam_b2,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        mask=decay_mask_without_layer_norm_fn,
    )

    if args.sgmcmc_step_schedule == "constant":
        sgmcmc_step_schedule = constant_schedule(
            args.sgmcmc_init_step_size,
        )
    elif args.sgmcmc_step_schedule == "cosine":
        steps_per_epoch = train_data_loader.size // total_train_batch_size
        sgmcmc_step_schedule = cosine_schedule(
            args.sgmcmc_init_step_size,
            total_steps=steps_per_epoch,
        )
    elif args.sgmcmc_step_schedule == "polynomial":
        sgmcmc_step_schedule = polynomial_schedule(
            a=args.sgmcmc_polynomial_schedule_a,
            b=args.sgmcmc_polynomial_schedule_b,
            gamma=args.sgmcmc_polynomial_schedule_gamma,
        )
    else:
        raise ValueError(f"Unknown SGMCMC step schedule {args.sgmcmc_step_schedule}.")

    if args.sgmcmc_preconditioner:
        sgmcmc_preconditioner = rmsprop_preconditioner()
    else:
        sgmcmc_preconditioner = identity_preconditioner()

    posterior_approximations = {
        SWAG_NAME: SWAGPosteriorApproximator(rank=args.swag_rank),
        MAP_NAME: MAPPosteriorApproximator(),
        DEEP_ENSEMBLE_NAME: DeepEnsemblePosteriorApproximator(),
        ADVI_NAME: ADVIPosteriorApproximator(),
        LAPLACE_NAME: LaplacePosteriorApproximator(),
        SNGP_NAME: SNGPPosteriorApproximator(
            output_dim=args.num_labels, mean_field_factor=args.sngp_mean_field_factor
        ),
        SGHMC_NAME: SGHMCPosteriorApproximator(
            n_samples=args.n_posterior_samples,
            n_thinning=args.sgmcmc_n_thinning,
            burnin_length=args.sgmcmc_burnin_length,
            momentum_decay=args.sghmc_momentum_decay,
            step_schedule=sgmcmc_step_schedule,
            preconditioner=sgmcmc_preconditioner,
        ),
    }

    model_editor = None
    if args.enable_probit_model_editor:
        probit_freeze_fun = lambda p, v: True if "classifier" in p else False if args.probit_last_layer_only else None
        model_editor = ProbitModelEditor(
            freeze_fun=probit_freeze_fun,
            init_log_var=args.probit_init_log_var,
            stop_gradient=args.probit_stop_gradient
        )

    ### TRAINING
    prob_model = ProbClassifier(
        model=model,
        posterior_approximator=posterior_approximations[
            args.posterior_approximator_name
        ],
        prior=IsotropicGaussianPrior(log_var=args.prior_log_var),
        output_calibrator=None,
        model_editor=model_editor,
    )

    fit_config = FitConfig(
        hyperparameters=FitHyperparameters(
            max_grad_norm=args.max_grad_norm,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        ),
        monitor=FitMonitor(
            disable_training_metrics_computation=False,
            metrics=(accuracy,),
            # early_stopping_monitor="val_loss",
            # early_stopping_patience=1,
        ),
        processor=FitProcessor(disable_jit=args.disable_jit, devices=args.devices),
        optimizer=FitOptimizer(
            method=optimizer,
            n_epochs=args.num_train_epochs,
        ),
        checkpointer=FitCheckpointer(
            save_checkpoint_dir=args.output_data_dir,
            save_every_n_steps=args.save_every_n_steps,
            keep_top_n_checkpoints=args.keep_top_n_checkpoints,
            restore_checkpoint_path=restore_checkpoint_path,
        ),
        callbacks=[
            ResetCovarianceCallback(
                precision_matrix_key_name="precision_matrix", ridge_penalty=1
            )
        ]
        if args.posterior_approximator_name == SNGPPosteriorApproximator
        else None,
    )

    if args.last_layer_only:
        if args.model_name_or_path == "roberta-base":
            freeze_fun = (
                lambda path, v: "trainable" if "classifier" in path else "frozen"
            )
        elif args.model_name_or_path.startswith("bert-"):
            freeze_fun = (
                lambda path, v: "trainable"
                if ("classifier" in path or "pooler" in path)
                else "frozen"
            )
        else:
            raise ValueError(
                f"Unknown model for last layer training: {args.model_name_or_path}."
            )

        if args.posterior_approximator_name in LAST_LAYER_APPROXIMATORS:
            last_layer_optimizer = FitOptimizer(
                method=optimizer, n_epochs=args.num_train_epochs, freeze_fun=freeze_fun
            )
            if restore_checkpoint_path is not None:
                fit_config.optimizer = last_layer_optimizer
                train_kwargs = {"fit_config": fit_config}
            else:
                map_fit_config = copy.copy(fit_config)
                map_fit_config.checkpointer = FitCheckpointer()
                fit_config.optimizer = last_layer_optimizer
                train_kwargs = {
                    "map_fit_config": map_fit_config,
                    "fit_config": fit_config,
                }
        else:
            raise ValueError(
                f"Last layer approximation is supported only for {LAST_LAYER_APPROXIMATORS}."
            )
    else:
        train_kwargs = {"fit_config": fit_config}

    if args.num_train_epochs > 0:
        status = prob_model.train(
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            calib_data_loader=None,
            **train_kwargs,
        )
    elif restore_checkpoint_path is not None:
        prob_model.load_state(restore_checkpoint_path)
    else:
        raise ValueError(
            "Either restore_checkpoint_path or num_train_epochs > 0 should be specified."
        )

    if args.enable_probit_model_editor:
        logger.info(
            f"Probit log-variance: {prob_model.posterior.state.get().params['model_editor']['params']['log_var']}"
        )

    ### IN-D PERFORMANCE
    test_inputs_loader = test_data_loader.to_inputs_loader()
    test_targets = test_data_loader.to_array_targets()
    test_means = prob_model.predictive.mean(
        inputs_loader=test_inputs_loader, n_posterior_samples=args.n_posterior_samples
    )
    pathlib.Path(args.output_data_dir).mkdir(exist_ok=True, parents=True)
    jnp.savez(
        pathlib.Path(args.output_data_dir) / "test_arrays",
        probs=test_means,
        targets=test_targets,
    )
    test_modes = prob_model.predictive.mode(
        inputs_loader=test_inputs_loader,
        means=test_means,
        n_posterior_samples=args.n_posterior_samples,
    )

    ind_acc = accuracy(preds=test_modes, targets=test_targets)
    ind_ece = expected_calibration_error(
        preds=test_modes,
        probs=test_means,
        targets=test_targets,
    )
    logger.info(f"IND Test accuracy: {ind_acc}")
    logger.info(f"IND ECE: {ind_ece}")

    ### OOD PERFORMANCE
    datasets = DatasetDict(
        {
            "test": load_dataset(
                args.ood_dataset_name,
                name=args.ood_task_name,
                split=args.ood_test_split,
            )
        }
    )
    rng = jax.random.PRNGKey(args.seed)
    datasets = hf_dataset.get_tokenized_datasets(
        datasets,
        text_columns=args.ood_text_columns or args.text_columns,
        target_column=args.ood_target_column or args.target_column,
    )
    test_data_loader = hf_dataset.get_data_loader(
        datasets["test"],
        per_device_batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        drop_last=False,
        rng=rng,
        verbose=True,
    )

    test_inputs_loader = test_data_loader.to_inputs_loader()
    test_means = prob_model.predictive.mean(
        inputs_loader=test_inputs_loader, n_posterior_samples=args.n_posterior_samples
    )
    test_modes = prob_model.predictive.mode(
        inputs_loader=test_inputs_loader,
        means=test_means,
        n_posterior_samples=args.n_posterior_samples,
    )

    test_targets = test_data_loader.to_array_targets()
    ood_acc = accuracy(preds=test_modes, targets=test_targets)
    ood_ece = expected_calibration_error(
        preds=test_modes,
        probs=test_means,
        targets=test_targets,
    )
    logger.info(f"OOD Test accuracy: {ood_acc}")
    logger.info(f"OOD ECE: {ood_ece}")

    results = {
        "ind_dataset": args.dataset_name,
        "ood_dataset": args.ood_dataset_name,
        "ind_acc": float(ind_acc),
        "ind_ece": float(ind_ece),
        "ood_acc": float(ood_acc),
        "ood_ece": float(ood_ece),
    }
    logger.info(results)
    json.dump(results, (pathlib.Path(args.output_data_dir) / "results.json").open("w"))

    logger.info(
        f"Saved artifacts at {args.output_data_dir}:\n {list(pathlib.Path(args.output_data_dir).rglob('*'))}"
    )
