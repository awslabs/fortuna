"""
Self-contained example that shows how to fine tune a Language Model from Hugging Face on a Text Classification task
using Fortuna.
"""
import argparse
import copy
import json
import logging
import pathlib
import tarfile

import jax.random
import optax
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification

from fortuna.data.dataset.huggingface_datasets import (
    HuggingFaceSequenceClassificationDataset,
)
from fortuna.metric.classification import accuracy, expected_calibration_error
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
    SNGPPosteriorApproximator,
    SWAGPosteriorApproximator,
)
from fortuna.prob_model.fit_config.hyperparametrs import FitHyperparametrs
from fortuna.prob_model.posterior.posterior_approximations import (
    ADVI_NAME,
    DEEP_ENSEMBLE_NAME,
    LAPLACE_NAME,
    MAP_NAME,
    SNGP_NAME,
    SWAG_NAME,
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

logger = logging.getLogger(__name__)


# just a bunch of utilities
def setup_logging() -> None:
    logging.basicConfig(format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def unpack_model_tar(model_ckpt_path: pathlib.Path) -> pathlib.Path:
    """
    Untar a tar.gz object

    Parameters
    ----------
    model_ckpt_path: pathlib.Path
        A local file system path that points to the `tar.gz` archive containing the model ckpts.
    Returns
    -------
    pathlib.Path
        The directory that contains the uncompress model checkpoint files
    """
    model_dir = model_ckpt_path.parent
    # untar the model
    tar = tarfile.open(str(model_ckpt_path))
    tar.extractall(model_dir)
    tar.close()
    return model_dir


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description="Transformers finetuning")
    # input folders
    parser.add_argument(
        "--save_checkpoint_dir",
        type=str,
        default=None,
        help="The folder containing the checkpoints that should be restored. "
        "Checkpoints should be in compressed in tar.gz",
    )
    parser.add_argument(
        "--restore_checkpoint_dir",
        type=str,
        default=None,
        help="The folder containing the checkpoints that should be restored. "
        "Checkpoints should be in compressed in tar.gz",
    )
    # general
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_jit", action="store_true")
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--no_labels_check", action="store_true")
    # data
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="imdb",
        help="The name of the dataset on huggingface (e.g glue)",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the task on huggingface (e.g, mnli)",
    )
    parser.add_argument(
        "--text_columns",
        nargs="+",
        default=("text",),
        help="A list of columns name containing the input text in the dataset",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        nargs="+",
        default="label",
        help="Column name of the target in the dataset",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help="Number of unique labels in the sequence classification task",
    )
    parser.add_argument("--train_split", type=str, default="train[:2%]+train[-2%:]")
    parser.add_argument("--validation_split", type=str, default="test[:1%]+test[-1%:]")
    parser.add_argument("--test_split", type=str, default="test[49%:51%]")
    # trainer/model
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-cased")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument(
        "--disable_gradient_checkpointing",
        action="store_true",
        help="Gradient chheckpointing is used by default. Use --disable_gradient_checkpointing if you do not want to use it.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    # posterior approximation
    parser.add_argument(
        "--posterior_approximator_name",
        type=str,
        default="map",
        choices=["map", "swag", "advi", "laplace", "deep_ensamble", "sngp"],
    )
    parser.add_argument(
        "--last_layer_only",
        action="store_true",
        help="Whether you want to run one between ADVI, SWAG or Laplace only on the last layer of the model",
    )
    parser.add_argument("--prior_log_var", type=float, default=1.0)
    parser.add_argument(
        "--n_posterior_samples",
        type=int,
        default=30,
        help="Number of sample to draw from the posterior when computing predictive stats",
    )
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
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Early stopping is not used when early_stopping_patience is set to a value <= 0 or None.",
    )
    parser.add_argument("--early_stopping_mode", type=str, default="min")
    parser.add_argument("--early_stopping_monitor", type=str, default="val_loss")

    args = parser.parse_args()
    rng = jax.random.PRNGKey(args.seed)

    try:
        logger.info(list(pathlib.Path(args.restore_checkpoint_dir).rglob("*")))
        restore_checkpoint_path = unpack_model_tar(
            list(pathlib.Path(args.restore_checkpoint_dir).rglob("*"))[0]
        )
        logger.info(list(pathlib.Path(restore_checkpoint_path).rglob("*")))
    except:
        logger.info("No checkpoint to restore")
        restore_checkpoint_path = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    #####################################
    ####       DATASET SETUP         ####
    #####################################
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
    if not args.no_labels_check:
        for k, v in datasets.items():
            unique_labels_set = len(v.unique("label"))
            if unique_labels_set != args.num_labels:
                raise ValueError(
                    f"The {k} set has {unique_labels_set} unique label but was expecting {args.num_labels}."
                    f"If this is not suspicious re-run the scripts with --no_labels_check"
                )

    hf_dataset = HuggingFaceSequenceClassificationDataset(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=args.max_sequence_length,
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

    #####################################
    ####         MODEL SETUP         ####
    #####################################
    model_kwargs = {"num_labels": args.num_labels}
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
    if not args.disable_gradient_checkpointing:
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

    posterior_approximations = {
        SWAG_NAME: SWAGPosteriorApproximator(rank=args.swag_rank),
        MAP_NAME: MAPPosteriorApproximator(),
        DEEP_ENSEMBLE_NAME: DeepEnsemblePosteriorApproximator(),
        ADVI_NAME: ADVIPosteriorApproximator(),
        LAPLACE_NAME: LaplacePosteriorApproximator(),
        SNGP_NAME: SNGPPosteriorApproximator(
            output_dim=args.num_labels, mean_field_factor=args.sngp_mean_field_factor
        ),
    }

    #####################################
    ####           TRAIN!            ####
    #####################################
    prob_model = ProbClassifier(
        model=model,
        posterior_approximator=posterior_approximations[
            args.posterior_approximator_name
        ],
        prior=IsotropicGaussianPrior(log_var=args.prior_log_var),
    )
    fit_config = FitConfig(
        hyperparameters=FitHyperparametrs(
            max_grad_norm=args.max_grad_norm,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        ),
        monitor=FitMonitor(
            disable_training_metrics_computation=False,
            metrics=(accuracy,),
            early_stopping_monitor=args.early_stopping_monitor,
            early_stopping_patience=1,
        ),
        processor=FitProcessor(disable_jit=args.disable_jit, devices=args.devices),
        optimizer=FitOptimizer(
            method=optimizer,
            n_epochs=args.num_train_epochs,
        ),
        checkpointer=FitCheckpointer(
            save_checkpoint_dir=args.save_checkpoint_dir,
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
    if args.last_layer_only and (
        args.posterior_approximator_name in [LAPLACE_NAME, ADVI_NAME, SWAG_NAME]
    ):
        last_layer_optimizer = FitOptimizer(
            method=optimizer,
            n_epochs=args.num_train_epochs,
            freeze_fun=(
                lambda path, v: "trainable" if "classifier" in path else "frozen"
            )
            if (args.posterior_approximator_name in [ADVI_NAME, LAPLACE_NAME])
            and args.last_layer_only
            else None,
        )
        if restore_checkpoint_path is not None:
            fit_config.optimizer = last_layer_optimizer
            train_kwargs = {"fit_config": fit_config}
        else:
            map_fit_config = copy.copy(fit_config)
            map_fit_config.checkpointer = FitCheckpointer()
            fit_config.optimizer = last_layer_optimizer
            train_kwargs = {"map_fit_config": map_fit_config, "fit_config": fit_config}
    else:
        train_kwargs = {"fit_config": fit_config}

    status = prob_model.train(
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        calib_data_loader=None,
        **train_kwargs,
    )
    #####################################
    ####          EVALUATE           ####
    #####################################
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
    ind_acc = accuracy(preds=test_modes, targets=test_targets)
    ind_ece = expected_calibration_error(
        preds=test_modes,
        probs=test_means,
        targets=test_targets,
    )
    logger.info(f"Test accuracy: {ind_acc}")
    logger.info(f"ECE: {ind_ece}")

    results = {
        "dataset": args.dataset_name,
        "accuracy": float(ind_acc),
        "ece": float(ind_ece),
    }
    logger.info(results)
    if args.save_checkpoint_dir is not None:
        save_checkpoint_dir = pathlib.Path(args.save_checkpoint_dir)
        save_checkpoint_dir.mkdir(exist_ok=True, parents=True)
        json.dump(results, (save_checkpoint_dir / "results.json").open("w"))
        logger.info(
            f"Saved artifacts at {save_checkpoint_dir}:\n {list(save_checkpoint_dir.rglob('*'))}"
        )
