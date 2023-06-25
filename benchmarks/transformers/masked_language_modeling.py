"""
Self-contained example that shows how to fine tune a Language Model from Hugging Face on a Masked Language Modeling task
using Fortuna.
"""
import argparse
import copy
import json
import logging
import pathlib
import tarfile

from datasets import (
    DatasetDict,
    load_dataset,
)
import jax.numpy as jnp
import jax.random
import optax
from transformers import (
    AutoTokenizer,
    FlaxAutoModelForMaskedLM,
)

from fortuna.data.dataset.huggingface_datasets import HuggingFaceMaskedLMDataset
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
    SWAGPosteriorApproximator,
)
from fortuna.prob_model.fit_config.hyperparameters import FitHyperparameters
from fortuna.prob_model.posterior.posterior_approximations import (
    ADVI_NAME,
    DEEP_ENSEMBLE_NAME,
    LAPLACE_NAME,
    MAP_NAME,
    SWAG_NAME,
)
from fortuna.prob_model.prior import IsotropicGaussianPrior
from fortuna.typing import Array
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
    parser.add_argument("--train_split", type=str, default="train[:50%]")
    parser.add_argument("--validation_split", type=str, default="test[:5%]")
    parser.add_argument("--test_split", type=str, default="test[:30%]")
    parser.add_argument("--mlm_probability", type=float, default=0.15)
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
        help="Gradient checkpointing is used by default. Use --disable_gradient_checkpointing if you do not want to use it.",
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
        restore_checkpoint_dir = unpack_model_tar(
            list(pathlib.Path(args.restore_checkpoint_dir).rglob("*"))[0]
        )
        logger.info(list(pathlib.Path(restore_checkpoint_dir).rglob("*")))
    except:
        logger.info("No checkpoint to restore")
        restore_checkpoint_dir = None

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

    hf_dataset = HuggingFaceMaskedLMDataset(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=args.max_sequence_length,
        mlm_probability=args.mlm_probability,
    )
    datasets = hf_dataset.get_tokenized_datasets(
        datasets,
        text_columns=args.text_columns
        if not isinstance(args.text_columns, str)
        else (args.text_columns,),
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
    try:
        model = FlaxAutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    except OSError:
        model = FlaxAutoModelForMaskedLM.from_pretrained()
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
    }

    #####################################
    ####           TRAIN!            ####
    #####################################
    def accuracy_mlm(preds: Array, targets: Array) -> jnp.ndarray:
        if preds.ndim > 1:
            raise ValueError(
                """`preds` must be a one-dimensional array of predicted classes."""
            )
        if targets.ndim > 1:
            raise ValueError(
                """`targets` must be a one-dimensional array of target classes."""
            )
        targets_mask = jnp.where(targets > 0, 1.0, 0.0)
        return jnp.sum(jnp.equal(preds, targets) * targets_mask) / targets_mask.sum()

    prob_model = ProbClassifier(
        model=model,
        posterior_approximator=posterior_approximations[
            args.posterior_approximator_name
        ],
        prior=IsotropicGaussianPrior(log_var=args.prior_log_var),
    )
    fit_config = FitConfig(
        hyperparameters=FitHyperparameters(
            max_grad_norm=args.max_grad_norm,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        ),
        monitor=FitMonitor(
            disable_training_metrics_computation=False,
            metrics=(accuracy_mlm,),
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
            restore_checkpoint_dir=restore_checkpoint_dir,
        ),
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
        if restore_checkpoint_dir is not None:
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

    if args.save_checkpoint_dir is not None:
        save_checkpoint_dir = pathlib.Path(args.save_checkpoint_dir)
        save_checkpoint_dir.mkdir(exist_ok=True, parents=True)
        logger.info(
            f"Saved artifacts at {save_checkpoint_dir}:\n {list(save_checkpoint_dir.rglob('*'))}"
        )
