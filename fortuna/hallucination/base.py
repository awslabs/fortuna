import logging
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from sklearn.manifold import locally_linear_embedding
from sklearn.mixture import GaussianMixture
import torch
from torch import nn
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from fortuna.conformal import BinaryClassificationMulticalibrator
from fortuna.hallucination.grouping.clustering.base import GroupingModel
from fortuna.hallucination.scoring.inv_perplexity import inv_perplexity


class HallucinationMulticalibrator:
    def __init__(
        self,
        generative_model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        embedding_reduction_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        clustering_models: Optional[List] = None,
        scoring_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
        ] = None,
    ):
        """
        A hallucination multicalibrator class.
        Given some context (e.g. a question), some text (e.g. one or multiple answers) and some scoring function,
        it multicalibrates the probability that the text is a hallucination or not based on a supervised calibration
        dataset.
        Under the hood, the method constructs model-based low-dimensional embeddings of the initial contexts plus texts,
        it clusters them to provide meaningful (possibly overlapping) groups,
        it computes the scores for each text,
        and it finally calibrates the scores on each subset formed by the intersection of each group and each level set
        of the score function.

        Parameters
        ----------
        generative_model: nn.Module
            A generative model.
        tokenizer: PreTrainedTokenizer
            A tokenizer.
        embedding_reduction_fn: Optional[Callable[[np.ndarray], np.ndarray]]
            A function aimed at reducing the embedding dimensionality.
        clustering_models: Optional[List]
            A list of clustering models.
        scoring_fn: Optional[Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]]
            A scoring function.
        """
        self.generative_model = generative_model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logging.info("`tokenizer.pad_token` is None. Set to `tokenizer.eos_token`.")
        self.embedding_reduction_fn = (
            embedding_reduction_fn or locally_linear_embedding_fn
        )
        self.scoring_fn = scoring_fn or inv_perplexity
        self.clustering_models = clustering_models or [
            GaussianMixture(n_components=i) for i in range(2, 21)
        ]
        self.grouping_model = None
        self.multicalibrator = None
        self._quantiles = None

    def fit(
        self,
        texts: Union[List[str], List[List[str]]],
        contexts: List[str],
        targets: List[str],
        quantile_group_scores_threshold: float = 0.8,
    ) -> Dict:
        """
        Fit the multicalibrator.
        Internally, it first embeds the inputs using the generative model.
        Then it reduces the embeddings dimensionality.
        Then it fits the provided clustering models and finds the best according to the BIC.
        Finally, it computes the scores and applies the multicalibrator.

        Parameters
        ----------
        texts: Union[List[str], List[List[str]]]
            The texts to fit.
            This may either be a list of strings (e.g. a list of single answers),
            or a list of lists of strings (e.g. a list of multi-choice answers).
        contexts: List[str]
            A list of contexts (e.g. a list of questions).
        targets: Union[List[str]]
            A list of target variables to be used for calibration.
            If `texts` is a list of strings, `targets` should be binary variables indicating whether each of the strings
            in the `texts` list should be marked as positive given the corresponding `contexts`.
            If `texts` is a list of lists of strings,
            then `targets` should be a list of integers indicating the position of the strings in the inner lists that
            should be marked as a positive class.
        quantile_group_scores_threshold: float
            A threshold for which to compute the quantiles of the clustering scores.
            This will determine the groups.

        Returns
        -------
        Dict
            The status returned by fitting the multicalibrator.
        """
        (
            scores,
            embeddings,
            which_choices,
        ) = self._compute_scores_embeddings_which_choices(
            texts=texts, contexts=contexts
        )
        if len(which_choices):
            targets = (which_choices == np.array(targets[: len(which_choices)])).astype(
                int
            )
        else:
            targets = np.array(targets)

        embeddings = self.embedding_reduction_fn(embeddings)
        embeddings = np.concatenate((embeddings, scores[:, None]), axis=1)

        self.grouping_model = GroupingModel()
        self.grouping_model.fit(embeddings, clustering_models=self.clustering_models)

        group_scores = self.grouping_model.predict_proba(
            embeddings=embeddings,
        )
        self._quantiles = self.grouping_model.compute_quantile(
            probs=group_scores, threshold=quantile_group_scores_threshold
        )
        groups = self._get_groups(group_scores)

        self.multicalibrator = BinaryClassificationMulticalibrator()
        status = self.multicalibrator.calibrate(
            probs=scores, targets=targets, groups=groups
        )

        return status

    def predict_proba(
        self,
        texts: Union[List[str], List[List[str]]],
        contexts: List[str],
        calibrate: bool = True,
    ) -> np.ndarray:
        """
        Predict probabilities of positive classes for each text given each context.

        Parameters
        ----------
        texts: Union[List[str], List[List[str]]]
            The texts to fit.
            This may either be a list of strings (e.g. a list of single answers),
            or a list of lists of strings (e.g. a list of multi-choice answers).
        contexts: List[str]
            A list of contexts (e.g. a list of questions).
        calibrate: bool
            Whether to calibration the initial probability estimates.

        Returns
        -------
        np.ndarray
            The predicted probabilities.
        """
        if self.multicalibrator is None:
            raise ValueError("`fit` must be called before this method.")

        (
            scores,
            embeddings,
            which_choices,
        ) = self._compute_scores_embeddings_which_choices(
            texts=texts, contexts=contexts
        )
        if not calibrate:
            return scores

        embeddings = self.embedding_reduction_fn(embeddings)
        embeddings = np.concatenate((embeddings, scores[:, None]), axis=1)

        group_scores = self.grouping_model.predict_proba(
            embeddings=embeddings,
        )
        groups = self._get_groups(group_scores)

        return self.multicalibrator.apply_patches(probs=scores, groups=groups)

    def predict(
        self,
        texts: Union[List[str], List[List[str]]],
        contexts: List[str],
        calibrate: bool = True,
        probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        A binary prediction for each text given each context.
        The method returns 1 of a predicted is greater than 0.5, and 0 otherwise.

        Parameters
        ----------
        texts: Union[List[str], List[List[str]]]
            The texts to fit.
            This may either be a list of strings (e.g. a list of single answers),
            or a list of lists of strings (e.g. a list of multi-choice answers).
        contexts: List[str]
            A list of contexts (e.g. a list of questions).
        calibrate: bool
            Whether to calibration the initial probability estimates.
        probs: Optional[np.ndarray]
            Predicted probabilities. If these are available, the method does not compute them under the hood.

        Returns
        -------
        np.ndarray
            The predicted probabilities.
        """
        if probs is None:
            probs = self.predict_proba(
                texts=texts, contexts=contexts, calibrate=calibrate
            )
        return (probs >= 0.5).astype(int)

    def _get_groups(self, group_scores: np.ndarray):
        groups = group_scores >= self._quantiles[None]
        return np.concatenate((groups, np.ones((groups.shape[0], 1))), axis=1).astype(
            bool
        )

    def _compute_scores_embeddings_which_choices(
        self,
        texts: Union[List[str], List[List[str]]],
        contexts: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores = []
        embeddings = []
        which_choices = []

        for text, context in tqdm(zip(texts, contexts)):
            _logits, _scores = self._get_logits_scores(text, context)
            _embeddings = _logits.mean(1)
            if isinstance(text, list):
                which_choice = np.argmax(_scores)
                which_choices.append(which_choice)
                scores.append(_scores[which_choice])
                embeddings.append(_embeddings[which_choice, None])
            elif isinstance(text, str):
                embeddings.append(_embeddings)
                scores.append(_scores)

        return (
            np.array(scores),
            np.concatenate(embeddings, axis=0),
            np.array(which_choices),
        )

    def _get_logits_scores(
        self, text: str, context: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        context_inputs = self.tokenizer(context, return_tensors="pt", padding=True).to(
            self.generative_model.device
        )
        text_inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(
            self.generative_model.device
        )
        inputs = {
            k: torch.cat((context_inputs[k].repeat((v.shape[0], 1)), v), dim=1)
            for k, v in text_inputs.items()
        }

        with torch.no_grad():
            _logits = self.generative_model(**inputs).logits

        _scores = self.scoring_fn(
            logits=_logits,
            labels=inputs["input_ids"],
            init_pos=len(context_inputs),
        )

        return _logits.cpu().numpy(), _scores.cpu().numpy()


def locally_linear_embedding_fn(x: np.ndarray) -> np.ndarray:
    return locally_linear_embedding(
        x, n_neighbors=300, n_components=200, method="modified"
    )[0]
