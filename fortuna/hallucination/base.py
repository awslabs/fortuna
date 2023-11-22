import logging
import pickle
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from sklearn.mixture import GaussianMixture
import torch
from torch import nn
from tqdm import tqdm
from transformers import PreTrainedTokenizer
import umap.umap_ as umap

from fortuna.conformal import BinaryClassificationMulticalibrator
from fortuna.hallucination.grouping.clustering.base import GroupingModel
from fortuna.hallucination.scoring.inv_perplexity import inv_perplexity


class HallucinationMulticalibrator:
    def __init__(
        self,
        generative_model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        embedding_reduction_model: Optional = None,
        clustering_models: Optional[List] = None,
        scoring_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
        ] = None,
        seed: int = 0,
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
        embedding_reduction_model: Optional
            An embedding reduction model.
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
        self.embedding_reduction_model = embedding_reduction_model or umap.UMAP(
            n_neighbors=20
        )
        self.scoring_fn = scoring_fn or inv_perplexity
        self.clustering_models = clustering_models or [
            GaussianMixture(n_components=i) for i in range(2, 21)
        ]
        self.grouping_model = None
        self.multicalibrator = None
        self._quantiles = None
        self.rng = np.random.default_rng(seed)

    def fit(
        self,
        texts: List[str],
        contexts: List[str],
        targets: List[int],
        batch_size: int = 16,
        quantile_group_scores_threshold: float = 0.8,
        balance: bool = True,
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
        batch_size: int
            The batch size.
        quantile_group_scores_threshold: float
            A threshold for which to compute the quantiles of the clustering scores.
            This will determine the groups.
        balance: bool
            Whether to balance the calibration data.

        Returns
        -------
        Dict
            The status returned by fitting the multicalibrator.
        """
        if balance:
            texts, contexts, targets = self._balance_data(texts, contexts, targets)

        (
            scores,
            embeddings,
        ) = self._compute_scores_embeddings(
            texts=texts, contexts=contexts, batch_size=batch_size
        )
        targets = np.array(targets, dtype="int32")

        embeddings = self.embedding_reduction_model.fit_transform(embeddings)
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
        texts: List[str],
        contexts: List[str],
        batch_size: int = 16,
        calibrate: bool = True,
    ) -> np.ndarray:
        """
        Predict probabilities of positive classes for each text given each context.

        Parameters
        ----------
        texts: List[str]
            The texts to fit.
            This may either be a list of strings (e.g. a list of single answers),
            or a list of lists of strings (e.g. a list of multi-choice answers).
        contexts: List[str]
            A list of contexts (e.g. a list of questions).
        batch_size: int
            The batch size.
        calibrate: bool
            Whether to calibration the initial probability estimates.

        Returns
        -------
        np.ndarray
            The predicted probabilities.
        """
        if self.multicalibrator is None:
            raise ValueError("`fit` must be called before this method.")

        (scores, embeddings) = self._compute_scores_embeddings(
            texts=texts, contexts=contexts, batch_size=batch_size
        )
        if not calibrate:
            return scores

        embeddings = self.embedding_reduction_model.transform(embeddings)
        embeddings = np.concatenate((embeddings, scores[:, None]), axis=1)

        group_scores = self.grouping_model.predict_proba(
            embeddings=embeddings,
        )
        groups = self._get_groups(group_scores)

        return self.multicalibrator.apply_patches(probs=scores, groups=groups)

    def predict(
        self,
        texts: List[str],
        contexts: List[str],
        batch_size: int = 16,
        calibrate: bool = True,
        probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        A binary prediction for each text given each context.
        The method returns 1 of a predicted is greater than 0.5, and 0 otherwise.

        Parameters
        ----------
        texts: List[str],
            The texts to fit.
            This may either be a list of strings (e.g. a list of single answers),
            or a list of lists of strings (e.g. a list of multi-choice answers).
        contexts: List[str]
            A list of contexts (e.g. a list of questions).
        batch_size: int
            The batch size.
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
                texts=texts,
                contexts=contexts,
                batch_size=batch_size,
                calibrate=calibrate,
            )
        return (probs >= 0.5).astype(int)

    def _get_groups(self, group_scores: np.ndarray):
        groups = group_scores >= self._quantiles[None]
        return np.concatenate((groups, np.ones((groups.shape[0], 1))), axis=1).astype(
            bool
        )

    def _compute_scores_embeddings(
        self, texts: List[str], contexts: List[str], batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores = []
        embeddings = []

        gen = self._batch(texts, contexts, batch_size)

        for batch_texts, batch_contexts in tqdm(
            gen, total=int(np.ceil(len(texts) / batch_size))
        ):
            logits, _scores = self._get_logits_scores(batch_texts, batch_contexts)
            embeddings.append(logits.mean(1))
            scores.append(_scores)

        return (
            np.concatenate(scores, axis=0).astype("float32"),
            np.concatenate(embeddings, axis=0),
        )

    @staticmethod
    def _batch(texts: List[str], contexts: List[str], batch_size: int):
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size], contexts[i : i + batch_size]

    def _get_logits_scores(
        self, texts: str, contexts: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        context_inputs = self.tokenizer(contexts, return_tensors="pt", padding=True).to(
            self.generative_model.device
        )
        text_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(
            self.generative_model.device
        )
        inputs = {
            k: torch.cat((context_inputs[k], v), dim=1) for k, v in text_inputs.items()
        }

        with torch.no_grad():
            _logits = self.generative_model(**inputs).logits

            _scores = self.scoring_fn(
                logits=_logits,
                labels=inputs["input_ids"],
                init_pos=len(context_inputs),
            )

        return _logits.cpu().numpy(), _scores.cpu().numpy()

    def _balance_data(
        self, texts: List[str], contexts: List[str], targets: List[int]
    ) -> Tuple[List[str], List[str], List[int]]:
        idx0 = [i for i, y in enumerate(targets) if y == 0]
        idx1 = [i for i, y in enumerate(targets) if y == 1]
        len_diff = len(idx1) - len(idx0)
        idx = self.rng.choice(
            idx0 if len_diff > 0 else idx1, np.abs(len_diff), replace=True
        )
        for i in idx:
            texts.append(texts[i])
            contexts.append(contexts[i])
            targets.append(targets[i])
        return texts, contexts, targets

    def save(self, path):
        state = dict(
            embedding_reduction_model=self.embedding_reduction_model,
            grouping_model=self.grouping_model,
            multicalibrator=self.multicalibrator,
            _quantiles=self._quantiles,
        )

        with open(path, "wb") as filehandler:
            pickle.dump(state, filehandler, -1)

    def load(self, path):
        state = pickle.load(open(path, "rb"))
        for k, v in state.items():
            setattr(self, k, v)
