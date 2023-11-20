import os
import pickle

from datasets import (
    get_dataset_config_names,
    load_dataset,
)
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from fortuna.hallucination import HallucinationMulticalibrator
from fortuna.hallucination.utils import string_cleaner
from fortuna.metric.classification import accuracy

SEED = 0
CALIB_FRAC = 0.8

if __name__ == "__main__":
    device = "cuda"
    model_id = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # download and prepare data
    task_list = get_dataset_config_names("lukaemon/mmlu")
    dataset_list = [
        (
            load_dataset(
                "lukaemon/mmlu",
                task,
                cache_dir=".cache/huggingface/datasets/"
                if os.path.isdir(".cache/huggingface/datasets/")
                else None,
            ),
            task,
        )
        for task in task_list
    ]

    answer_map = dict(zip(["A", "B", "C", "D"], [0, 1, 2, 3]))
    samples = []
    for datasets, task in dataset_list:
        for dataset_key, dataset in datasets.items():
            for sample in dataset:
                samples.append(
                    dict(
                        question=string_cleaner(sample["input"]),
                        choices=[sample[letter] for letter in answer_map.keys()],
                        targets=answer_map[sample["target"]],
                    )
                )

    # shuffle and split
    rng = np.random.default_rng(seed=SEED)
    tot_size = len(samples)
    perm = rng.choice(tot_size, tot_size, replace=False)
    samples = [samples[i] for i in perm]

    calib_size = int(np.ceil(CALIB_FRAC * tot_size))
    calib_choices, calib_questions, calib_targets = [], [], []
    test_choices, test_questions, test_targets = [], [], []
    for i, sample in enumerate(samples):
        if i < calib_size:
            calib_questions.append(sample["question"])
            calib_choices.append(sample["choices"])
            calib_targets.append(sample["targets"])
        else:
            test_questions.append(sample["question"])
            test_choices.append(sample["choices"])
            test_targets.append(sample["targets"])

    # calibrate
    calibrator = HallucinationMulticalibrator(
        generative_model=model, tokenizer=tokenizer
    )

    status = calibrator.fit(
        texts=calib_choices,
        contexts=calib_questions,
        targets=calib_targets,
    )

    with open("fitted_calibrator.pth", "wb") as filehandler:
        pickle.dump(calibrator, filehandler, -1)

    # test
    test_probs = calibrator.predict_proba(
        texts=test_choices, contexts=test_questions, calibrate=False
    )
    test_preds = calibrator.predict(
        texts=test_choices, contexts=test_questions, probs=test_probs
    )

    calib_test_probs = calibrator.predict_proba(
        texts=test_choices, contexts=test_questions
    )
    calib_test_preds = calibrator.predict(
        texts=test_choices, contexts=test_questions, probs=calib_test_probs
    )

    # measure
    mse_before = calibrator.multicalibrator.mean_squared_error(
        probs=test_probs, targets=np.array(test_targets)
    )
    acc_before = accuracy(test_preds, np.array(test_targets))
    mse_after = calibrator.multicalibrator.mean_squared_error(
        probs=calib_test_probs, targets=np.array(test_targets)
    )
    acc_after = accuracy(calib_test_preds, np.array(test_targets))

    print(f"MSE before calibration: {round(float(mse_before), 4)}.")
    print(f"Accuracy before calibration: {round(float(acc_before), 4)}.")
    print(f"MSE after calibration: {round(float(mse_after), 4)}.")
    print(f"Accuracy after calibration: {round(float(acc_before), 4)}.")
