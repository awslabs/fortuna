import os

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
    model_id = "tiiuae/falcon-7b"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", load_in_8bit=True
    )
    model.eval()
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
    calib_answers, calib_questions, calib_targets = [], [], []
    test_answers, test_questions, test_targets = [], [], []
    for i, sample in enumerate(samples):
        if i < calib_size:
            calib_questions.append(sample["question"])
            calib_answers.append(sample["choices"][0])
            calib_targets.append(int(sample["targets"] == 0))
        else:
            test_questions.append(sample["question"])
            test_answers.append(sample["choices"][0])
            test_targets.append(int(sample["targets"] == 0))
    test_targets = np.array(test_targets)

    # calibrate
    calibrator = HallucinationMulticalibrator(
        generative_model=model, tokenizer=tokenizer
    )

    status = calibrator.fit(
        texts=calib_answers,
        contexts=calib_questions,
        targets=calib_targets,
    )

    calibrator.save(f"fitted_calibrator_{model_id.replace('/', '_')}.pth")

    # test
    test_probs = calibrator.predict_proba(
        texts=test_answers, contexts=test_questions, calibrate=False
    )
    test_preds = calibrator.predict(
        texts=test_answers, contexts=test_questions, probs=test_probs
    )

    calib_test_probs = calibrator.predict_proba(
        texts=test_answers, contexts=test_questions
    )
    calib_test_preds = calibrator.predict(
        texts=test_answers, contexts=test_questions, probs=calib_test_probs
    )

    # measure
    mse_before = calibrator.multicalibrator.mean_squared_error(
        probs=test_probs, targets=test_targets
    )
    acc_before = accuracy(test_preds, test_targets)
    mse_after = calibrator.multicalibrator.mean_squared_error(
        probs=calib_test_probs, targets=test_targets
    )
    acc_after = accuracy(calib_test_preds, test_targets)

    print(f"MSE before calibration: {round(float(mse_before), 4)}.")
    print(f"Accuracy before calibration: {round(float(acc_before), 4)}.")
    print(f"MSE after calibration: {round(float(mse_after), 4)}.")
    print(f"Accuracy after calibration: {round(float(acc_after), 4)}.")
