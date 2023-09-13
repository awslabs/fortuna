import json

import matplotlib.pyplot as plt
import numpy as np

with open("tabular_results.json", "r") as j:
    metrics = json.loads(j.read())

TOL =  1e-4

# ~~~REGRESSION~~~

# MAP
map_nlls = [
    metrics["regression"][k]["map"]["nll"] for k in metrics["regression"].keys()
]
map_picp_errors = [
    np.abs(metrics["regression"][k]["map"]["picp"] - 0.95) for k in metrics["regression"].keys()
]

map_times = [
    metrics["regression"][k]["map"]["time"] for k in metrics["regression"].keys()
]

# TEMPERATURE SCALING
temp_scaling_nlls = [
    metrics["regression"][k]["temp_scaling"]["nll"]
    for k in metrics["regression"].keys()
]
win_temp_scaling_nlls = np.array(temp_scaling_nlls) < np.array(map_nlls) - TOL
lose_temp_scaling_nlls = np.array(temp_scaling_nlls) > np.array(map_nlls) + TOL

temp_scaling_picp_errors = [
    np.abs(metrics["regression"][k]["temp_scaling"]["picp"] - 0.95)
    for k in metrics["regression"].keys()
]
win_temp_scaling_picp_errors = np.array(temp_scaling_picp_errors) < np.array(map_picp_errors) - TOL
lose_temp_scaling_picp_errors = np.array(temp_scaling_picp_errors) > np.array(map_picp_errors) + TOL

temp_scaling_times = [
    metrics["regression"][k]["temp_scaling"]["time"]
    for k in metrics["regression"].keys()
]

temp_scaling_best_win = np.max(np.array(map_picp_errors) - np.array(temp_scaling_picp_errors))
temp_scaling_worst_loss = np.min(np.array(map_picp_errors) - np.array(temp_scaling_picp_errors))

# CQR
cqr_picp_errors = [
    np.abs(metrics["regression"][k]["cqr"]["picp"] - 0.95)
    for k in metrics["regression"].keys()
]
win_cqr_picp_errors = np.array(cqr_picp_errors) < np.array(map_picp_errors) - TOL
lose_cqr_picp_errors = np.array(cqr_picp_errors) > np.array(map_picp_errors) + TOL

cqr_times = [
    metrics["regression"][k]["cqr"]["time"]
    for k in metrics["regression"].keys()
]

cqr_best_win = np.max(np.array(map_picp_errors) - np.array(cqr_picp_errors))
cqr_worst_loss = np.min(np.array(map_picp_errors) - np.array(cqr_picp_errors))

### Regression plots

plt.figure(figsize=(10, 3))
plt.suptitle("Scatter plots for regression datasets")
plt.subplot(1, 2, 1)
plt.title("PICP errors")
plt.grid()
plt.xlabel("MAP")
plt.ylabel("temp scaling")
_min, _max = min(np.array(map_picp_errors).min(), np.array(temp_scaling_picp_errors).min()), max(np.array(map_picp_errors).max(), np.array(temp_scaling_picp_errors).max())
plt.plot([_min, _max], [_min, _max], color="gray", linestyle="--", alpha=0.2)
plt.scatter(map_picp_errors, temp_scaling_picp_errors, s=3, color=["C2" if w else "C3" if l else "grey" for w, l in zip(win_temp_scaling_picp_errors, lose_temp_scaling_picp_errors)])
plt.xscale("log")
plt.yscale("log")

plt.subplot(1, 2, 2)
plt.title("PICP errors")
plt.grid()
plt.xlabel("MAP")
plt.ylabel("CQR")
_min, _max = min(np.array(map_picp_errors).min(), np.array(cqr_picp_errors).min()), max(np.array(map_picp_errors).max(), np.array(cqr_picp_errors).max())
plt.plot([_min, _max], [_min, _max], color="gray", linestyle="--", alpha=0.2)
plt.scatter(map_picp_errors, cqr_picp_errors, s=3, color=["C2" if w else "C3" if l else "grey" for w, l in zip(win_cqr_picp_errors, lose_cqr_picp_errors)])
plt.xscale("log")
plt.yscale("log")

plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 3))
plt.suptitle("Scatter plots for regression datasets on other metrics")
plt.title("NLL")
plt.grid()
plt.xlabel("MAP")
plt.ylabel("temp scaling")
_min, _max = min(np.array(map_nlls).min(), np.array(temp_scaling_nlls).min()), max(np.array(map_nlls).max(), np.array(temp_scaling_nlls).max())
plt.plot([_min, _max], [_min, _max], color="gray", linestyle="--", alpha=0.2)
plt.scatter(map_nlls, temp_scaling_nlls, s=3, color=["C2" if w else "C3" if l else "grey" for w, l in zip(win_temp_scaling_nlls, lose_temp_scaling_nlls)])
plt.xscale("log")
plt.yscale("log")

plt.tight_layout()
plt.show()


# ~~~CLASSIFICATION~~~

# MAP
map_nlls = [
    metrics["classification"][k]["map"]["nll"] for k in metrics["classification"].keys()
]

map_mse = [
    metrics["classification"][k]["map"]["mse"] for k in metrics["classification"].keys()
]
map_ece = [
    metrics["classification"][k]["map"]["ece"] for k in metrics["classification"].keys()
]
map_rocauc = [
    metrics["classification"][k]["map"]["rocauc"]
    for k in metrics["classification"].keys()
    if "rocauc" in metrics["classification"][k]["map"]
]
map_prauc = [
    metrics["classification"][k]["map"]["prauc"]
    for k in metrics["classification"].keys()
    if "prauc" in metrics["classification"][k]["map"]
]
map_acc = [
    metrics["classification"][k]["map"]["accuracy"]
    for k in metrics["classification"].keys()
]

map_times = [
    metrics["regression"][k]["map"]["time"] for k in metrics["regression"].keys()
]

# TEMPERATURE SCALING
temp_scaling_nlls = [
    metrics["classification"][k]["temp_scaling"]["nll"]
    for k in metrics["classification"].keys()
]
win_temp_scaling_nlls = np.array(temp_scaling_nlls) < np.array(map_nlls) - TOL
lose_temp_scaling_nlls = np.array(temp_scaling_nlls) > np.array(map_nlls) + TOL

temp_scaling_mse = [
    metrics["classification"][k]["temp_scaling"]["mse"]
    for k in metrics["classification"].keys()
]
win_temp_scaling_mse = np.array(temp_scaling_mse) < np.array(map_mse) - TOL
lose_temp_scaling_mse = np.array(temp_scaling_mse) > np.array(map_mse) + TOL

temp_scaling_rocauc = [
    metrics["classification"][k]["temp_scaling"]["rocauc"]
    for k in metrics["classification"].keys()
    if "rocauc" in metrics["classification"][k]["temp_scaling"]
]
win_temp_scaling_rocauc = np.array(temp_scaling_rocauc) < np.array(map_rocauc) - TOL
lose_temp_scaling_rocauc = np.array(temp_scaling_rocauc) > np.array(map_rocauc) + TOL

temp_scaling_prauc = [
    metrics["classification"][k]["temp_scaling"]["prauc"]
    for k in metrics["classification"].keys()
    if "prauc" in metrics["classification"][k]["temp_scaling"]
]
win_temp_scaling_prauc = np.array(temp_scaling_prauc) < np.array(map_prauc) - TOL
lose_temp_scaling_prauc = np.array(temp_scaling_prauc) > np.array(map_prauc) + TOL

temp_scaling_acc = [
    metrics["classification"][k]["temp_scaling"]["accuracy"]
    for k in metrics["classification"].keys()
]

temp_scaling_times = [
    metrics["classification"][k]["temp_scaling"]["time"]
    for k in metrics["classification"].keys()
]

temp_scaling_best_win = np.max(np.array(map_mse) - np.array(temp_scaling_mse))
temp_scaling_worst_loss = np.min(np.array(map_mse) - np.array(temp_scaling_mse))

# MULTICALIBRATE CONF
mc_conf_nlls = [
    metrics["classification"][k]["mc_conf"]["nll"]
    for k in metrics["classification"].keys()
]
win_mc_conf_nlls = np.array(mc_conf_nlls) < np.array(map_nlls) - TOL
lose_mc_conf_nlls = np.array(mc_conf_nlls) > np.array(map_nlls) + TOL

mc_conf_mse = [
    metrics["classification"][k]["mc_conf"]["mse"]
    for k in metrics["classification"].keys()
]
win_mc_conf_mse = np.array(mc_conf_mse) < np.array(map_mse) - TOL
lose_mc_conf_mse = np.array(mc_conf_mse) > np.array(map_mse) + TOL

mc_conf_rocauc = [
    metrics["classification"][k]["mc_conf"]["rocauc"]
    for k in metrics["classification"].keys()
    if "rocauc" in metrics["classification"][k]["mc_conf"]
]
win_mc_conf_rocauc = np.array(mc_conf_rocauc) < np.array(map_rocauc) - TOL
lose_mc_conf_rocauc = np.array(mc_conf_rocauc) > np.array(map_rocauc) + TOL

mc_conf_prauc = [
    metrics["classification"][k]["mc_conf"]["prauc"]
    for k in metrics["classification"].keys()
    if "prauc" in metrics["classification"][k]["mc_conf"]
]
win_mc_conf_prauc = np.array(mc_conf_prauc) < np.array(map_prauc) - TOL
lose_mc_conf_prauc = np.array(mc_conf_prauc) > np.array(map_prauc) + TOL

mc_conf_acc = [
    metrics["classification"][k]["mc_conf"]["accuracy"]
    for k in metrics["classification"].keys()
]

mc_conf_times = [
    metrics["classification"][k]["mc_conf"]["time"]
    for k in metrics["classification"].keys()
]

mc_conf_best_win = np.max(np.array(map_mse) - np.array(mc_conf_mse))
mc_conf_worst_loss = np.min(np.array(map_mse) - np.array(mc_conf_mse))

### Classification plots

plt.figure(figsize=(10, 3))
plt.suptitle("Scatter plots for classification datasets")
plt.subplot(1, 2, 1)
plt.title("MSE")
plt.grid()
plt.xlabel("MAP")
plt.ylabel("temp scaling")
_min, _max = min(np.array(map_mse).min(), np.array(temp_scaling_mse).min()), max(np.array(map_mse).max(), np.array(temp_scaling_mse).max())
plt.plot([_min, _max], [_min, _max], color="gray", linestyle="--", alpha=0.2)
plt.scatter(map_mse, temp_scaling_mse, s=3, color=["C2" if w else "C3" if l else "grey" for w, l in zip(win_temp_scaling_mse, lose_temp_scaling_mse)])

plt.subplot(1, 2, 2)
plt.title("MSE")
plt.grid()
plt.xlabel("MAP")
plt.ylabel("TLMC")
_min, _max = min(np.array(map_mse).min(), np.array(mc_conf_mse).min()), max(np.array(map_mse).max(), np.array(mc_conf_mse).max())
plt.plot([_min, _max], [_min, _max], color="gray", linestyle="--", alpha=0.2)
plt.scatter(map_mse, mc_conf_mse, s=3, color=["C2" if w else "C3" if l else "grey" for w, l in zip(win_mc_conf_mse, lose_mc_conf_mse)])

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.suptitle("Scatter plots for classification datasets on other metrics")
plt.subplot(3, 2, 1)
plt.title("NLL")
plt.grid()
plt.xlabel("MAP")
plt.ylabel("temp scaling")
_min, _max = min(np.array(map_nlls).min(), np.array(temp_scaling_nlls).min()), max(np.array(map_nlls).max(), np.array(temp_scaling_nlls).max())
plt.plot([_min, _max], [_min, _max], color="gray", linestyle="--", alpha=0.2)
plt.scatter(map_nlls, temp_scaling_nlls, s=3, color=["C2" if w else "C3" if l else "grey" for w, l in zip(win_temp_scaling_nlls, lose_temp_scaling_nlls)])
plt.xscale("log")
plt.yscale("log")

plt.subplot(3, 2, 2)
plt.title("NLL")
plt.grid()
plt.xlabel("MAP")
plt.ylabel("TLMC")
_min, _max = min(np.array(map_nlls).min(), np.array(mc_conf_nlls).min()), max(np.array(map_nlls).max(), np.array(mc_conf_nlls).max())
plt.plot([_min, _max], [_min, _max], color="gray", linestyle="--", alpha=0.2)
plt.scatter(map_nlls, mc_conf_nlls, s=3, color=["C2" if w else "C3" if l else "grey" for w, l in zip(win_mc_conf_nlls, lose_mc_conf_nlls)])
plt.xscale("log")
plt.yscale("log")

plt.subplot(3, 2, 3)
plt.title("ROCAUC")
plt.grid()
plt.xlabel("MAP")
plt.ylabel("temp scaling")
_min, _max = min(np.array(map_rocauc).min(), np.array(temp_scaling_rocauc).min()), max(np.array(map_rocauc).max(), np.array(temp_scaling_rocauc).max())
plt.plot([_min, _max], [_min, _max], color="gray", linestyle="--", alpha=0.2)
plt.scatter(map_rocauc, temp_scaling_rocauc, s=3, color=["C2" if w else "C3" if l else "grey" for w, l in zip(win_temp_scaling_rocauc, lose_temp_scaling_rocauc)])

plt.subplot(3, 2, 4)
plt.title("ROCAUC")
plt.grid()
plt.xlabel("MAP")
plt.ylabel("TLMC")
_min, _max = min(np.array(map_rocauc).min(), np.array(mc_conf_rocauc).min()), max(np.array(map_rocauc).max(), np.array(mc_conf_rocauc).max())
plt.plot([_min, _max], [_min, _max], color="gray", linestyle="--", alpha=0.2)
plt.scatter(map_rocauc, mc_conf_rocauc, s=3, color=["C2" if w else "C3" if l else "grey" for w, l in zip(win_mc_conf_rocauc, lose_mc_conf_rocauc)])

plt.subplot(3, 2, 5)
plt.title("PRAUC")
plt.grid()
plt.xlabel("MAP")
plt.ylabel("temp scaling")
_min, _max = min(np.array(map_prauc).min(), np.array(temp_scaling_prauc).min()), max(np.array(map_prauc).max(), np.array(temp_scaling_prauc).max())
plt.plot([_min, _max], [_min, _max], color="gray", linestyle="--", alpha=0.2)
plt.scatter(map_prauc, temp_scaling_prauc, s=3, color=["C2" if w else "C3" if l else "grey" for w, l in zip(win_temp_scaling_prauc, lose_temp_scaling_prauc)])

plt.subplot(3, 2, 6)
plt.title("PRAUC")
plt.grid()
plt.xlabel("MAP")
plt.ylabel("TLMC")
_min, _max = min(np.array(map_prauc).min(), np.array(mc_conf_prauc).min()), max(np.array(map_prauc).max(), np.array(mc_conf_prauc).max())
plt.plot([_min, _max], [_min, _max], color="gray", linestyle="--", alpha=0.2)
plt.scatter(map_prauc, mc_conf_prauc, s=3, color=["C2" if w else "C3" if l else "grey" for w, l in zip(win_mc_conf_prauc, lose_mc_conf_prauc)])

plt.tight_layout()
plt.show()
