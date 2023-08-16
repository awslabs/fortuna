import json
import numpy as np
import matplotlib.pyplot as plt

with open("tabular_results.json", 'r') as j:
    metrics = json.loads(j.read())

# ~~~REGRESSION~~~

# MAP
map_nlls = [metrics["regression"][k]["map"]["nll"] for k in metrics["regression"].keys()]
map_quantiles_nlls = np.percentile(map_nlls, [10, 20, 30, 40, 50, 60, 70, 80, 90])

map_picp_errors = [np.abs(0.95 - metrics["regression"][k]["map"]["picp"]) for k in metrics["regression"].keys()]
map_quantiles_picp_errors = np.percentile(map_picp_errors, [10, 20, 30, 40, 50, 60, 70, 80, 90])

map_times = [metrics["regression"][k]["map"]["time"] for k in metrics["regression"].keys()]

# TEMPERATURE SCALING
temp_scaling_nlls = [metrics["regression"][k]["temp_scaling"]["nll"] for k in metrics["regression"].keys()]
temp_scaling_quantiles_nlls = np.percentile(temp_scaling_nlls, [10, 20, 30, 40, 50, 60, 70, 80, 90])
win_temp_scaling_nlls = np.sum(np.array(temp_scaling_nlls) / np.array(map_nlls) <= 1)
winlose_temp_scaling_nlls = f"{win_temp_scaling_nlls} / {len(map_nlls) - win_temp_scaling_nlls}"
rel_improve_temp_scaling_nlls = (np.array(map_nlls) - np.array(temp_scaling_nlls)) / np.array(map_nlls)
max_loss_temp_scaling_nlls = str(np.round(100 * np.abs(np.max(rel_improve_temp_scaling_nlls[rel_improve_temp_scaling_nlls < 0])), 2)) + "%"
med_improv_temp_scaling_nlls = f"{np.round(np.median(rel_improve_temp_scaling_nlls), 2)}"


temp_scaling_picp_errors = [np.abs(0.95 - metrics["regression"][k]["temp_scaling"]["picp"]) for k in metrics["regression"].keys()]
temp_scaling_quantiles_picp_errors = np.percentile(temp_scaling_picp_errors, [10, 20, 30, 40, 50, 60, 70, 80, 90])
win_temp_scaling_picp_errors = np.sum(np.array(temp_scaling_picp_errors) / np.array(map_picp_errors) <= 1)
winlose_temp_scaling_picp_errors = f"{win_temp_scaling_picp_errors} / {len(map_picp_errors) - win_temp_scaling_picp_errors}"
rel_improve_temp_scaling_picp_errors = (np.array(map_picp_errors) - np.array(temp_scaling_picp_errors)) / np.array(map_picp_errors)
max_loss_temp_scaling_picp_errors = str(np.round(100 * np.abs(np.max(rel_improve_temp_scaling_picp_errors[rel_improve_temp_scaling_picp_errors < 0])), 2)) + "%"
med_improv_temp_scaling_picp_errors = f"{np.round(np.median(rel_improve_temp_scaling_picp_errors), 2)}"

temp_scaling_times = [metrics["regression"][k]["temp_scaling"]["time"] for k in metrics["regression"].keys()]

# CQR
cqr_picp_errors = [np.abs(0.95 - metrics["regression"][k]["cqr"]["picp"]) for k in metrics["regression"].keys()]
cqr_quantiles_picp_errors = np.percentile(cqr_picp_errors, [10, 20, 30, 40, 50, 60, 70, 80, 90])
win_cqr_picp_errors = np.sum(np.array(cqr_picp_errors) / np.array(map_picp_errors) <= 1)
winlose_cqr_picp_errors = f"{win_cqr_picp_errors} / {len(map_picp_errors) - win_cqr_picp_errors}"
rel_improve_cqr_picp_errors = (np.array(map_picp_errors) - np.array(cqr_picp_errors)) / np.array(map_picp_errors)
max_loss_cqr_picp_errors = str(np.round(100 * np.abs(np.max(rel_improve_cqr_picp_errors[rel_improve_cqr_picp_errors < 0])), 2)) + "%"
med_improv_cqr_picp_errors = f"{np.round(np.median(rel_improve_cqr_picp_errors), 2)}"

cqr_times = [metrics["regression"][k]["cqr"]["time"] for k in metrics["regression"].keys()]

# # TEMPERED CQR
temp_cqr_picp_errors = [np.abs(0.95 - metrics["regression"][k]["temp_cqr"]["picp"]) for k in metrics["regression"].keys()]
temp_cqr_quantiles_picp_errors = np.percentile(temp_cqr_picp_errors, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_temp_cqr_picp_errors = f"{np.sum(np.array(temp_cqr_picp_errors) / np.array(map_picp_errors) <= 1)} / {len(map_picp_errors)}"
med_improv_temp_cqr_picp_errors = f"{np.round(np.median((np.array(map_picp_errors) - np.array(temp_cqr_picp_errors)) / np.array(map_picp_errors)), 2)}"

temp_cqr_times = [metrics["regression"][k]["temp_cqr"]["time"] for k in metrics["regression"].keys()]

plt.figure(figsize=(8, 6))
plt.suptitle("Quantile-quantile plots of metrics on regression datasets")

plt.subplot(2, 2, 1)
plt.title("NLL")
plt.scatter(map_quantiles_nlls, temp_scaling_quantiles_nlls, s=3)
_min, _max = min(map_quantiles_nlls.min(), temp_scaling_quantiles_nlls.min()), max(map_quantiles_nlls.max(), temp_scaling_quantiles_nlls.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("temp scaling quantiles")
plt.grid()

plt.subplot(2, 2, 2)
plt.title("PICP absolute error")
plt.scatter(map_quantiles_picp_errors, temp_scaling_quantiles_picp_errors, s=3)
_min, _max = min(map_quantiles_picp_errors.min(), temp_scaling_quantiles_picp_errors.min()), max(map_quantiles_picp_errors.max(), temp_scaling_quantiles_picp_errors.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("temp scaling quantiles")
plt.grid()

plt.subplot(2, 2, 4)
plt.title("PICP absolute error")
plt.scatter(map_quantiles_picp_errors, cqr_quantiles_picp_errors, s=3)
_min, _max = min(map_quantiles_picp_errors.min(), cqr_quantiles_picp_errors.min()), max(map_quantiles_picp_errors.max(), cqr_quantiles_picp_errors.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("CQR quantiles")
plt.grid()

plt.tight_layout()

plt.show()

print("~~~REGRESSION~~~\n")
print("## TEMPERATURE SCALING ##")
print(f"Fraction of times temp_scaling is at least on a par w.r.t. the NLL: {winlose_temp_scaling_nlls}")
print(f"Fraction of times temp_scaling is at least on a par w.r.t. the PICP error: {winlose_temp_scaling_picp_errors}")
print()
print(f"Median of relative NLL improvement given by temp_scaling: {med_improv_temp_scaling_nlls}")
print(f"Median of relative PICP error improvement given by temp_scaling: {med_improv_temp_scaling_picp_errors}")
print()
print()
print("## CQR ##")
print(f"Fraction of times CQR is at least on a par w.r.t. the PICP error: {winlose_cqr_picp_errors}")
print()
print(f"Median of relative PICP error improvement given by temp_scaling: {med_improv_cqr_picp_errors}")


# ~~~CLASSIFICATION~~~

# MAP
map_nlls = [metrics["classification"][k]["map"]["nll"] for k in metrics["classification"].keys()]
map_quantiles_nlls = np.percentile(map_nlls, [10, 20, 30, 40, 50, 60, 70, 80, 90])

map_mse = [metrics["classification"][k]["map"]["mse"] for k in metrics["classification"].keys()]
map_quantiles_mse = np.percentile(map_mse, [10, 20, 30, 40, 50, 60, 70, 80, 90])

map_ece = [metrics["classification"][k]["map"]["ece"] for k in metrics["classification"].keys()]
map_quantiles_ece = np.percentile(map_ece, [10, 20, 30, 40, 50, 60, 70, 80, 90])

map_rocauc = [metrics["classification"][k]["map"]["rocauc"] for k in metrics["classification"].keys() if "rocauc" in metrics["classification"][k]["map"]]
map_quantiles_rocauc = np.percentile(map_rocauc, [10, 20, 30, 40, 50, 60, 70, 80, 90])

map_prauc = [metrics["classification"][k]["map"]["prauc"] for k in metrics["classification"].keys() if "prauc" in metrics["classification"][k]["map"]]
map_quantiles_prauc = np.percentile(map_prauc, [10, 20, 30, 40, 50, 60, 70, 80, 90])

map_acc = [metrics["classification"][k]["map"]["accuracy"] for k in metrics["classification"].keys()]

map_times = [metrics["regression"][k]["map"]["time"] for k in metrics["regression"].keys()]

# TEMPERATURE SCALING
temp_scaling_nlls = [metrics["classification"][k]["temp_scaling"]["nll"] for k in metrics["classification"].keys()]
temp_scaling_quantiles_nlls = np.percentile(temp_scaling_nlls, [10, 20, 30, 40, 50, 60, 70, 80, 90])
win_temp_scaling_nlls = np.sum(np.array(temp_scaling_nlls) / np.array(map_nlls) <= 1)
winlose_temp_scaling_nlls = f"{win_temp_scaling_nlls} / {len(map_nlls) - win_temp_scaling_nlls}"
rel_improve_temp_scaling_nlls = (np.array(map_nlls) - np.array(temp_scaling_nlls)) / np.array(map_nlls)
max_loss_temp_scaling_nlls = str(np.round(100 * np.abs(np.max(rel_improve_temp_scaling_nlls[rel_improve_temp_scaling_nlls < 0])), 2)) + "%"
med_improv_temp_scaling_nlls = f"{np.round(np.median(rel_improve_temp_scaling_nlls), 2)}"

temp_scaling_mse = [metrics["classification"][k]["temp_scaling"]["mse"] for k in metrics["classification"].keys()]
temp_scaling_quantiles_mse = np.percentile(temp_scaling_mse, [10, 20, 30, 40, 50, 60, 70, 80, 90])
win_temp_scaling_mse = np.sum(np.array(temp_scaling_mse) / np.array(map_mse) <= 1)
winlose_temp_scaling_mse = f"{win_temp_scaling_mse} / {len(map_mse) - win_temp_scaling_mse}"
rel_improve_temp_scaling_mse = (np.array(map_mse) - np.array(temp_scaling_mse)) / np.array(map_mse)
max_loss_temp_scaling_mse = str(np.round(100 * np.abs(np.max(rel_improve_temp_scaling_mse[rel_improve_temp_scaling_mse < 0])), 2)) + "%"
med_improv_temp_scaling_mse = f"{np.round(np.median(rel_improve_temp_scaling_mse), 2)}"

temp_scaling_ece = [metrics["classification"][k]["temp_scaling"]["ece"] for k in metrics["classification"].keys()]
temp_scaling_quantiles_ece = np.percentile(temp_scaling_ece, [10, 20, 30, 40, 50, 60, 70, 80, 90])
win_temp_scaling_ece = np.sum(np.array(temp_scaling_ece) / np.array(map_ece) <= 1)
winlose_temp_scaling_ece = f"{win_temp_scaling_ece} / {len(map_ece) - win_temp_scaling_ece}"
rel_improve_temp_scaling_ece = (np.array(map_ece) - np.array(temp_scaling_ece)) / np.array(map_ece)
max_loss_temp_scaling_ece = str(np.round(100 * np.abs(np.max(rel_improve_temp_scaling_ece[rel_improve_temp_scaling_ece < 0])), 2)) + "%"
med_improv_temp_scaling_ece = f"{np.round(np.median(rel_improve_temp_scaling_ece), 2)}"

temp_scaling_rocauc = [metrics["classification"][k]["temp_scaling"]["rocauc"] for k in metrics["classification"].keys() if "rocauc" in metrics["classification"][k]["temp_scaling"]]
temp_scaling_quantiles_rocauc = np.percentile(temp_scaling_rocauc, [10, 20, 30, 40, 50, 60, 70, 80, 90])
win_temp_scaling_rocauc = np.sum(np.array(temp_scaling_rocauc) / np.array(map_rocauc) <= 1)
winlose_temp_scaling_rocauc = f"{win_temp_scaling_rocauc} / {len(map_rocauc) - win_temp_scaling_rocauc}"
rel_improve_temp_scaling_rocauc = (np.array(map_rocauc) - np.array(temp_scaling_rocauc)) / np.array(map_rocauc)
max_loss_temp_scaling_rocauc = str(np.round(100 * np.abs(np.max(rel_improve_temp_scaling_rocauc[rel_improve_temp_scaling_rocauc < 0])), 2)) + "%"
med_improv_temp_scaling_rocauc = f"{np.round(np.median(rel_improve_temp_scaling_rocauc), 2)}"

temp_scaling_prauc = [metrics["classification"][k]["temp_scaling"]["prauc"] for k in metrics["classification"].keys() if "prauc" in metrics["classification"][k]["temp_scaling"]]
temp_scaling_quantiles_prauc = np.percentile(temp_scaling_prauc, [10, 20, 30, 40, 50, 60, 70, 80, 90])
win_temp_scaling_prauc = np.sum(np.array(temp_scaling_prauc) / np.array(map_prauc) <= 1)
winlose_temp_scaling_prauc = f"{win_temp_scaling_prauc} / {len(map_prauc) - win_temp_scaling_prauc}"
rel_improve_temp_scaling_prauc = (np.array(map_prauc) - np.array(temp_scaling_prauc)) / np.array(map_prauc)
max_loss_temp_scaling_prauc = str(np.round(100 * np.abs(np.max(rel_improve_temp_scaling_prauc[rel_improve_temp_scaling_prauc < 0])), 2)) + "%"
med_improv_temp_scaling_prauc = f"{np.round(np.median(rel_improve_temp_scaling_prauc), 2)}"

temp_scaling_acc = [metrics["classification"][k]["temp_scaling"]["accuracy"] for k in metrics["classification"].keys()]

temp_scaling_times = [metrics["classification"][k]["temp_scaling"]["time"] for k in metrics["classification"].keys()]

# MULTICALIBRATE CONF
mc_conf_nlls = [metrics["classification"][k]["mc_conf"]["nll"] for k in metrics["classification"].keys()]
mc_conf_nlls = np.array(mc_conf_nlls)
mc_conf_quantiles_nlls = np.percentile(mc_conf_nlls, [10, 20, 30, 40, 50, 60, 70, 80, 90])
win_mc_conf_nlls = np.sum(np.array(mc_conf_nlls) / np.array(np.array(map_nlls)) <= 1)
winlose_mc_conf_nlls = f"{win_mc_conf_nlls} / {len(mc_conf_nlls) - win_mc_conf_nlls}"
rel_improve_mc_conf_nlls = (np.array(map_nlls) - np.array(mc_conf_nlls)) / np.array(map_nlls)
max_loss_mc_conf_nlls = str(np.round(100 * np.abs(np.max(rel_improve_mc_conf_nlls[rel_improve_mc_conf_nlls < 0])), 2)) + "%"
med_improv_mc_conf_nlls = f"{np.round(np.median(rel_improve_mc_conf_nlls), 2)}"

mc_conf_mse = [metrics["classification"][k]["mc_conf"]["mse"] for k in metrics["classification"].keys()]
mc_conf_mse = np.array(mc_conf_mse)
mc_conf_quantiles_mse = np.percentile(mc_conf_mse, [10, 20, 30, 40, 50, 60, 70, 80, 90])
win_mc_conf_mse = np.sum(np.array(mc_conf_mse) / np.array(np.array(map_mse)) <= 1)
winlose_mc_conf_mse = f"{win_mc_conf_mse} / {len(mc_conf_mse) - win_mc_conf_mse}"
rel_improve_mc_conf_mse = (np.array(map_mse) - np.array(mc_conf_mse)) / np.array(map_mse)
max_loss_mc_conf_mse = str(np.round(100 * np.abs(np.max(rel_improve_mc_conf_mse[rel_improve_mc_conf_mse < 0])), 2)) + "%"
med_improv_mc_conf_mse = f"{np.round(np.median(rel_improve_mc_conf_mse), 2)}"

mc_conf_ece = [metrics["classification"][k]["mc_conf"]["ece"] for k in metrics["classification"].keys()]
mc_conf_quantiles_ece = np.percentile(mc_conf_ece, [10, 20, 30, 40, 50, 60, 70, 80, 90])
win_mc_conf_ece = np.sum(np.array(mc_conf_ece) / np.array(map_ece) <= 1)
winlose_mc_conf_ece = f"{win_mc_conf_ece} / {len(map_ece) - win_mc_conf_ece}"
rel_improve_mc_conf_ece = (np.array(map_ece) - np.array(mc_conf_ece)) / np.array(map_ece)
max_loss_mc_conf_ece = str(np.round(100 * np.abs(np.max(rel_improve_mc_conf_ece[rel_improve_mc_conf_ece < 0])), 2)) + "%"
med_improv_mc_conf_ece = f"{np.round(np.median(rel_improve_mc_conf_ece), 2)}"

mc_conf_rocauc = [metrics["classification"][k]["mc_conf"]["rocauc"] for k in metrics["classification"].keys()]
mc_conf_rocauc = np.array(mc_conf_rocauc)
mc_conf_quantiles_rocauc = np.percentile(mc_conf_rocauc, [10, 20, 30, 40, 50, 60, 70, 80, 90])
win_mc_conf_rocauc = np.sum(np.array(mc_conf_rocauc) / np.array(np.array(map_rocauc)) <= 1)
winlose_mc_conf_rocauc = f"{win_mc_conf_rocauc} / {len(mc_conf_rocauc) - win_mc_conf_rocauc}"
rel_improve_mc_conf_rocauc = (np.array(map_rocauc) - np.array(mc_conf_rocauc)) / np.array(map_rocauc)
max_loss_mc_conf_rocauc = str(np.round(100 * np.abs(np.max(rel_improve_mc_conf_rocauc[rel_improve_mc_conf_rocauc < 0])), 2)) + "%"
med_improv_mc_conf_rocauc = f"{np.round(np.median(rel_improve_mc_conf_rocauc), 2)}"

mc_conf_prauc = [metrics["classification"][k]["mc_conf"]["prauc"] for k in metrics["classification"].keys()]
mc_conf_prauc = np.array(mc_conf_prauc)
mc_conf_quantiles_prauc = np.percentile(mc_conf_prauc, [10, 20, 30, 40, 50, 60, 70, 80, 90])
win_mc_conf_prauc = np.sum(np.array(mc_conf_prauc) / np.array(np.array(map_prauc)) <= 1)
winlose_mc_conf_prauc = f"{win_mc_conf_prauc} / {len(mc_conf_prauc) - win_mc_conf_prauc}"
rel_improve_mc_conf_prauc = (np.array(map_prauc) - np.array(mc_conf_prauc)) / np.array(map_prauc)
max_loss_mc_conf_prauc = str(np.round(100 * np.abs(np.max(rel_improve_mc_conf_prauc[rel_improve_mc_conf_prauc < 0])), 2)) + "%"
med_improv_mc_conf_prauc = f"{np.round(np.median(rel_improve_mc_conf_prauc), 2)}"

mc_conf_acc = [metrics["classification"][k]["mc_conf"]["accuracy"] for k in metrics["classification"].keys()]

mc_conf_times = [metrics["classification"][k]["mc_conf"]["time"] for k in metrics["classification"].keys()]

# TEMPERED MULTICALIBRATE CONF
temp_mc_conf_nlls = [metrics["classification"][k]["temp_mc_conf"]["nll"] for k in metrics["classification"].keys()]
temp_mc_conf_nlls = np.array(temp_mc_conf_nlls)
temp_mc_conf_quantiles_nlls = np.percentile(temp_mc_conf_nlls, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_temp_mc_conf_nlls = f"{np.sum(np.array(temp_mc_conf_nlls) / np.array(np.array(map_nlls)) <= 1)} / {len(temp_mc_conf_nlls)}"
med_improv_temp_mc_conf_nlls = f"{np.round(np.median((np.array(map_nlls) - np.array(temp_mc_conf_nlls)) / np.array(map_nlls)), 2)}"

temp_mc_conf_mse = [metrics["classification"][k]["temp_mc_conf"]["mse"] for k in metrics["classification"].keys()]
temp_mc_conf_mse = np.array(temp_mc_conf_mse)
temp_mc_conf_quantiles_mse = np.percentile(temp_mc_conf_mse, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_temp_mc_conf_mse = f"{np.sum(np.array(temp_mc_conf_mse) / np.array(np.array(map_mse)) <= 1)} / {len(temp_mc_conf_mse)}"
med_improv_temp_mc_conf_mse = f"{np.round(np.median((np.array(map_mse) - np.array(temp_mc_conf_mse)) / np.array(map_mse)), 2)}"

temp_mc_conf_ece = [metrics["classification"][k]["temp_mc_conf"]["ece"] for k in metrics["classification"].keys()]
temp_mc_conf_quantiles_ece = np.percentile(temp_mc_conf_ece, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_temp_mc_conf_ece = f"{np.sum(np.array(temp_mc_conf_ece) / np.array(map_ece) <= 1)} / {len(map_ece)}"
med_improv_temp_mc_conf_ece = f"{np.round(np.median((np.array(map_ece) - np.array(temp_mc_conf_ece)) / np.array(map_ece)), 2)}"

temp_mc_conf_rocauc = [metrics["classification"][k]["temp_mc_conf"]["rocauc"] for k in metrics["classification"].keys()]
temp_mc_conf_rocauc = np.array(temp_mc_conf_rocauc)
temp_mc_conf_quantiles_rocauc = np.percentile(temp_mc_conf_rocauc, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_temp_mc_conf_rocauc = f"{np.sum(np.array(temp_mc_conf_rocauc) / np.array(np.array(map_rocauc)) <= 1)} / {len(temp_mc_conf_rocauc)}"
med_improv_temp_mc_conf_rocauc = f"{np.round(np.median((np.array(map_rocauc) - np.array(temp_mc_conf_rocauc)) / np.array(map_rocauc)), 2)}"

temp_mc_conf_prauc = [metrics["classification"][k]["temp_mc_conf"]["prauc"] for k in metrics["classification"].keys()]
temp_mc_conf_prauc = np.array(temp_mc_conf_prauc)
temp_mc_conf_quantiles_prauc = np.percentile(temp_mc_conf_prauc, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_temp_mc_conf_prauc = f"{np.sum(np.array(temp_mc_conf_prauc) / np.array(np.array(map_prauc)) <= 1)} / {len(temp_mc_conf_prauc)}"
med_improv_temp_mc_conf_prauc = f"{np.round(np.median((np.array(map_prauc) - np.array(temp_mc_conf_prauc)) / np.array(map_prauc)), 2)}"

temp_mc_conf_acc = [metrics["classification"][k]["temp_mc_conf"]["accuracy"] for k in metrics["classification"].keys()]

temp_mc_conf_times = [metrics["classification"][k]["temp_mc_conf"]["time"] for k in metrics["classification"].keys()]

# MULTICALIBRATE PROB
idx_overlap = [i for i, k in enumerate(metrics["classification"]) if len(metrics["classification"][k]["mc_prob"])]

mc_prob_nlls = [metrics["classification"][k]["mc_prob"]["nll"] for k in metrics["classification"].keys() if "nll" in metrics["classification"][k]["mc_prob"]]
mc_prob_quantiles_nlls = np.percentile(mc_prob_nlls, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_mc_prob_nlls = f"{np.sum(np.array(mc_prob_nlls) / np.array(map_nlls)[idx_overlap] <= 1)} / {len(idx_overlap)}"
med_improv_mc_prob_nlls = f"{np.round(np.median((np.array(map_nlls)[idx_overlap] - np.array(mc_prob_nlls)) / np.array(map_nlls)[idx_overlap]), 2)}"

mc_prob_mse = [metrics["classification"][k]["mc_prob"]["mse"] for k in metrics["classification"].keys() if "mse" in metrics["classification"][k]["mc_prob"]]
mc_prob_quantiles_mse = np.percentile(mc_prob_mse, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_mc_prob_mse = f"{np.sum(np.array(mc_prob_mse) / np.array(map_mse)[idx_overlap] <= 1)} / {len(idx_overlap)}"
med_improv_mc_prob_mse = f"{np.round(np.median((np.array(map_mse)[idx_overlap] - np.array(mc_prob_mse)) / np.array(map_mse)[idx_overlap]), 2)}"

mc_prob_ece = [metrics["classification"][k]["mc_prob"]["ece"] for k in metrics["classification"].keys() if "ece" in metrics["classification"][k]["mc_prob"]]
mc_prob_quantiles_ece = np.percentile(mc_prob_ece, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_mc_prob_ece = f"{np.sum(np.array(mc_prob_ece) / np.array(map_ece)[idx_overlap] <= 1)} / {len(idx_overlap)}"
med_improv_mc_prob_ece = f"{np.round(np.median((np.array(map_ece)[idx_overlap] - np.array(mc_prob_ece)) / np.array(map_ece)[idx_overlap]), 2)}"

mc_prob_rocauc = [metrics["classification"][k]["mc_prob"]["rocauc"] for k in metrics["classification"].keys() if "rocauc" in metrics["classification"][k]["mc_prob"]]
mc_prob_quantiles_rocauc = np.percentile(mc_prob_rocauc, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_mc_prob_rocauc = f"{np.sum(np.array(mc_prob_rocauc) / np.array(map_rocauc)[idx_overlap] <= 1)} / {len(idx_overlap)}"
med_improv_mc_prob_rocauc = f"{np.round(np.median((np.array(map_rocauc)[idx_overlap] - np.array(mc_prob_rocauc)) / np.array(map_rocauc)[idx_overlap]), 2)}"

mc_prob_prauc = [metrics["classification"][k]["mc_prob"]["prauc"] for k in metrics["classification"].keys() if "prauc" in metrics["classification"][k]["mc_prob"]]
mc_prob_quantiles_prauc = np.percentile(mc_prob_prauc, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_mc_prob_prauc = f"{np.sum(np.array(mc_prob_prauc) / np.array(map_prauc)[idx_overlap] <= 1)} / {len(idx_overlap)}"
med_improv_mc_prob_prauc = f"{np.round(np.median((np.array(map_prauc)[idx_overlap] - np.array(mc_prob_prauc)) / np.array(map_prauc)[idx_overlap]), 2)}"

mc_prob_acc = [metrics["classification"][k]["mc_prob"]["accuracy"] for k in metrics["classification"].keys() if "accuracy" in metrics["classification"][k]["mc_prob"]]

mc_prob_times = [metrics["classification"][k]["mc_prob"]["time"] for k in metrics["classification"].keys() if "time" in metrics["classification"][k]["mc_prob"]]

plt.figure(figsize=(10, 6))
plt.suptitle("Quantile-quantile plots of metrics on classification datasets")

plt.subplot(2, 4, 1)
plt.title("NLL")
plt.scatter(map_quantiles_nlls, temp_scaling_quantiles_nlls, s=3)
_min, _max = min(map_quantiles_nlls.min(), temp_scaling_quantiles_nlls.min()), max(map_quantiles_nlls.max(), temp_scaling_quantiles_nlls.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("temp scaling quantiles")
plt.grid()

plt.subplot(2, 4, 2)
plt.title("MSE")
plt.scatter(map_quantiles_mse, temp_scaling_quantiles_mse, s=3)
_min, _max = min(map_quantiles_mse.min(), temp_scaling_quantiles_mse.min()), max(map_quantiles_mse.max(), temp_scaling_quantiles_mse.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("temp scaling quantiles")
plt.grid()

plt.subplot(2, 4, 3)
plt.title("ROCAUC")
plt.scatter(map_quantiles_rocauc, temp_scaling_quantiles_rocauc, s=3)
_min, _max = min(map_quantiles_rocauc.min(), temp_scaling_quantiles_rocauc.min()), max(map_quantiles_rocauc.max(), temp_scaling_quantiles_rocauc.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("temp scaling quantiles")
plt.grid()

plt.subplot(2, 4, 4)
plt.title("PRAUC")
plt.scatter(map_quantiles_prauc, temp_scaling_quantiles_prauc, s=3)
_min, _max = min(map_quantiles_prauc.min(), temp_scaling_quantiles_prauc.min()), max(map_quantiles_prauc.max(), temp_scaling_quantiles_prauc.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("temp scaling quantiles")
plt.grid()

plt.subplot(2, 4, 5)
plt.title("NLL")
plt.scatter(map_quantiles_nlls, mc_conf_quantiles_nlls, s=3)
_min, _max = min(map_quantiles_nlls.min(), mc_conf_quantiles_nlls.min()), max(map_quantiles_nlls.max(), mc_conf_quantiles_nlls.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("TLMC quantiles")
plt.grid()

plt.subplot(2, 4, 6)
plt.title("ECE")
plt.scatter(map_quantiles_ece, mc_conf_quantiles_ece, s=3)
_min, _max = min(map_quantiles_ece.min(), mc_conf_quantiles_ece.min()), max(map_quantiles_ece.max(), mc_conf_quantiles_ece.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("TLMC quantiles")
plt.grid()

plt.subplot(2, 4, 6)
plt.title("MSE")
plt.scatter(map_quantiles_mse, mc_conf_quantiles_mse, s=3)
_min, _max = min(map_quantiles_mse.min(), mc_conf_quantiles_mse.min()), max(map_quantiles_mse.max(), mc_conf_quantiles_mse.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("TLMC quantiles")
plt.grid()

plt.subplot(2, 4, 7)
plt.title("ROCAUC")
plt.scatter(map_quantiles_rocauc, mc_conf_quantiles_rocauc, s=3)
_min, _max = min(map_quantiles_rocauc.min(), mc_conf_quantiles_rocauc.min()), max(map_quantiles_rocauc.max(), mc_conf_quantiles_rocauc.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("TLMC quantiles")
plt.grid()

plt.subplot(2, 4, 8)
plt.title("PRAUC")
plt.scatter(map_quantiles_prauc, mc_conf_quantiles_prauc, s=3)
_min, _max = min(map_quantiles_prauc.min(), mc_conf_quantiles_prauc.min()), max(map_quantiles_prauc.max(), mc_conf_quantiles_prauc.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("TLMC quantiles")
plt.grid()

plt.tight_layout()
plt.show()
