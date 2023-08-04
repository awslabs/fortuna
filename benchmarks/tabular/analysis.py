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

# TEMPERATURE SCALING
temp_scaling_nlls = [metrics["regression"][k]["temp_scaling"]["nll"] for k in metrics["regression"].keys()]
temp_scaling_quantiles_nlls = np.percentile(temp_scaling_nlls, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_temp_scaling_nlls = f"{np.sum(np.array(temp_scaling_nlls) / np.array(map_nlls) <= 1)} / {len(map_nlls)}"
med_improv_temp_scaling_nlls = f"{np.round(np.median((np.array(map_nlls) - np.array(temp_scaling_nlls)) / np.array(map_nlls)), 2)}"

temp_scaling_picp_errors = [np.abs(0.95 - metrics["regression"][k]["temp_scaling"]["picp"]) for k in metrics["regression"].keys()]
temp_scaling_quantiles_picp_errors = np.percentile(temp_scaling_picp_errors, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_temp_scaling_picp_errors = f"{np.sum(np.array(temp_scaling_picp_errors) / np.array(map_picp_errors) <= 1)} / {len(map_picp_errors)}"
med_improv_temp_scaling_picp_errors = f"{np.round(np.median((np.array(map_picp_errors) - np.array(temp_scaling_picp_errors)) / np.array(map_picp_errors)), 2)}"

# CQR
cqr_picp_errors = [np.abs(0.95 - metrics["regression"][k]["cqr"]["picp"]) for k in metrics["regression"].keys()]
cqr_quantiles_picp_errors = np.percentile(cqr_picp_errors, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_cqr_picp_errors = f"{np.sum(np.array(cqr_picp_errors) / np.array(map_picp_errors) <= 1)} / {len(map_picp_errors)}"
med_improv_cqr_picp_errors = f"{np.round(np.median((np.array(map_picp_errors) - np.array(cqr_picp_errors)) / np.array(map_picp_errors)), 2)}"

plt.figure(figsize=(10, 6))
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

map_brier = [metrics["classification"][k]["map"]["brier_score"] for k in metrics["classification"].keys()]
map_quantiles_brier = np.percentile(map_brier, [10, 20, 30, 40, 50, 60, 70, 80, 90])

map_ece = [metrics["classification"][k]["map"]["ece"] for k in metrics["classification"].keys()]
map_quantiles_ece = np.percentile(map_ece, [10, 20, 30, 40, 50, 60, 70, 80, 90])

map_rocauc = [metrics["classification"][k]["map"]["rocauc"] for k in metrics["classification"].keys() if "rocauc" in metrics["classification"][k]["map"]]
map_quantiles_rocauc = np.percentile(map_rocauc, [10, 20, 30, 40, 50, 60, 70, 80, 90])

map_prauc = [metrics["classification"][k]["map"]["prauc"] for k in metrics["classification"].keys() if "prauc" in metrics["classification"][k]["map"]]
map_quantiles_prauc = np.percentile(map_prauc, [10, 20, 30, 40, 50, 60, 70, 80, 90])

# TEMPERATURE SCALING
temp_scaling_nlls = [metrics["classification"][k]["temp_scaling"]["nll"] for k in metrics["classification"].keys()]
temp_scaling_quantiles_nlls = np.percentile(temp_scaling_nlls, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_temp_scaling_nlls = f"{np.sum(np.array(temp_scaling_nlls) / np.array(map_nlls) <= 1)} / {len(map_nlls)}"
med_improv_temp_scaling_nlls = f"{np.round(np.median((np.array(map_nlls) - np.array(temp_scaling_nlls)) / np.array(map_nlls)), 2)}"

temp_scaling_brier = [metrics["classification"][k]["temp_scaling"]["brier_score"] for k in metrics["classification"].keys()]
temp_scaling_quantiles_brier = np.percentile(temp_scaling_brier, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_temp_scaling_brier = f"{np.sum(np.array(temp_scaling_brier) / np.array(map_brier) <= 1)} / {len(map_brier)}"
med_improv_temp_scaling_brier = f"{np.round(np.median((np.array(map_brier) - np.array(temp_scaling_brier)) / np.array(map_brier)), 2)}"

temp_scaling_ece = [metrics["classification"][k]["temp_scaling"]["ece"] for k in metrics["classification"].keys()]
temp_scaling_quantiles_ece = np.percentile(temp_scaling_ece, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_temp_scaling_ece = f"{np.sum(np.array(temp_scaling_ece) / np.array(map_ece) <= 1)} / {len(map_ece)}"
med_improv_temp_scaling_ece = f"{np.round(np.median((np.array(map_ece) - np.array(temp_scaling_ece)) / np.array(map_ece)), 2)}"

temp_scaling_rocauc = [metrics["classification"][k]["temp_scaling"]["rocauc"] for k in metrics["classification"].keys() if "rocauc" in metrics["classification"][k]["temp_scaling"]]
temp_scaling_quantiles_rocauc = np.percentile(temp_scaling_rocauc, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_temp_scaling_rocauc = f"{np.sum(np.array(temp_scaling_rocauc) / np.array(map_rocauc) <= 1)} / {len(map_rocauc)}"
med_improv_temp_scaling_rocauc = f"{np.round(np.median((np.array(map_rocauc) - np.array(temp_scaling_rocauc)) / np.array(map_rocauc)), 2)}"

temp_scaling_prauc = [metrics["classification"][k]["temp_scaling"]["prauc"] for k in metrics["classification"].keys() if "prauc" in metrics["classification"][k]["temp_scaling"]]
temp_scaling_quantiles_prauc = np.percentile(temp_scaling_prauc, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_temp_scaling_prauc = f"{np.sum(np.array(temp_scaling_prauc) / np.array(map_prauc) <= 1)} / {len(map_prauc)}"
med_improv_temp_scaling_prauc = f"{np.round(np.median((np.array(map_prauc) - np.array(temp_scaling_prauc)) / np.array(map_prauc)), 2)}"

# MULTICALIBRATE CONF
mc_conf_ece = [metrics["classification"][k]["mc_conf"]["ece"] for k in metrics["classification"].keys()]
mc_conf_quantiles_ece = np.percentile(mc_conf_ece, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_mc_conf_ece = f"{np.sum(np.array(mc_conf_ece) / np.array(map_ece) <= 1)} / {len(map_ece)}"
med_improv_mc_conf_ece = f"{np.round(np.median((np.array(map_ece) - np.array(mc_conf_ece)) / np.array(map_ece)), 2)}"

# MULTICALIBRATE PROB
mc_prob_brier = [metrics["classification"][k]["mc_prob"]["brier_score"] for k in metrics["classification"].keys() if "brier_score" in metrics["classification"][k]["mc_prob"]]
mc_prob_quantiles_brier = np.percentile(mc_prob_brier, [10, 20, 30, 40, 50, 60, 70, 80, 90])
idx_overlap = [i for i, k in enumerate(metrics["classification"]) if len(metrics["classification"][k]["mc_prob"])]
winlose_mc_prob_brier = f"{np.sum(np.array(mc_prob_brier) / np.array(map_brier)[idx_overlap] <= 1)} / {len(idx_overlap)}"
med_improv_mc_prob_brier = f"{np.round(np.median((np.array(map_brier)[idx_overlap] - np.array(mc_prob_brier)) / np.array(map_brier)[idx_overlap]), 2)}"

mc_prob_rocauc = [metrics["classification"][k]["mc_prob"]["rocauc"] for k in metrics["classification"].keys() if "rocauc" in metrics["classification"][k]["mc_prob"]]
mc_prob_quantiles_rocauc = np.percentile(mc_prob_rocauc, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_mc_prob_rocauc = f"{np.sum(np.array(mc_prob_rocauc) / np.array(map_rocauc) <= 1)} / {len(map_rocauc)}"
med_improv_mc_prob_rocauc = f"{np.round(np.median((np.array(map_rocauc) - np.array(mc_prob_rocauc)) / np.array(map_rocauc)), 2)}"

mc_prob_prauc = [metrics["classification"][k]["mc_prob"]["prauc"] for k in metrics["classification"].keys() if "prauc" in metrics["classification"][k]["mc_prob"]]
mc_prob_quantiles_prauc = np.percentile(mc_prob_prauc, [10, 20, 30, 40, 50, 60, 70, 80, 90])
winlose_mc_prob_prauc = f"{np.sum(np.array(mc_prob_prauc) / np.array(map_prauc) <= 1)} / {len(map_prauc)}"
med_improv_mc_prob_prauc = f"{np.round(np.median((np.array(map_prauc) - np.array(mc_prob_prauc)) / np.array(map_prauc)), 2)}"


plt.figure(figsize=(15, 8))
plt.suptitle("Quantile-quantile plots of metrics on classification datasets")

plt.subplot(3, 5, 1)
plt.title("NLL")
plt.scatter(map_quantiles_nlls, temp_scaling_quantiles_nlls, s=3)
_min, _max = min(map_quantiles_nlls.min(), temp_scaling_quantiles_nlls.min()), max(map_quantiles_nlls.max(), temp_scaling_quantiles_nlls.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("temp scaling quantiles")
plt.grid()

plt.subplot(3, 5, 2)
plt.title("ECE")
plt.scatter(map_quantiles_ece, temp_scaling_quantiles_ece, s=3)
_min, _max = min(map_quantiles_ece.min(), temp_scaling_quantiles_ece.min()), max(map_quantiles_ece.max(), temp_scaling_quantiles_ece.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("temp scaling quantiles")
plt.grid()

plt.subplot(3, 5, 3)
plt.title("Brier score")
plt.scatter(map_quantiles_brier, temp_scaling_quantiles_brier, s=3)
_min, _max = min(map_quantiles_brier.min(), temp_scaling_quantiles_brier.min()), max(map_quantiles_brier.max(), temp_scaling_quantiles_brier.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("temp scaling quantiles")
plt.grid()

plt.subplot(3, 5, 4)
plt.title("ROCAUC")
plt.scatter(map_quantiles_rocauc, temp_scaling_quantiles_rocauc, s=3)
_min, _max = min(map_quantiles_rocauc.min(), temp_scaling_quantiles_rocauc.min()), max(map_quantiles_rocauc.max(), temp_scaling_quantiles_rocauc.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("temp scaling quantiles")
plt.grid()

plt.subplot(3, 5, 5)
plt.title("PRAUC")
plt.scatter(map_quantiles_prauc, temp_scaling_quantiles_prauc, s=3)
_min, _max = min(map_quantiles_prauc.min(), temp_scaling_quantiles_prauc.min()), max(map_quantiles_prauc.max(), temp_scaling_quantiles_prauc.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("temp scaling quantiles")
plt.grid()

plt.subplot(3, 5, 7)
plt.title("ECE")
plt.scatter(map_quantiles_ece, mc_conf_quantiles_ece, s=3)
_min, _max = min(map_quantiles_ece.min(), mc_conf_quantiles_ece.min()), max(map_quantiles_ece.max(), mc_conf_quantiles_ece.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("MC-conf quantiles")
plt.grid()

plt.subplot(3, 5, 13)
plt.title("Brier score")
plt.scatter(map_quantiles_brier, mc_prob_quantiles_brier, s=3)
_min, _max = min(map_quantiles_brier.min(), mc_prob_quantiles_brier.min()), max(map_quantiles_brier.max(), mc_prob_quantiles_brier.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("MC-prob quantiles")
plt.grid()

plt.subplot(3, 5, 14)
plt.title("ROCAUC")
plt.scatter(map_quantiles_rocauc, mc_prob_quantiles_rocauc, s=3)
_min, _max = min(map_quantiles_rocauc.min(), mc_prob_quantiles_rocauc.min()), max(map_quantiles_rocauc.max(), mc_prob_quantiles_rocauc.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("MC-prob quantiles")
plt.grid()

plt.subplot(3, 5, 15)
plt.title("PRAUC")
plt.scatter(map_quantiles_prauc, mc_prob_quantiles_prauc, s=3)
_min, _max = min(map_quantiles_prauc.min(), mc_prob_quantiles_prauc.min()), max(map_quantiles_prauc.max(), mc_prob_quantiles_prauc.max())
plt.plot([_min, _max], [_min, _max], color='gray', linestyle="--", alpha=0.2)
plt.xlabel("MAP quantiles")
plt.ylabel("MC-prob quantiles")
plt.grid()

plt.tight_layout()
plt.show()

print("\n\n~~~CLASSIFICATION~~~\n")
print("## TEMPERATURE SCALING ##")
print(f"Fraction of times temp_scaling is at least on a par w.r.t. the NLL: {winlose_temp_scaling_nlls}")
print(f"Fraction of times temp_scaling is at least on a par w.r.t. the Brier score: {winlose_temp_scaling_brier}")
print(f"Fraction of times temp_scaling is at least on a par w.r.t. the ECE: {winlose_temp_scaling_ece}")
print(f"Fraction of times temp_scaling is at least on a par w.r.t. the ROCAUC: {winlose_temp_scaling_rocauc}")
print(f"Fraction of times temp_scaling is at least on a par w.r.t. the PRAUC: {winlose_temp_scaling_prauc}")
print()
print(f"Median of relative NLL improvement given by temp_scaling: {med_improv_temp_scaling_nlls}")
print(f"Median of relative Brier score improvement given by temp_scaling: {med_improv_temp_scaling_brier}")
print(f"Median of relative ECE improvement given by temp_scaling: {med_improv_temp_scaling_ece}")
print(f"Median of relative ROCAUC improvement given by temp_scaling: {med_improv_temp_scaling_rocauc}")
print(f"Median of relative PRAUC improvement given by temp_scaling: {med_improv_temp_scaling_prauc}")
print()
print()
print("## MC-CONF ##")
print(f"Fraction of times MC-conf is at least on a par w.r.t. the ECE: {winlose_mc_conf_ece}")
print()
print(f"Median of relative ECE improvement given by temp_scaling: {med_improv_mc_conf_ece}")
print()
print()
print("## MC-PROB ##")
print(f"Fraction of times MC-prob is at least on a par w.r.t. the Brier score: {winlose_mc_prob_brier}")
print(f"Fraction of times MC-prob is at least on a par w.r.t. the ROCAUC: {winlose_mc_prob_rocauc}")
print(f"Fraction of times MC-prob is at least on a par w.r.t. the PRAUC: {winlose_mc_prob_prauc}")
print()
print(f"Median of relative Brier score improvement given by temp_scaling: {med_improv_mc_prob_brier}")
print(f"Median of relative ROCAUC improvement given by temp_scaling: {med_improv_mc_prob_rocauc}")
print(f"Median of relative PRAUC improvement given by temp_scaling: {med_improv_mc_prob_prauc}")
