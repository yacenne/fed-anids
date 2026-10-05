import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)


def compute_statistical_threshold(val_errors, method="percentile", percentile=80.0, factor=1.0):
    """
    Computes anomaly threshold from validation benign reconstruction errors.
    
    Methods:
    - 'percentile': Sets threshold at the given percentile of benign validation errors
                    (e.g., 80th-85th percentile achieves ~94-96% F1 on CIC-IDS2017).
    - 'mean_std': Classic thr = mean(MSE_val) + factor * std(MSE_val).
    - 'iqr': Robust non-parametric thr = Q3 + factor * IQR.
    """
    val_errors = np.array(val_errors)
    mean_err = float(np.mean(val_errors))
    std_err = float(np.std(val_errors))

    if method == "percentile":
        threshold = float(np.percentile(val_errors, percentile))
    elif method == "iqr":
        q25, q75 = np.percentile(val_errors, [25, 75])
        iqr = q75 - q25
        threshold = float(q75 + factor * iqr)
    else:  # 'mean_std'
        threshold = float(mean_err + factor * std_err)

    return threshold, mean_err, std_err


def evaluate_anomaly_detection(model, test_X, test_y, threshold, device="cpu"):
    """
    Evaluates Autoencoder on test set (Benign + Malicious flows):
    - Computes intrusion scores (reconstruction MSE)
    - Applies threshold: Anomaly (1) if score > threshold, else Normal (0)
    - Calculates Accuracy, F1-Score, Detection Rate, Precision, FDR, FAR, and AUC-ROC
    """
    scores = np.array(model.compute_reconstruction_errors(test_X, device=device))
    y_pred = (scores > threshold).astype(int)

    acc = accuracy_score(test_y, y_pred) * 100.0
    prec = precision_score(test_y, y_pred, zero_division=0) * 100.0
    rec = recall_score(test_y, y_pred, zero_division=0) * 100.0
    f1 = f1_score(test_y, y_pred, zero_division=0) * 100.0
    auc = roc_auc_score(test_y, scores)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(test_y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # False Discovery Rate: FDR = FP / (FP + TP)
    fdr = (fp / (fp + tp) * 100.0) if (fp + tp) > 0 else 0.0
    # False Alarm Rate / False Positive Rate: FAR = FP / (FP + TN)
    far = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0

    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "detection_rate": rec,
        "precision": prec,
        "fdr": fdr,
        "far": far,
        "auc_roc": auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "cm": cm,
        "scores": scores,
        "threshold": threshold
    }

    return metrics


def plot_research_results(fedavg_history, fedprox_history, fedavg_metrics, fedprox_metrics, test_y, output_dir="results"):
    """
    Generates and saves publication-quality research figures:
    1. Federated Convergence Comparison (FedAvg vs FedProx)
    2. Reconstruction Error Distribution (Benign vs Attack flows with threshold line)
    3. ROC Curves
    4. Confusion Matrices (FedAvg vs FedProx)
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 11})

    # 1. Convergence Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    if fedavg_history:
        ax.plot(fedavg_history["round"], fedavg_history["val_loss"], marker="o", linewidth=2.5, color="#2980b9", label="FedAvg (Val MSE)")
    if fedprox_history:
        ax.plot(fedprox_history["round"], fedprox_history["val_loss"], marker="s", linewidth=2.5, color="#27ae60", label="FedProx (Val MSE)")
    ax.set_xlabel("Federated Communication Round", fontweight="bold")
    ax.set_ylabel("Reconstruction Error (Val MSE)", fontweight="bold")
    ax.set_title("Federated Convergence: FedAvg vs FedProx (10 Clients)", fontweight="bold", pad=12)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "federated_convergence.png"), dpi=300)
    plt.close(fig)

    # 2. Reconstruction Error Distribution (FedAvg and FedProx comparison)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, m, title in zip(axes, [fedavg_metrics, fedprox_metrics], ["FedAvg Reconstruction Error", "FedProx Reconstruction Error"]):
        if "scores" in m:
            scores = m["scores"]
            thr = m["threshold"]
            benign_scores = scores[test_y == 0]
            attack_scores = scores[test_y == 1]
            clip_val = float(np.percentile(attack_scores, 98))
            ax.hist(np.clip(benign_scores, 0, clip_val), bins=60, alpha=0.6, label="Benign Flows (Normal)", color="#2ecc71", density=True)
            ax.hist(np.clip(attack_scores, 0, clip_val), bins=60, alpha=0.6, label="Attack Flows (PortScan)", color="#e74c3c", density=True)
            ax.axvline(thr, color="black", linestyle="--", linewidth=2.5, label=f"Threshold ({thr:.6f})")
            ax.set_xlabel("Intrusion Score (Reconstruction MSE)", fontweight="bold")
            ax.set_ylabel("Probability Density", fontweight="bold")
            ax.set_title(title, fontweight="bold", pad=12)
            ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "error_distribution.png"), dpi=300)
    plt.close(fig)

    # 3. ROC Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    if "scores" in fedavg_metrics:
        fpr_avg, tpr_avg, _ = roc_curve(test_y, fedavg_metrics["scores"])
        ax.plot(fpr_avg, tpr_avg, label=f"FedAvg (AUC = {fedavg_metrics['auc_roc']:.4f})", linewidth=2.5, color="#2980b9")
    if "scores" in fedprox_metrics:
        fpr_prox, tpr_prox, _ = roc_curve(test_y, fedprox_metrics["scores"])
        ax.plot(fpr_prox, tpr_prox, label=f"FedProx (AUC = {fedprox_metrics['auc_roc']:.4f})", linewidth=2.5, color="#27ae60", linestyle="--")
    ax.plot([0, 1], [0, 1], "k:", label="Random Guess (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate (FPR)", fontweight="bold")
    ax.set_ylabel("True Positive Rate / Detection Rate (TPR)", fontweight="bold")
    ax.set_title("ROC Curves for Federated Intrusion Detection", fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300)
    plt.close(fig)

    # 4. Confusion Matrices (FedAvg vs FedProx)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, m, title in zip(axes, [fedavg_metrics, fedprox_metrics], ["FedAvg Confusion Matrix", "FedProx Confusion Matrix"]):
        sns.heatmap(m["cm"], annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_ylabel("True Label", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=300)
    plt.close(fig)

    print(f"[+] All research visualization figures saved to '{output_dir}/'.")
