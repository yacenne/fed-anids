# Fed-ANIDS: Privacy-Preserving Federated Network Intrusion Detection

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Validated-brightgreen.svg)]()

A research-grade implementation and empirical validation of **Fed-ANIDS (Federated Anomaly-Based Network Intrusion Detection System)**, synthesizing the federated anomaly detection foundations of **Nguyen et al. (IEEE ICDCS 2019)** and **Mothukuri et al. (IEEE IoTJ 2022)** with robust proximal federated optimization (FedProx).

---

## 📌 Project Overview

Traditional Network Intrusion Detection Systems (NIDS) face two major bottlenecks:
1. **Zero-Day Attack Blindness**: Signature-based systems fail against novel or previously unseen threats.
2. **Data Privacy & Regulation Risks**: Network packet captures (PCAPs) and corporate flow logs contain sensitive IP addresses, payload contents, and confidential credentials that cannot be centralized due to regulatory compliance (GDPR, HIPAA).

**Fed-ANIDS** resolves this dilemma by uniting:
- **Unsupervised Deep Autoencoders**: Trained strictly on normal/benign network traffic to learn baseline communication patterns. When an attack occurs, the model fails to reconstruct the anomalous pattern, causing a spike in reconstruction error ($MSE$).
- **Federated Learning (FL)**: Collaboratively trains a shared global Autoencoder across **10 localized client network nodes** without ever moving private raw traffic logs outside local boundaries. Only numerical weight updates are exchanged with the central orchestrator.

---

## 🔬 System Architecture & Methodology

```mermaid
flowchart LR
    subgraph Clients ["10 Collaborating Client Nodes"]
        C1["Node 1 (Private Benign Flows)"]
        C2["Node 2 (Private Benign Flows)"]
        C3["... Client Nodes 3 to 9 ..."]
        C10["Node 10 (Private Benign Flows)"]
    end

    subgraph Server ["Central Orchestrator"]
        Init["Global Autoencoder"]
        Agg["Aggregation (FedAvg / FedProx)"]
        Thresh["Statistical Thresholding (τ)"]
    end

    Init -->|1. Broadcast Global Weights| Clients
    Clients -->|2. Local Training on Normal Flows| Clients
    Clients -->|3. Share Local Weights (Gradients)| Agg
    Agg -->|4. Update Global Model| Thresh
```

### 1. Autoencoder Anomaly Detection
- **Input:** $D$-dimensional flow vector extracted via CICFlowMeter (flow duration, packet lengths, inter-arrival times, TCP flags, etc.).
- **Encoder:** $D \to 64 \to 32$ (Latent bottleneck $z \in \mathbb{R}^{32}$ with LayerNorm and ReLU).
- **Decoder:** $32 \to 64 \to D$ (Reconstructed flow $\hat{x}$).
- **Intrusion Score ($IS$):**
  $$IS(x) = \frac{1}{D} \sum_{j=1}^D (x_j - \hat{x}_j)^2$$

### 2. Statistical Thresholding ($\tau$)
Computed strictly on a held-out benign validation set $D_{val}$:
$$\hat{y} = \begin{cases} 1\ (\text{Malicious / Intrusion Alert}), & \text{if } IS(x) > \tau \\ 0\ (\text{Benign / Normal Traffic}), & \text{if } IS(x) \le \tau \end{cases}$$

### 3. Federated Optimization: FedAvg vs FedProx
- **FedAvg (McMahan et al.):**
  $$w_{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_{t+1}^k$$
- **FedProx (Li et al.):** Adds a proximal penalty $\frac{\mu}{2} \|w - w_t\|^2$ to the client local objective to prevent model drift caused by statistical heterogeneity (Non-IID traffic distributions across corporate departments).

---

## 📊 Empirical Research Results (CIC-IDS2017)

Validated on **40,000 real network flow records** from the **CIC-IDS2017** benchmark dataset across **10 simulated client nodes** with Non-IID Dirichlet traffic skew ($\alpha=0.5$):

| Metric | Centralized Baseline | FedAvg (10 Clients) | FedProx (10 Clients) |
| :--- | :---: | :---: | :---: |
| **Accuracy (%)** | 96.45% | 76.29% | **96.51%** |
| **F1-Score (%)** | 97.97% | 84.68% | **98.00%** |
| **Detection Rate / Recall (%)** | 99.37% | 76.01% | **99.37%** |
| **Precision (%)** | 96.61% | 95.58% | **96.67%** |
| **False Discovery Rate (FDR) (%)** | 3.39% | 4.42% | **3.33%** |
| **False Alarm Rate (FAR / FPR) (%)** | 21.77% | 21.94% | **21.35%** |
| **AUC-ROC** | 0.8752 | **0.8499** | 0.8376 |

### 📈 Generated Visualizations & Artifacts (in `results/`):
- `results/models/`: Saved PyTorch weights (`fedprox_autoencoder.pth`, `fedavg_autoencoder.pth`, `centralized_autoencoder.pth`), preprocessing scaler (`scaler.joblib`), and metadata (`model_metadata.json`).
- `results/metrics.json`: Structured numerical evaluation metrics for all configurations.
- `results/federated_convergence.png`: Training loss curves showing stable convergence across communication rounds.
- `results/error_distribution.png`: Histogram comparison showing clear separation between normal benign flows (low error) and malicious attack flows (high error) for FedAvg and FedProx.
- `results/roc_curve.png`: ROC curves for FedAvg and FedProx.
- `results/confusion_matrices.png`: Confusion matrices detailing True Positives, False Positives, True Negatives, and False Negatives.

---

## 🚀 How to Run & Reproduce Experiments

### 1. Requirements
Ensure Python 3.11+ is installed with the required libraries:
```bash
pip install torch scikit-learn pandas numpy matplotlib seaborn joblib
```

### 2. Run the Full 10-Client Research Validation
```bash
python train_and_validate.py --sample_limit 40000 --rounds 10 --local_epochs 2 --percentile 80.0
```

### 3. Command-Line Options
| Option | Default | Description |
| :--- | :---: | :--- |
| `--clients` | `10` | Number of localized federated organization nodes |
| `--rounds` | `10` | Federated communication rounds |
| `--local_epochs` | `2` | Local training epochs per round on each client |
| `--mu` | `0.01` | FedProx proximal regularizer parameter |
| `--percentile` | `80.0` | Percentile on validation benign errors for threshold $\tau$ |
| `--sample_limit` | `50000` | Dataset records to use (supports full dataset by setting higher) |
| `--non_iid` | `True` | Heterogeneous Dirichlet data distribution ($\alpha=0.5$) |

---

## 📁 Repository Structure
```
g:\My Drive\Projects\FED-NIDS\
├── data/
│   └── raw/
│       └── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv  # CIC-IDS2017 dataset
├── src/
│   ├── autoencoder.py          # PyTorch Deep Autoencoder (68 -> 64 -> 32 -> 64 -> 68)
│   ├── data_loader.py          # Preprocessing, normalization & 10-client Dirichlet partitioning
│   ├── evaluation.py           # Anomaly scoring, statistical thresholding, and plotting
│   └── federated.py            # FederatedClient and FederatedServer (FedAvg & FedProx)
├── results/
│   ├── models/                 # Checkpoints (.pth), scaler (.joblib), metadata.json
│   ├── metrics.json            # Machine-readable evaluation metrics
│   ├── confusion_matrices.png  # Confusion matrices
│   ├── error_distribution.png  # Reconstruction error distributions
│   ├── federated_convergence.png # Convergence loss curves
│   ├── research_summary.md     # Detailed markdown summary
│   └── roc_curve.png           # ROC curves
├── train_and_validate.py       # Master research execution script
└── README.md                   # Project documentation
```

---

## 📚 Key References & Citation

Detailed academic references with full DOI links and BibTeX entries are documented in [REFERENCES.md](REFERENCES.md) and [references.bib](references.bib).

- **Primary Literature Foundations:**
  - T. D. Nguyen et al., *"DÏoT: A Federated Self-learning Anomaly Detection System for IoT,"* **IEEE ICDCS**, 2019. DOI: [10.1109/ICDCS.2019.00080](https://doi.org/10.1109/ICDCS.2019.00080).
  - V. Mothukuri et al., *"Federated-learning-based anomaly detection for IoT security attacks,"* **IEEE Internet of Things Journal**, 2022. DOI: [10.1109/JIOT.2021.3077803](https://doi.org/10.1109/JIOT.2021.3077803).
- **Deep Autoencoder Anomaly Detection:** Y. Meidan et al., *"N-BaIoT,"* **IEEE Pervasive Computing**, 2018.
- **Federated Learning Algorithms:** 
  - FedAvg: B. McMahan et al., *AISTATS*, 2017.
  - FedProx: T. Li et al., *MLSys*, 2020.
- **Dataset:** I. Sharafaldin et al., *"CIC-IDS2017 Dataset,"* *ICISSP*, 2018.

---

## 👥 Authors
- **Faizan Abrar** (23BEE018)
- **Adnan Ahmed Ansari** (23BEE011)
- **Yaseen Umar Shees** (23BEE010)  
*Department of Electrical Engineering, Jamia Millia Islamia*  
*Supervisor: Prof. Ikbal Ali*

