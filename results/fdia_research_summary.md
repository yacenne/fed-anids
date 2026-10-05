# Fed-FDIAD: Privacy-Preserving Federated Learning for Stealthy False Data Injection Attack Detection in Smart Grids

**Power System Benchmark:** IEEE 14-Bus Transmission System (14 Buses, 20 Lines)  
**Measurement Space:** 34 SCADA Telemetry Points (14 Bus Injections + 20 Branch Power Flows)  
**Decentralized Topology:** 10 Localized Substation / Control Center Nodes  
**Partitioning:** Heterogeneous (Non-IID Dirichlet $\alpha=0.5$)  
**Optimization:** FedAvg vs FedProx ($\mu=0.01$) vs Centralized Baseline vs Classical BDD  

## Performance Benchmark Table

| Metric | Classical BDD (State Estimation) | Centralized Baseline | FedAvg (10 Clients) | FedProx (10 Clients) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy (%)** | 20.42% | 96.58% | 96.85% | **96.94%** |
| **F1-Score (%)** | 10.51% | 97.99% | 98.15% | **98.20%** |
| **Detection Rate / Recall (%)** | 5.61% | 100.00% | 100.00% | **100.00%** |
| **Precision (%)** | 83.48% | 96.06% | 96.36% | **96.46%** |
| **Miss Rate (FNR) (%)** | 94.39% | 0.00% | 0.00% | **0.00%** |
| **False Alarm Rate (FAR) (%)** | 5.55% | 20.50% | 18.90% | **18.35%** |
| **AUC-ROC** | 0.5000 | **1.0000** | 1.0000 | 1.0000 |

## Key Electrical Engineering Research Findings
1. **Classical Bad Data Detection (BDD) Complete Failure**: Against mathematically stealthy FDI attacks ($a = H c$), the measurement residual satisfies $\|z_a - H \hat{x}_a\| = \|z - H \hat{x}\|$, rendering classical $\chi^2$ and normalized residual tests completely blind (Detection Rate: 5.61%, Miss Rate: 94.39%).
2. **Federated Autoencoder Resiliency**: While BDD is blind to linear algebraic residual invariance, the Deep Autoencoder learns the non-linear physical correlation manifold of actual grid power flows. When false data is injected, reconstruction error spikes dramatically, allowing FedProx to achieve a **100.00% Detection Rate**.
3. **Data Confidentiality for Interconnected Utilities**: Neighboring utilities and substations collaboratively train a robust defense system without ever exchanging proprietary operational telemetry or consumer load curves, complying with NERC-CIP cyber-physical security standards.
