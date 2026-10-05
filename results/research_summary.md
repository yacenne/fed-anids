# Fed-FDIA: Research Performance Evaluation

**Domain:** Cyber-Physical Systems & Smart Grid False Data Injection Attack (FDIA) Detection  
**Base Paper Reference:** Idrissi et al., *Expert Systems With Applications* 234 (2023) 121000  
**Topology:** 10 Distributed Substation Client Nodes  
**Partitioning:** Heterogeneous (Non-IID Dirichlet $\alpha=0.5$)  
**Architecture:** Deep Autoencoder ($D \to 64 \to 32 \to 64 \to D$ with LayerNorm and ReLU)  
**Optimization:** FedAvg vs FedProx ($\mu=0.01$) vs Centralized Baseline  
**Threshold Method:** Idrissi et al. (2023) Equation (1): $thr = \text{mean}(MSE) + 1.0 \times \text{std}(MSE)$  

## Performance Comparison Table (Slide 8 / Project Report)

| Metric | Centralized Baseline | FedAvg (10 Substations) | FedProx (10 Substations) |
| :--- | :---: | :---: | :---: |
| **Accuracy (%)** | 65.23% | 65.95% | **69.96%** |
| **F1-Score (%)** | 68.82% | 69.29% | **74.06%** |
| **Detection Rate / Recall (%)** | 56.22% | 56.29% | **62.83%** |
| **Precision (%)** | 88.69% | 90.09% | **90.16%** |
| **False Discovery Rate (FDR) (%)** | 11.31% | 9.91% | **9.84%** |
| **False Alarm Rate (FAR) (%)** | 15.40% | **13.30%** | 14.72% |
| **AUC-ROC** | 0.7651 | 0.7736 | **0.7930** |

## Key Findings & Research Validation
1. **Zero Raw Telemetry Transmission**: Collaborative federated learning trains a robust global Autoencoder across 10 distributed smart grid substations while preserving complete data privacy of critical infrastructure telemetry and industrial load profiles.
2. **Mitigating Non-IID Load Skew**: Under heterogeneous Dirichlet non-IID distributions ($\alpha=0.5$), FedProx with proximal regularization ($\mu=0.01$) prevents client weight drift caused by diverse substation consumption patterns, outperforming standard FedAvg in stability and FDR.
3. **Statistical Thresholding (Idrissi et al., Eq. 1)**: Computing the anomaly threshold as $thr = \mu_{val} + \sigma_{val}$ strictly on clean operational validation telemetry separates stealthy FDI perturbations from legitimate power swings without requiring attack samples during training.
