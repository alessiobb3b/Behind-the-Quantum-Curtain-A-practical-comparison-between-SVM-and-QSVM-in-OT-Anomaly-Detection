# Behind the Quantum Curtain: A Practical Comparison Between SVM and QSVM in OT Anomaly Detection

This repository contains the source code used in the paper:

**"Behind the Quantum Curtain: A Practical Comparison Between SVM and QSVM in OT Anomaly Detection"**
<p align="justify">
*Alessio Di Santo, Nicola Camarda, Walter Tiberti, Dajana Cassioli*

Presented in the context of applying both classical and quantum machine learning techniques for detecting anomalies in Operational Technology (OT) environments, this work rigorously compares traditional Support Vector Machines (SVM) with Quantum Support Vector Machines (QSVM), providing real-world insights into their current capabilities and limitations.
</p>
---

## 🧪 Research Context
<p align="justify">
This repository directly supports the experiments and results presented in the paper. The codebase enables reproducibility of:

- Preprocessing and cleaning of OT network traffic data.
- Feature engineering and selection using Random Forests.
- Implementation and hyperparameter tuning of classical SVM with RBF kernel.
- Implementation of QSVM using Qiskit, with reduced dimensionality and dataset due to simulation constraints.
- Evaluation and visualization: confusion matrix, ROC and precision-recall curves, learning curve.

The project demonstrates that **classical SVMs currently outperform QSVMs** for large-scale OT anomaly detection due to hardware limitations, while **QSVMs exhibit promising results** on reduced data and hint at future potential.
</p>
---

## 📁 Repository Structure
- `main.py`: Coordinates the overall pipeline — preprocessing, feature selection, training and evaluation of both SVM and QSVM.
- `preprocessing_function.py`: Handles data cleaning, normalization using `RobustScaler`, label encoding, and one-hot encoding for ports/protocols.
- `feature_selection_function.py`: Implements Random Forest feature selection, retaining features with importance above the median.
- `svm_function.py`: Defines classical SVM with RBF kernel and performs Grid Search hyperparameter tuning.
- `qsvm_function.py`: Implements a QSVM using Qiskit's `ZZFeatureMap` and `FidelityQuantumKernel`. Configured for 4 qubits due to simulation constraints.
---

## 🚀 How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/alessiobb3b/Behind-the-Quantum-Curtain-A-practical-comparison-between-SVM-and-QSVM-in-OT-Anomaly-Detection.git
cd Behind-the-Quantum-Curtain-A-practical-comparison-between-SVM-and-QSVM-in-OT-Anomaly-Detection
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the main script:**

```bash
python main.py
```

The script will execute both pipelines and output classification reports and performance metrics.

---

## 📈 Summary of Findings

| Model        | Dataset Size      | Accuracy | F1-Score | Notes |
|--------------|-------------------|----------|----------|-------|
| SVM (full)   | 360,964 samples   | 98%      | 0.98     | Best overall performer |
| QSVM         | 5000 train, 2500 test | 89%  | 0.89     | Quantum simulation constraint |
| SVM (same subset) | 5000 train, 2500 test | 78% | 0.77 | Lower than QSVM in this subset |

**Takeaway**: QSVM outperforms classical SVM on reduced data, but classical SVM dominates in real-world scale due to computational feasibility.

---

## 📄 Citation

If this work was helpful, please cite the original paper:

```
Waiting for Acceptance!!!
```

---

## 📬 Contact

- **Alessio Di Santo**  <br />
  Department of Information Engineering, University of L'Aquila  <br />
  [GitHub](https://github.com/alessiobb3b)
- **Nicola Camarda** <br />
  Independet Reseracher <br />
  [GitHub](https://github.com/camardanic)

---

## 📘 Reserach Funds

This research was funded under the ISP5G+ project (CUP D33C22001300002), part of the SERICS program (PE00000014) from the NRRP MUR, funded by EU-NGEU.
