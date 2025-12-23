# Superconductor Critical Temperature Prediction via Physics-Informed Residual Networks

**Course Project 2 | Machine Learning (Fall 2025)** **Department of Artificial Intelligence, Westlake University**

## 📖 Introduction

This project aims to predict the **Critical Temperature ($T_c$)** of superconducting materials based on their chemical formulas. 

Unlike traditional "black-box" approaches, this framework integrates **Domain Knowledge** from materials science with Deep Learning. We implement a **Physics-Informed Machine Learning (PIML)** pipeline that combines:
1. **Physics-Based Feature Engineering**: Extracting thermodynamic and electronic properties (e.g., Debye temperature proxy, lattice stiffness) via `pymatgen`.
2. **Physics-Constrained Loss**: Enforcing physical validity (Non-negativity, Upper bounds) during training.
3. **ResNet Architecture**: Using a deep residual network with adaptive learning rates.

## 🚀 Key Achievements & Scientific Insights

| Metric | Baseline (MLP) | **Ours (PIML ResNet)** |
| :--- | :--- | :--- |
| **Feature Dim** | 9 (Basic Atomic Props) | **25+ (Physics-Informed)** |
| **Architecture** | Simple Feed-Forward | **Deep Residual Network** |
| **Loss Function** | MSE Only | **MSE + Physical Constraints ($T_c \ge 0$)** |
| **Convergence** | Slow (~1000 epochs) | **Fast (~300 epochs)** |
| **$R^2$ Score** | ~0.86 | **~0.88** |

### 💡 Key Insight: The Power of Constraints
While the final accuracy ($R^2 \approx 0.88$) is constrained by data quality, the **Physics-Constrained Loss** significantly accelerated training. By penalizing physically impossible predictions (e.g., negative temperatures), we pruned the search space, achieving optimal convergence **3x faster** than the baseline.

## 🧠 Model Architecture

The following diagram illustrates our **Physics-Informed Residual Network**.

```mermaid
graph LR
    subgraph Data Processing
    A[Chemical Formula] --> B(Pymatgen Parser)
    B --> C[Physics Features]
    C -->|Normalization| D[Input Vector 25+ dim]
    end

    subgraph Neural Network ResNet
    D --> E[Input Embedding]
    E --> F{Residual Block 1}
    F --> G{Residual Block 2}
    G --> H{Residual Block 3}
    H --> I[Output Layer]
    end

    subgraph Physics-Informed Loss
    I --> J[Predicted Tc]
    J --> K((Total Loss))
    K --> L[MSE Loss]
    K --> M[Constraint: Tc >= 0]
    K --> N[Constraint: Tc < 350K]
    end
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style J fill:#bfb,stroke:#333,stroke-width:2px
    style K fill:#fbb,stroke:#333,stroke-width:4px
📂 Project StructurePlaintextPROJECT_SUPERCONDUCTOR/
├── main.py                    # [ENTRY POINT] Master pipeline
├── requirements.txt           # Dependencies
├── src/                       # Source code modules
│   ├── data_processor.py      # Feature engineering
│   ├── model.py               # PyTorch architectures
│   └── ...
└── figures/                   # Generated plots
🛠️ Installation & Environment SetupEnsure you have Python 3.8+ installed. It is recommended to use a virtual environment.Bashpip install -r requirements.txt
Key dependencies: torch, pymatgen, pandas, numpy, scikit-learn, matplotlib, seaborn.🏃 Usage (Reproducibility)We provide a one-click script main.py that runs the entire scientific pipeline.Bashpython main.py
🧠 Methodology DetailsPhysics-Constrained LossWe introduced a custom loss function:$$ \mathcal{L}{total} = \mathcal{L}{MSE} + \lambda_1 \mathcal{L}{non_neg} + \lambda_2 \mathcal{L}{upper} $$Where $\mathcal{L}_{non\_neg} = \text{ReLU}(-y_{pred})^2$. This effectively prevents the model from predicting non-physical negative temperatures.👨‍💻 AuthorName: Hongyi LIUInstitution: Westlake University
