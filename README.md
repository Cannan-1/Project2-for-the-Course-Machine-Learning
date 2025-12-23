# Superconductor Critical Temperature Prediction via Physics-Informed Residual Networks

**Course Project 2 | Machine Learning (Fall 2025)**
**Department of Artificial Intelligence, Westlake University**

## 📖 Introduction
This project aims to predict the **Critical Temperature ($T_c$)** of superconducting materials using a Physics-Informed Machine Learning (PIML) pipeline.

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
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
├── src/                       # Source modules
└── figures/                   # Plots
🛠️ InstallationBashpip install -r requirements.txt
🧠 Methodology DetailsPhysics-Constrained LossWe introduced a custom loss function:$$\mathcal{L}_{total} = \mathcal{L}_{MSE} + \lambda_1 \mathcal{L}_{non\_neg} + \lambda_2 \mathcal{L}_{upper}$$👨‍💻 AuthorName: Hongyi LIUInstitution: Westlake University
