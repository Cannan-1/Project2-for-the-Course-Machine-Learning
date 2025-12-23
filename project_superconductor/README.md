# Superconductor Critical Temperature Prediction via Physics-Informed Residual Networks

**Course Project 2 | Machine Learning (Fall 2025)**
**Department of Artificial Intelligence, Westlake University**


## 📖 Introduction

This project aims to predict the **Critical Temperature ($T_c$)** of superconducting materials based on their chemical formulas. 

Unlike traditional "black-box" approaches, this framework integrates **Domain Knowledge** from materials science with Deep Learning. We implement a **Physics-Informed Machine Learning (PIML)** pipeline that combines:
1.  **Physics-Based Feature Engineering**: Extracting thermodynamic and electronic properties (e.g., Debye temperature proxy, lattice stiffness) via `pymatgen`.
2.  **Physics-Constrained Loss**: Enforcing physical validity (Non-negativity, Upper bounds) during training.
3.  **ResNet Architecture**: Using a deep residual network with adaptive learning rates.

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

The following diagram illustrates our **Physics-Informed Residual Network**. It processes chemical formulas into physical features, passes them through residual blocks, and is optimized using a physics-constrained loss function.

```mermaid
graph LR
    subgraph Data Processing
    A[Chemical Formula] --> B(Pymatgen Parser)
    B --> C[Physics Features]
    C -->|Normalization| D[Input Vector (25+ dim)]
    end

    subgraph Neural Network (ResNet)
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



## 📂 Project Structure

```text
PROJECT_SUPERCONDUCTOR/
├── main.py                    # [ENTRY POINT] Master pipeline for EDA, Training, and Inference
├── requirements.txt           # Dependencies
├── best_model.pth             # Saved weights of the best performing model
├── submission.csv             # Final predictions for the test set
├── evaluation_report.txt      # Detailed metrics (RMSE, MAE, R2)
├── error_analysis_worst_cases.csv # Top 10 worst predictions for analysis
├── train.tsv / test.tsv       # Dataset files
├── src/                       # Source code modules
│   ├── data_processor.py      # Feature engineering & Formula parsing (pymatgen)
│   ├── model.py               # PyTorch architectures (TcPredictor & TcPredictorAdvanced)
│   ├── train_tc_prediction.py # Training loop with Physics Loss & Cross-Validation
│   ├── evaluation.py          # Metrics calculation & Error Analysis
│   ├── inference_tc.py        # Single-instance inference engine
│   └── eda.py                 # Exploratory Data Analysis script
└── figures/                   # Generated plots
    ├── training_curve_physics.pdf # Loss & R2 curves (PIML vs Baseline)
    ├── supercon_predictions.pdf   # Predicted vs True Tc scatter plot
    ├── eda_feature_correlation.pdf# Feature correlation heatmap
    └── eda_tc_distribution.pdf    # Target variable distribution



🛠️ InstallationEnvironment Setup:Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.Bashpip install -r requirements.txt
Key dependencies: torch, pymatgen, pandas, numpy, scikit-learn, matplotlib, seaborn.🏃 Usage (Reproducibility)We provide a one-click script main.py that runs the entire scientific pipeline.1. Run the Full PipelineBashpython main.py
What happens inside:[Step 0] EDA: Generates distribution and correlation plots in figures/.[Step 1] Training:Extracts physics-informed features.Runs 5-Fold Cross-Validation.Trains the ResNet with Physics-Constrained Loss.Note: Forced to CPU mode for maximum compatibility.

[Step 2] Error Analysis: Identifies the top 10 worst predictions and saves them to error_analysis_worst_cases.csv.

[Step 3] Inference: Generates submission.csv for the test set.2. Standalone Inference (Demo)To test specific materials (e.g., MgB2, YBCO):Bashpython src/inference_tc.py
3. Standalone EvaluationTo generate metrics report:Bashpython src/evaluation.py



🧠 Methodology DetailsFeature EngineeringWe moved beyond basic atomic numbers. Using pymatgen, we constructed a feature vector including:Thermodynamic: Melting point, Enthalpy of fusion (proxy for lattice stiffness).Electronic: Electronegativity variance, Thermal conductivity.Structural: Atomic radius, row/group trends.Derived: Debye temperature proxy ($\sqrt{T_m / M}$).Physics-Constrained LossWe introduced a custom loss function:$$ \mathcal{L}{total} = \mathcal{L}{MSE} + \lambda_1 \mathcal{L}{non_neg} + \lambda_2 \mathcal{L}{upper} $$Where $\mathcal{L}_{non\_neg} = \text{ReLU}(-y_{pred})^2$. This effectively prevents the model from predicting non-physical negative temperatures.

⚖️ Ethical Risk AssessmentDual-Use: Accelerated discovery of superconductors has potential military applications (e.g., advanced weaponry components).Environmental Impact: The model relies on chemical composition and might bias towards materials containing toxic (e.g., Hg, Tl) or rare-earth elements, indirectly encouraging environmentally damaging mining practices. Future iterations should incorporate "sustainability scores" as constraints.


👨‍💻 AuthorName: Hongyi LIU
Institution: Westlake University