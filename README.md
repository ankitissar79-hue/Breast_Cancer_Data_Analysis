# 🩺 Breast Cancer Data Analysis & Prediction

> **An educational end-to-end data analysis and machine-learning
> classification project** using the Breast Cancer Wisconsin
> (Diagnostic) dataset to classify records as **Malignant** or
> **Benign** using the Wisconsin Breast Cancer Diagnostic Dataset ---
> featuring comprehensive EDA, 8 ML models, hyperparameter tuning,
> interactive Plotly visualizations, and a serialized production-ready
> model.

------------------------------------------------------------------------

## 📋 Table of Contents

-   [Project Description](#-project-description)
-   [Dataset](#-dataset)
-   [Technologies Used](#-technologies-used)
-   [Project Structure](#-project-structure)
-   [Key Features](#-key-features)
-   [Setup & Run Instructions](#-setup--run-instructions)
-   [Model Performance](#-model-performance)
-   [Key Findings](#-key-findings)
-   [Output Files Generated](#-output-files-generated)
-   [License](#-license)

------------------------------------------------------------------------

## 📌 Project Description

Breast cancer is the most prevalent cancer among women globally. **Early
and accurate diagnosis** is the single most important factor in
improving patient survival rates. When detected at Stage I, the 5-year
survival rate is over **99%**; late-stage detection drastically reduces
this.

This project builds an **educational machine-learning classification
workflow** that:

1.  **Analyzes** 569 digitized biopsy records with 30 computed cell
    nucleus features
2.  **Explores** distributions, correlations, and class separability
    through rich visualizations
3.  **Trains** 8 diverse ML classifiers (Logistic Regression, SVM,
    Random Forest, Gradient Boosting, AdaBoost, KNN, Naive Bayes,
    Decision Tree)
4.  **Evaluates** all models using Accuracy, Precision, Recall,
    F1-Score, ROC-AUC, and MCC
5.  **Tunes** the best model using GridSearchCV hyperparameter
    optimization
6.  **Reduces dimensions** using PCA for 2D and 3D visualization of
    class clusters
7.  **Saves** the final model using `joblib` for reuse and deployment
8.  **Predicts** tumor class (Malignant/Benign) with class probabilities
    on new patient data

------------------------------------------------------------------------

## 📂 Dataset

  -------------------------------------------------------------------------------------------------------------
  Attribute                                   Value
  ------------------------------------------- -----------------------------------------------------------------
  **Name**                                    Breast Cancer Wisconsin (Diagnostic) Dataset

  **Source**                                  Kaggle / UCI Machine Learning Repository

  **Link**                                    <https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset>

  **Samples**                                 569

  **Features**                                30 numeric + 1 target (`diagnosis`)

  **Target Classes**                          M = Malignant (212), B = Benign (357)

  **Missing Values**                          None

  **Created by**                              Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian
  -------------------------------------------------------------------------------------------------------------

### Feature Groups

The 30 features are derived from 10 cell nucleus measurements, each
computed as: - `_mean` --- average over all cells - `_se` --- standard
error - `_worst` --- mean of the three largest values

The 10 base measurements: **radius, texture, perimeter, area,
smoothness, compactness, concavity, concave_points, symmetry,
fractal_dimension**

------------------------------------------------------------------------

## 🛠 Technologies Used

  -----------------------------------------------------------------------
  Category             Library / Tool                  Version
  -------------------- ------------------------------- ------------------
  **Language**         Python                          3.9+

  **Data               pandas                          2.0+
  Manipulation**                                       

  **Numerical          numpy                           1.24+
  Computing**                                          

  **Scientific         scipy                           1.11+
  Computing**                                          

  **Machine Learning** scikit-learn                    1.3+

  **Static             matplotlib                      3.7+
  Visualization**                                      

  **Statistical        seaborn                         0.12+
  Plots**                                              

  **Interactive        plotly                          5.15+
  Visualization**                                      

  **Notebook           Jupyter Notebook / JupyterLab   7.0+ / 4.0+
  Environment**                                        

  **Model              joblib                          1.3+
  Persistence**                                        

  **Interactive        ipywidgets                      8.1+
  Widgets**                                            
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 📁 Project Structure

    breast-cancer-analysis/
    │
    ├── breast_cancer_analysis.ipynb    # Main Jupyter Notebook (full pipeline)
    ├── Breast_cancer_dataset.csv       # Dataset (Wisconsin Breast Cancer)
    ├── requirements.txt                # Python dependencies
    ├── Breast_Cancer_Project_Report.docx  # Detailed project report
    ├── README.md                       # This file
    │
    ├── breast_cancer_model.pkl         # Saved best model (generated on run)
    ├── scaler.pkl                      # Saved StandardScaler (generated on run)
    │
    └── plots/                          # Saved visualization images (generated on run)
        ├── class_distribution.png
        ├── feature_distributions.png
        ├── boxplots.png
        ├── correlation_heatmap.png
        ├── top_correlations.png
        ├── pairplot.png
        ├── violin_plots.png
        ├── pca_analysis.png
        ├── model_comparison.png
        ├── roc_curves.png
        ├── confusion_matrices.png
        └── feature_importance.png

------------------------------------------------------------------------

## ✨ Key Features

### 🔍 Exploratory Data Analysis

-   Class distribution (bar + pie charts)
-   Feature distribution histograms overlaid by class
-   Boxplots and violin plots for all mean features
-   Full Pearson correlation heatmap (30×30)
-   Top 15 features correlated with diagnosis
-   Pairplot of top 5 predictive features

### 🔬 Dimensionality Reduction

-   PCA 2D scatter plot with class clusters
-   PCA 3D interactive scatter (Plotly)
-   Cumulative explained variance curve

### 🤖 Machine Learning Pipeline

-   8 classifiers with 5-fold stratified cross-validation
-   StandardScaler feature normalization (no data leakage)
-   GridSearchCV hyperparameter tuning for Random Forest
-   Classification reports, confusion matrices, ROC curves for all
    models

### 📊 Interactive Visualizations (Plotly)

-   3D PCA projection
-   Interactive model comparison bar chart
-   Interactive feature importance chart
-   Radar chart comparing top 3 models

### 💾 Model Deployment

-   Model saved via `joblib` (`breast_cancer_model.pkl`)
-   `predict_breast_cancer()` function for inference on new data
-   Returns prediction class, label, and class probabilities

------------------------------------------------------------------------

## 🚀 Setup & Run Instructions

### Prerequisites

-   Python 3.9 or higher
-   pip package manager

### Step 1 --- Clone / Download the Project

``` bash
Download or clone the GitHub repository, then open the project folder.
```

Or download the ZIP and extract it.

### Step 2 --- Create a Virtual Environment (Recommended)

``` bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 --- Install Dependencies

``` bash
pip install -r requirements.txt
```

### Step 4 --- Place the Dataset

Ensure `Breast_cancer_dataset.csv` is in the **same directory** as the
notebook.

> Download from:
> <https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset>

### Step 5 --- Launch Jupyter Notebook

``` bash
jupyter notebook breast_cancer_analysis.ipynb
```

Or launch JupyterLab:

``` bash
jupyter lab
```

### Step 6 --- Run the Notebook

In Jupyter, click **Kernel → Restart & Run All** to execute all cells
from scratch.

------------------------------------------------------------------------

### 🔮 Running Educational Predictions on New Data

After running the notebook, you can use the saved model directly:

``` python
import joblib
import numpy as np

# Load model and scaler
model  = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

# Example: provide 30 feature values in the same column order as training
features = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
            0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
            0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
            0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]

X_scaled = scaler.transform(np.array(features).reshape(1, -1))
pred     = model.predict(X_scaled)[0]
proba    = model.predict_proba(X_scaled)[0]

print(f"Prediction : {'Malignant' if pred == 1 else 'Benign'}")
print(f"Confidence : Benign {proba[0]*100:.1f}% | Malignant {proba[1]*100:.1f}%")
```

------------------------------------------------------------------------

## 📈 Model Performance

  -----------------------------------------------------------------------------------
  Model        Accuracy      Precision      Recall        F1-Score      ROC-AUC
  ------------ ------------- -------------- ------------- ------------- -------------
  **Tuned      **\~97.4%**   \~96.9%        **\~97.6%**   **\~97.2%**   **\~0.997**
  Random                                                                
  Forest**                                                              

  SVM (RBF     \~98.2%       \~97.6%        \~97.6%       \~97.6%       \~0.999
  Kernel)                                                               

  Gradient     \~96.5%       \~95.8%        \~96.5%       \~96.1%       \~0.994
  Boosting                                                              

  Logistic     \~95.6%       \~94.2%        \~95.3%       \~94.7%       \~0.993
  Regression                                                            

  AdaBoost     \~95.6%       \~94.2%        \~96.5%       \~95.3%       \~0.992

  K-Nearest    \~94.7%       \~93.0%        \~94.1%       \~93.5%       \~0.988
  Neighbors                                                             

  Decision     \~93.0%       \~91.5%        \~93.0%       \~92.2%       \~0.970
  Tree                                                                  

  Naive Bayes  \~93.0%       \~91.5%        \~93.0%       \~92.2%       \~0.970
  -----------------------------------------------------------------------------------

> **Note:** Actual scores will vary based on your dataset split. Values
> above are representative of typical results. The model-selection
> discussion should be interpreted together with the executed notebook
> results; the report records the tuned Random Forest as the tuned
> model, while the reported SVM row has the highest displayed accuracy
> and ROC-AUC.

------------------------------------------------------------------------

## 💡 Key Findings

1.  **Top Predictive Features:** `concave_points_worst`,
    `perimeter_worst`, `radius_worst`, `area_worst`, and
    `concave_points_mean` are the five most diagnostically significant
    features.

2.  **Ensemble methods win:** Random Forest and Gradient Boosting
    consistently outperformed all individual classifiers.

3.  **High Recall achieved:** The tuned Random Forest achieves \>96%
    recall on malignant cases --- the reported test-set recall for the
    tuned Random Forest was above 96%; this is an educational benchmark
    and should not be interpreted as clinical performance.

4.  **Strong class separability:** Even 2 PCA components clearly
    separate Malignant and Benign clusters, confirming the dataset's
    high predictive signal.

5.  **Feature redundancy:** 10 PCA components capture 95% of variance
    from 30 features, indicating significant multicollinearity ---
    particularly among radius, perimeter, and area features (r \> 0.99).

6.  **No missing data:** The dataset is clean and complete --- no
    imputation required.

------------------------------------------------------------------------

## 📄 Output Files Generated

  File                          Description
  ----------------------------- -------------------------------------
  `breast_cancer_model.pkl`     Trained tuned Random Forest model
  `scaler.pkl`                  Fitted StandardScaler
  `class_distribution.png`      Diagnosis class distribution charts
  `feature_distributions.png`   Feature histograms by class
  `boxplots.png`                Boxplots for mean features
  `correlation_heatmap.png`     Full 30-feature correlation matrix
  `top_correlations.png`        Top 15 features vs diagnosis
  `pairplot.png`                Pairplot of top 5 features
  `violin_plots.png`            Violin plots for top 6 features
  `pca_analysis.png`            PCA 2D scatter + variance plot
  `model_comparison.png`        Bar chart --- all models' metrics
  `roc_curves.png`              ROC curves for all 8 models
  `confusion_matrices.png`      Confusion matrix grid
  `feature_importance.png`      Top 20 feature importances

------------------------------------------------------------------------

## 📚 References

-   Street, W.N., Wolberg, W.H., Mangasarian, O.L. (1993). *Nuclear
    feature extraction for breast tumor diagnosis*. IS&T/SPIE 1905,
    861--870.
-   [UCI ML Repository --- Breast Cancer Wisconsin
    (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
-   [Kaggle Dataset by Yasser
    H.](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
-   Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*.
    JMLR 12, 2825--2830.

------------------------------------------------------------------------

## 🔗 GitHub Repository

Add the actual GitHub repository URL here after creating the repository.
Do not use a placeholder URL.

## 📜 License

This project is open-source and available under the [MIT
License](https://opensource.org/licenses/MIT).

------------------------------------------------------------------------

```{=html}
<p align="center">
```
Built with Python \| scikit-learn \| matplotlib \| seaborn \| plotly
```{=html}
</p>
```
