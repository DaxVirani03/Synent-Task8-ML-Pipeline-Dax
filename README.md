# AutoML & Deep Learning Pipeline

A sophisticated, automated machine learning pipeline built in Jupyter Notebook that handles data exploration, preprocessing, model selection, and clustering for both classification and regression tasks.

## Features

### 1. Automated Data Extraction & EDA
- **Dynamic Loading**: Supports local CSV files or URLs.
- **Visual Exploration**: 
  - distribution histograms for all numeric features.
  - Correlation heatmaps to identify feature relationships.
  - Missing value analysis with horizontal bar charts.
- **Smart Detection**: Automatically detects target columns and determines if the task is **Classification** or **Regression** based on data types and cardinality.

### 2. Market-Leading Model Zoo
The pipeline trains and compares a wide variety of state-of-the-art algorithms:
- **Tree-Based**: Decision Trees, Random Forests, and Extra Trees.
- **Boosting**: Gradient Boosting, AdaBoost, and HistGradientBoosting.
- **Advanced Boosting**: Logic for **XGBoost** and **LightGBM** (handles dependencies gracefully).
- **Metric-Based**: k-Nearest Neighbors (k-NN).
- **Deep Learning (ANN)**: Multi-configuration Artificial Neural Network with automated hyperparameter testing (layer depth, units, dropout).

### 3. Comprehensive Preprocessing
- **Imputation**: Median imputer for numbers; Most Frequent for categories.
- **Scaling**: Standard scaling for all numeric features.
- **Encoding**: One-Hot Encoding for categorical variables.
- **Dimensionality Reduction**: PCA used for clustering visualizations.

### 4. Unsupervised Learning (Clustering)
- **KMeans**: Automated cluster count selection using silhouette scores.
- **Hierarchical**: Agglomerative clustering comparison.
- **Visualization**: 2D PCA projection of clusters for high-dimensional data interpretation.

### 5. Evaluation & Explainability
- **Classification Metrics**: Accuracy, Weighted F1-Score, and Confusion Matrix.
- **Regression Metrics**: MAE, RMSE, and $R^2$ Score.
- **Feature Importance**: Visual ranking of the most influential features for tree-based models.
- **Learning Curves**: Training vs. Validation loss plots for the ANN.

## 🛠️ Setup

1. **Install Dependencies**:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow xgboost lightgbm
   ```

2. **Run the Pipeline**:
   Open `ann_pipeline_clean.ipynb` and set your dataset path:
   ```python
   DATA_SOURCE = "your_data.csv"
   TARGET_COL = "your_target_column"
   ```

## 📊 File Structure
- `ann_pipeline_clean.ipynb`: The main automation notebook.
- `mall_customers.csv`: Default sample dataset for demonstration.
- `superstore.csv`: Alternative large-scale dataset.

---
*Created as part of the Synent ML Automation Suite.*
