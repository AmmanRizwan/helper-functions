# Helper Functions for Data Analysis and Machine Learning

A comprehensive Python utility library providing helper functions for data analysis, visualization, preprocessing, feature engineering, and machine learning tasks. This module is specifically designed for handling both numerical and categorical data, outlier detection, data distribution analysis, and neural network modeling.

## Features

- **Data Visualization**: 
  - Box plots for numerical outlier detection
  - Count plots for categorical data distribution
  - Histograms with KDE for numerical distributions
  - Correlation heatmaps and confusion matrices

- **Data Preprocessing**:
  - Outlier handling using IQR method
  - Skewness correction with Box-Cox transformation
  - Data scaling with StandardScaler
  - Dataset rebalancing with oversampling techniques

- **Feature Engineering**:
  - One-Hot Encoding for categorical features
  - Label Encoding for ordinal features
  - Mutual Information scoring for feature selection
  - Correlation analysis

- **Machine Learning**:
  - Neural network model templates
  - Dataset rebalancing with SMOTE
  - Model evaluation with confusion matrices

## Installation

### Prerequisites

Make sure you have Python 3.7+ installed. This module requires the following dependencies:

```bash
pip install pandas matplotlib seaborn scipy scikit-learn imbalanced-learn tensorflow
```

## Quick Start

```python
from helper-1 import *
import pandas as pd

# Load data
data = pd.read_csv('your_data.csv')

# Visualize and analyze
visualize_numerical_box_plots(data)
calculate_outliers_percentage(data)

# Preprocess data
cleaned_data = handle_outliers(data)
transformed_data, _ = handle_skewness(cleaned_data)

# Feature engineering
encoded_data = feature_one_hot_encoder(transformed_data, 'category_col')
```

## Function Reference

### Visualization Functions
- `visualize_numerical_box_plots(df)` - Box plots for outlier detection
- `visualize_categorical_distributions(df)` - Count plots for categorical data
- `visualize_numerical_distributions(df)` - Histograms with KDE for numerical data
- `correlation_feature_matrix(df)` - Correlation matrix with target

### Preprocessing Functions
- `handle_outliers(df, columns=None, all_columns=True)` - IQR-based outlier capping
- `handle_skewness(df, columns=None, threshold=1.0, all_columns=True)` - Box-Cox transformation
- `calculate_outliers_percentage(df)` - Calculate outlier percentages
- `scale_dataset(dataframe, OverSample=True)` - StandardScaler with optional oversampling

### Feature Engineering Functions
- `feature_one_hot_encoder(df, column, drop_columns=True)` - One-hot encoding
- `feature_label_encoding(df, encode_column)` - Label encoding
- `feature_rebalanced(X, y)` - SMOTE rebalancing

### Feature Selection Functions
- `numerical_mi_score(X, y)` - Mutual information for numerical features
- `categorial_mi_score(X, y)` - Mutual information for categorical features

## Neural Network Template

Includes a basic TensorFlow neural network model for binary classification with 64→32→1 architecture.

## Key Algorithms

- **Outlier Detection**: IQR method (Q1 - 1.5×IQR, Q3 + 1.5×IQR)
- **Skewness Correction**: Box-Cox transformation with automatic positive shift
- **Rebalancing**: RandomOverSampler and SMOTE techniques
- **Feature Selection**: Mutual Information scoring
- **Encoding**: One-Hot for nominal, Label for ordinal features

## Requirements

- Python 3.7+
- pandas, matplotlib, seaborn, scipy, scikit-learn, imbalanced-learn, tensorflow
- Functions work with pandas DataFrames
- Target variable should be in the last column for `scale_dataset()`

## Typical ML Workflow

1. **Explore**: Use visualization functions to understand data distribution and outliers
2. **Preprocess**: Handle outliers and skewness with preprocessing functions  
3. **Engineer**: Apply encoding and feature selection using MI scores
4. **Balance**: Use rebalancing functions for imbalanced datasets
5. **Scale**: Apply scaling before training (except tree-based models)
6. **Model**: Train using provided neural network template or other algorithms
7. **Evaluate**: Use confusion matrix template for classification results

## Proper Step to do the ML Prediction

### Import

- Import the essential Libraries.
- Import the Dataset.

### Summary

- Information of the Dataset.
- Shape of the Dataset (Row, Column).
- Check the Duplicated Values.
- Check the Null Values.
- Describe the Dataset.
- Check the Correlation between the Features and Label.
- Check the Mutual Information of the Features.

### Visualize

- Plot the Numerical Features of the Features (Box Plot)
- Plot the Numerical Distribution of the Features (Skewness)
- Plot the Categorical Distribution of the Features (Count Plots)
- Correlation Distribution of the Features (Heatmap)
- Describe Distribution of the Features (Heatmap)
- Describe the Scatter Plot of the Features (Regression Problem)

### Feature Engineering

- Handle the misserable entries if exists
- Handle the Missing Value if exists
- Handle the Duplicate if exists
- Handle Skewness if exists
- Handle Outliers if exists
- Handle Qualitative Feature
  - One Hot Encoding if no natural order exists
  - Label Encoding if natural order exists

### Prediction

- Split the Dataset into Train and Test
- Over Sample the Dataset if imbalance data exists
- Preprocessing the Dataset into standard scaler if not using the (Random Forest, Tree Classifier, XGBoost)
- Predict the Test Set
- Save the Model

## Contributing

Feel free to contribute to this project by:

1. Forking the repository
2. Creating a feature branch
3. Making your changes
4. Submitting a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Amman Rizwan**

Created: September 22, 2025

---

*This module is designed for educational and research purposes in data analysis and machine learning preprocessing tasks.*