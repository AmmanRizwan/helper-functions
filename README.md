# Helper Functions for Data Analysis

A Python utility library providing helper functions for data analysis, visualization, and preprocessing tasks. This module is specifically designed for handling numerical data, outlier detection, and data distribution analysis.

## Features

- **Outlier Visualization**: Create box plots to visualize outliers in numerical data
- **Outlier Handling**: Remove or cap outliers using the IQR (Interquartile Range) method
- **Outlier Analysis**: Calculate the percentage of outliers in your dataset
- **Data Distribution**: Visualize numerical distributions with histograms and KDE plots
- **Data Scaling**: Scale datasets with optional oversampling for imbalanced data

## Installation

### Prerequisites

Make sure you have Python 3.6+ installed. This module requires the following dependencies:

```bash
pip install pandas matplotlib seaborn scipy scikit-learn imbalanced-learn
```

### Usage

Import the functions you need from the helper module:

```python
from helper-1 import (
    visualize_box_plots,
    handle_outliers,
    calculate_outliers_percentage,
    visualize_numerical_distributions,
    scale_dataset
)
```

## Function Documentation

### `visualize_box_plots(df)`

Visualizes the distribution of numerical features using box plots to identify outliers.

**Parameters:**
- `df` (pd.DataFrame): The DataFrame containing numerical features to visualize

**Example:**
```python
import pandas as pd
from helper-1 import visualize_box_plots

# Load your data
data = pd.read_csv('your_data.csv')

# Visualize outliers
visualize_box_plots(data)
```

### `handle_outliers(df)`

Handles outliers in a DataFrame by capping them based on the IQR method.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to process

**Returns:**
- `pd.DataFrame`: DataFrame with outliers handled

**Example:**
```python
# Handle outliers in your dataset
cleaned_data = handle_outliers(data)
```

### `calculate_outliers_percentage(df)`

Calculates and prints the percentage of outliers for each numerical column.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame

**Example:**
```python
# Calculate outlier percentages
calculate_outliers_percentage(data)
```

### `visualize_numerical_distributions(df)`

Creates histograms with KDE plots for all numerical features in the DataFrame.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame

**Example:**
```python
# Visualize data distributions
visualize_numerical_distributions(data)
```

### `scale_dataset(dataframe, OverSample=True)`

Scales the dataset using StandardScaler with optional oversampling for imbalanced datasets.

**Parameters:**
- `dataframe` (pd.DataFrame): Input DataFrame (assumes target is the last column)
- `OverSample` (bool): Whether to apply random oversampling (default: True)

**Returns:**
- `tuple`: (scaled_data, X_scaled, y)

**Example:**
```python
# Scale dataset with oversampling
scaled_data, X, y = scale_dataset(data, OverSample=True)
```

## Data Requirements

- The functions are designed to work with pandas DataFrames
- Numerical columns are automatically detected using `select_dtypes(include=['number'])`
- For `scale_dataset()`, the target variable should be in the last column

## Outlier Detection Method

This library uses the **IQR (Interquartile Range) method** for outlier detection:

- Lower bound: Q1 - 1.5 × IQR
- Upper bound: Q3 + 1.5 × IQR
- Values outside these bounds are considered outliers

## AI/ML Proper Steps

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

### Feature Engineering

- Handle the Missing Value if exists
- Handle the Duplicate if exists
- Handle the misserable entries if exists
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