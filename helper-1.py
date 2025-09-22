"""
Helper Functions for Data Analysis and Visualization

This module provides a collection of utility functions for data analysis, particularly focused on
handling and visualizing outliers, analyzing distributions, and data preprocessing.
The module includes functions for:

- Visualizing outliers using box plots
- Handling outliers using the IQR method
- Calculating outlier percentages
- Visualizing numerical distributions
- Handling data skewness

Author: Amman Rizwan

Created: 22 Sep, 2025

Dependencies:
  - pandas
  - matplotlib
  - seaborn
  - scipy

Usage:
  Import the required functions to perform various data analysis tasks:
  from helper-1 import visualize_box_plots, handle_outliers, calculate_outliers_percentage

Note:
  This module is designed to work with pandas DataFrames containing numerical data.
  Some functions assume the presence of numerical columns in the input DataFrame.

"""

# Visualize the Outliers in Box Plots

def visualize_box_plots(df):
    """
    Visualizes the distribution of numerical features in the DataFrame using box plots to identify outliers.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the numerical features to visualize.
    
    Returns:
    - None: Displays the box plots.
    """
    # Identify numerical columns
    numerical_columns = df.select_dtypes(include=['number']).columns

    # Set up the figure for multiple subplots
    num_cols = 3  # Number of columns for the subplot grid
    num_rows = (len(numerical_columns) + num_cols - 1) // num_cols  # Calculate number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows))
    fig.suptitle('Box Plot of Numerical Features', fontsize=16)

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    # Iterate over each numerical column and create a box plot
    for i, col in enumerate(numerical_columns):
        sns.boxplot(x=df[col], ax=axes[i], color="skyblue")
        axes[i].set_title(f'Box Plot of {col}', fontsize=14)
        axes[i].set_xlabel(col, fontsize=12)

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the main title space
    plt.show()

# Handle Outliers in Dataset

def handle_outliers(df):
    """
    Handles outliers in a DataFrame by capping based on the IQR method.

    Parameters:
    - df (pd.DataFrame): DataFrame to process.

    Returns:
    - pd.DataFrame: DataFrame with outliers handled.
    """
    for column in df.select_dtypes(include=['number']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Capping
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    return df
  
# Calculate the Outlier as Percentage

def calculate_outliers_percentage(df):
    outlier_counts = {}
    
    for column in df.select_dtypes(include=['number']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Calculate outliers
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outlier_counts[column] = len(outliers)
    
    # Print the percentage of outliers for each column
    for column in outlier_counts:
        percentage = (outlier_counts[column] / len(df)) * 100
        print(f"Percentage of Outliers in {column}: {percentage:.2f}%")
        
# Visualize Numerical Discribution 

def visualize_numerical_distributions(df):
    # Identify numerical columns
    numerical_columns = df.select_dtypes(include=['number']).columns
    
    # Setup the figure for multiple subplots
    num_cols = 5
    num_rows = (len(numerical_columns) + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows))
    fig.suptitle("Distribution of Numerical Features", fontsize=16)
    
    # Flatten axes array for easy iteration
    axes = axes.flatten()
    
    # Iterate over each numerical column and create a histogram with KDE
    for i, col in enumerate(numerical_columns):
        sns.histplot(df[col], kde=True, ax=axes[i], palette="viridis", element='step', stat='density')
        axes[i].set_title(f'Distribution of {col}', fontsize=14)
        axes[i].set_xlabel(col, fontsize=12)
        axes[i].set_ylabel('Density', fontsize=12)
        
    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
# Handle the Skewness in Dataset

def visualize_numerical_distributions(df):
    # Identify numerical columns
    numerical_columns = df.select_dtypes(include=['number']).columns
    
    # Setup the figure for multiple subplots
    num_cols = 5
    num_rows = (len(numerical_columns) + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows))
    fig.suptitle("Distribution of Numerical Features", fontsize=16)
    
    # Flatten axes array for easy iteration
    axes = axes.flatten()
    
    # Iterate over each numerical column and create a histogram with KDE
    for i, col in enumerate(numerical_columns):
        sns.histplot(df[col], kde=True, ax=axes[i], palette="viridis", element='step', stat='density')
        axes[i].set_title(f'Distribution of {col}', fontsize=14)
        axes[i].set_xlabel(col, fontsize=12)
        axes[i].set_ylabel('Density', fontsize=12)
        
    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    
# Scale The Dataset

def scale_dataset(dataframe, OverSample=True):
  X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values
  
  scaler = StanderScaler()
  X = scaler.fit_transform(X)
  
  if OverSample:
    ros = RandomOverSample()
    X, y = ros.fit_resample(X, y)
    
  data = np.hstack((X, np.reshape(y, (-1, 1))))
  
  return data, X, y
