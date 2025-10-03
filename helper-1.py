"""
Focusing Mainly in Supervised Learning Datasets

Helper Functions for Data Analysis and Visualization

This module provides a collection of utility functions for data analysis, particularly focused on
handling and visualizing outliers, analyzing distributions, and data preprocessing.
The module includes functions for:

- Visualizing outliers using box plots

- Visualizing Categorical using count plots

- Handling outliers using the IQR method

- Calculating outlier percentages

- Visualizing numerical distributions

- Handling data skewness

- Mutual Information
    - Categorical Features
    - Numerical Features
    
- Feature Engineering
    - Label Encoding
    - One Hot Encoding
    - Scale Dataset
    
- Nueral Network Model
    - Classification Problem
    - Regression Problem (due)

Author: Amman Rizwan

Created: 22 Sep, 2025

Dependencies:
  - pandas
  - matplotlib
  - seaborn
  - scipy

Usage:
  Import the required functions to perform various data analysis tasks:
  from helper-1 import visualize_box_plots, handle_outliers, calculate_outliers_percentage, mutual_information, feature_engineering

Note:
  This module is designed to work with pandas DataFrames containing numerical data.
  Some functions assume the presence of numerical columns in the input DataFrame.

"""
# Relational Plot from the Feature and Label

def visualize_relation_plot(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Embarked', y='Survived', pallete='viridis')
    plt.title("Relational Plot for the Feature and Label")
    plt.xlabel("Embarked X Survived")
    plt.ylabel("Count")
    plt.show()

# Visualize the Outliers in Box Plots

def visualize_numerical_box_plots(df):
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

# Alternative version: If you want to visualize categorical data distribution itself
def visualize_categorical_distributions(df):
    """
    Visualizes the distribution of categorical features using count plots and bar charts.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the categorical features to visualize.
    
    Returns:
    - None: Displays the plots.
    """
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object'])

    # Set up the figure
    num_cols = 3
    num_rows = (len(categorical_columns) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows))
    fig.suptitle('Distribution of Categorical Features', fontsize=16)

    axes = axes.flatten()
    
    # Create count plots for each categorical column
    for i, col in enumerate(categorical_columns):
        sns.countplot(data=df, x=col, ax=axes[i], palette='viridis')
        axes[i].set_title(f'Distribution of {col}', fontsize=14)
        axes[i].set_xlabel(col, fontsize=12)
        axes[i].set_ylabel('Count', fontsize=12)
        axes[i].tick_params(axis='x', rotation=45)
    
    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Handle Outliers in Dataset

def handle_outliers(df, columns, all_columns=True):
    """
    Handles outliers in a DataFrame by capping based on the IQR method.

    Parameters:
    - df (pd.DataFrame): DataFrame to process.

    Returns:
    - pd.DataFrame: DataFrame with outliers handled.
    """
    numerical_cols = []
    
    if all_columns:
        numerical_cols = df.select_dtypes(include=['number']).columns
    else:
        numerical_cols = columns
    
    for column in numerical_cols:
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
    """
    Calculate the Percentage of the Outliers which is present in 
    each columns
    
    Parameters:
    - df (pd.DataFrame): DataFrame to process.
    
    Returns:
    - None: Print the Calculated Outliers Percentage
    """
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

def handle_skewness(df, columns, threshold=1.0, all_columns=True):
    """
    Applies Box-Cox transformation to numerical columns in the DataFrame where skewness exceeds a threshold.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - threshold (float): Skewness threshold to decide which columns to transform.
    
    Returns:
    - pd.DataFrame: DataFrame with transformed columns.
    - dict: Dictionary of lambda values used for Box-Cox transformation for each column.
    """
    if all_columns:
        numeric_cols = df.select_dtypes(include=['number']).columns
    else:
        numeric_cols = columns
   
    lambda_dict = {} 
    
    for col in  numeric_cols:
        skewness = df[col].skew()
        # Check the skewness and ensure positive values for Box-Cox
        if skewness > threshold:
            # Adding 1 to shift all data to positive if there are zero or negative values
            df[col] = df[col] + 1
            df[col], fitted_lambda = boxcox(df[col])
            lambda_dict[col] = fitted_lambda
    
    return df, lambda_dict

# Example usage:
# train_data is your DataFrame containing the numerical data
train_data, lambda_values = handle_skewness(train_data)
    
    
# Scale The Dataset

def scale_dataset(dataframe, OverSample=True):
    """
    Simple way to rebalance the dataset for the categorical features
    
    Parameters:
    - df (pd.DataFrame): DataFrame to process
    - OverSample (Boolean): Need to be ReSample Data
    
    Return:
    - data (numpy.array()): return an 2d numpy array of the dataset with feature and target
    - X (numpy.array()): return an 2d numpy array of the dataset with only features
    - y (numpy.array()): return an 1d numpy array of the dataset with only label
    """
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values
    
    scaler = StanderScaler()
    X = scaler.fit_transform(X)
    
    if OverSample:
        ros = RandomOverSample()
        X, y = ros.fit_resample(X, y)
        
    data = np.hstack((X, np.reshape(y, (-1, 1))))
    
    return data, X, y

# Feature Engineering of OneHotEncoding

def feature_one_hot_encoder(df, column, drop_columns=True):
    ohe = OneHotEncoder(sparse_output=False, drop=None)
    matrix = ohe.fit_transform(df[[column]])
    ohe_cols = sorted(df[column].unique())
    ohe_df = pd.DataFrame(matrix, columns=ohe_cols, index=df.index)
    df = pd.concat([df, ohe_df], axis=1)

    if drop_columns:
        df.drop(columns=columns, inplace=True)

    return df


# Feature Enginerring of LabelEncoding

# It work for only one column not for the multiple columns
def feature_label_encoding(df, encode_column):
    le = LabelEncoder()
    
    df[encode_column] = le.fit_transform(df[encode_column])
    
    return df

# Numerical Mutual Information

def numerical_mi_score(X, y):
    """
    Find the MI Scores of all the Numerical Feature in the dataset.
    It help to identify which features are more important for the model performance.
    (if the dataset is large enough to handle)
    
    Parameters:
    - X (pd.DataFrame): Features to process
    - y (pd.DataFrame): Label to process
    
    Returns:
    - None
    """
    mi_score = mutual_info_regression(X, y, random_state=42)
    mi_score = pd.Series(mi_score, index=X.columns)
    mi_score = mi_score.sort_values(ascending=False)
    print(mi_score)
    
# Categorical Mutual Information

def categorial_mi_score(X, y):
    """
    Find the MI Scores of all the Categorical Feature in the dataset.
    It help to identify which features are more important for the model performance.
    (if the dataset is large enough to handle)
    
    Parameters:
    - X (pd.DataFrame): Features to process
    - y (pd.DataFrame): Label to process
    
    Returns:
    - None
    """
    categorical_columns = X.select_dtypes(include=['object']).columns
    order_encoded = OrdinalEncoder()
    categorical_decode= order_encoded.fit_transform(X[categorical_columns])
    
    mi_score = mutual_info_classif(categorical_decode, y, discrete_features=True, random_state=42)
    mi_score = pd.Series(mi_score, index=categorical_columns)
    mi_score = mi_score.sort_values(ascending=False)
    print(mi_score)
    

# Neural Network Model

# input_size will be consider by the column of the dataset
nn_model = tf.keras.Sequantial([
    tf.keras.layers.Dense(64, activation='relu', input_size=(20,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

pred = nn_model.predict(X_test)

pred = (pred > 0.5).astype(int).reshape(-1,)


# OverSampling Dataset for Rebalanced the dataset

def feature_rebalanced(X, y):
    """
    Impliment the resample library to rebalance the dataset for the 
    balance dataset of features
    
    Parameters:
    - X (pd.DataFrame): Features to process
    - y (pd.DataFrame): Labels to process
    
    Returns:
    - X (pd.DataFrame): Rebalance the Feature
    - y (pd.DataFrame): Rebalance the Label
    """
    
    smote = SMOTE(random_state=42)
    
    X, y = smote.fit_resample(X, y)
    
    return X, y

# Correlation Matrix

def correlation_feature_matrix(df):
    """
    Print the Correlation with the feature and target
    
    Parameters:
    - df (pd.DataFrame): DataFrame to process
    
    Returns:
    - None
    """
    
    numerical_columns = df.select_dtypes(include=['number']).columns
    
    corr = df[numerical_columns].corr()['loan_status'].sort_values(ascending=False)
    corr_abs = corr.abs().sort_values(ascending=False)
    
    print(corr_abs)


# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Regression Prediction Plot

def visualize_prediction_plot(X, y, X_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual Data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label="Regression Line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Regression Plot")
    plt.legend()
    plt.show()
    
# Regression Prediction
# R2 Score
# Mean Squared Error