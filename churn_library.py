"""
Description: Helper functions to assist with the customer churn classification
Author: V.Manousakis-Kokorakis
Date: 11-09-2023
"""

# Standard Libraries
import os
import joblib
import logging

# Third-party Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
from tqdm import tqdm
import shap
import yaml

# Local imports
from constant import CAT_COLUMNS, KEEP_COLUMNS

# Read the configuration file
with open('./config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)


def import_data(file_path):
    """
    Import data from a CSV file and return it as a Pandas DataFrame.

    Parameters:
    file_path (str): The file path of the CSV file to import.

    Returns:
    pd.DataFrame: A Pandas DataFrame containing the imported data.
    """
    return pd.read_csv(file_path)


def save_and_show_plot(file_path, show_flag):
    """
    Save the current matplotlib plot to a file and optionally display it.

    Parameters:
    file_path (str): The file path where the plot will be saved.
    show_flag (bool): A flag that indicates whether to display the plot.

    Returns:
    None
    """
    plt.savefig(file_path)
    if show_flag:
        plt.show()


def perform_eda(data_frame, show_flag=True):
    """
    Perform Exploratory Data Analysis (EDA) on the given DataFrame.

    This function creates histograms, bar plots, and a heatmap to visualize
    the distribution and correlations of the features in the DataFrame.

    Parameters:
    data_frame (pd.DataFrame): The DataFrame on which to perform EDA.
    show_flag (bool, optional): Flag to indicate whether to show the plots.
                                Default is True.

    Returns:
    None
    """
    # Transform the 'Attrition_Flag' column into a binary 'Churn' column
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    # Define the list of plots to create
    plots = [
        ('Churn', lambda: data_frame['Churn'].hist(), 'test.png'),
        ('Customer_Age',
         lambda: data_frame['Customer_Age'].hist(),
         'Customer_Age.png'),
        ('Marital_Status', lambda: data_frame['Marital_Status'].value_counts(
            'normalize').plot(kind='bar'), 'marital_status.png'),
        ('Total_Trans_Ct',
         lambda: sns.histplot(data_frame['Total_Trans_Ct'],
                              stat='density',
                              kde=True),
         'total_trans_hist.png'),
        ('Heatmap', lambda: sns.heatmap(data_frame.corr(), annot=False,
         cmap='Dark2_r', linewidths=2), 'corr_heatmap.png')
    ]

    # Create each plot
    for title, plot_fn, file_name in plots:
        plt.figure(figsize=(20, 10))
        plot_fn()
        plt.title(title)
        save_and_show_plot(os.path.join("eda", file_name), show_flag)


def feature_importance_plot(model, feature_data, output_path):
    """
    Creates and stores a feature importance plot. The plot illustrates
    the importance of each feature according to the trained model.

    Parameters
    ----------
    model : object
        A trained model object that has a `feature_importances_` attribute.

    feature_data : pd.DataFrame
        A pandas DataFrame containing the features used for training/testing.

    output_path : str
        The directory path where the feature importance plot will be saved.

    Returns
    -------
    None

    Notes
    -----
    The function saves the feature importance plot to the specified path
    and also displays it. The plot is saved as 'feature_importance.png'.
    """

    # Calculate feature importances from the model
    importances = model.feature_importances_

    # Sort feature importances in descending order and get the indices
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [feature_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title and labels
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars for each feature
    plt.bar(range(feature_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(feature_data.shape[1]), names, rotation=90)

    # Save figure to output path
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'feature_importance.png'))

    # Show the plot
    plt.show()
    plt.close()


def encode_categorical_variables(data_frame, category_list, response_column):
    """
    Encodes categorical variables based on the mean of the response variable.

    For each category in category_list, this function creates a new column
    in the DataFrame called '{category}_Churn', where 'category' is the name
    of the category. Each entry in the new column is the mean of 'response_column'
    for that category.

    Parameters
    ----------
    data_frame : pd.DataFrame
        The DataFrame containing the categorical variables to be encoded.

    category_list : list of str
        List of column names in data_frame that are categorical and should be encoded.

    response_column : str
        The name of the column in data_frame that holds the response variable.

    Returns
    -------
    pd.DataFrame
        The original DataFrame but with added '{category}_Churn' columns for each category in category_list.
    """

    for category in category_list:
        avg_churn = data_frame.groupby(category).mean()[response_column]
        data_frame[f'{category}_Churn'] = data_frame[category].map(avg_churn)

    return data_frame


def perform_feature_engineering(data_frame, response='Churn'):
    """
    Perform feature engineering on the given DataFrame and then split it into training and test sets.

    This function first encodes the categorical variables based on the mean of the response variable,
    then keeps only the important columns, and finally splits the data into training and test sets.

    Parameters
    ----------
    data_frame : pd.DataFrame
        The DataFrame containing the variables to be used for feature engineering.

    response : str, optional
        The name of the column in data_frame that holds the response variable.
        Default is 'Churn'.

    Returns
    -------
    tuple
        A tuple containing four elements: X_train, X_test, y_train, y_test.

    Notes
    -----
    - CAT_COLUMNS and KEEP_COLUMNS are expected to be defined before calling this function.
    - The function uses a test size of 0.3 and random_state of 42 for splitting the data.
    """

    # Extract the response variable
    y = data_frame[response]

    # Encode categorical columns
    data_frame = encode_categorical_variables(
        data_frame, CAT_COLUMNS, response)

    # Keep only necessary columns
    X = data_frame[KEEP_COLUMNS]

    # Split the data into training and test sets
    return train_test_split(X, y, test_size=0.3, random_state=42)


def generate_classification_report(model_name, y_train, y_test, y_train_preds, y_test_preds):
    """
    Generates a classification report for a given model and saves it as a PNG image.

    The function generates a classification report for both the training and testing sets.
    It then saves this information as a PNG image. The report includes metrics such as
    precision, recall, f1-score, and support for each class.

    Parameters
    ----------
    model_name : str
        The name of the machine learning model for which the report is generated.

    y_train : pd.Series or np.array
        True labels for the training set.

    y_test : pd.Series or np.array
        True labels for the test set.

    y_train_preds : pd.Series or np.array
        Predicted labels for the training set.

    y_test_preds : pd.Series or np.array
        Predicted labels for the test set.

    Returns
    -------
    None
        The function saves the classification report as a PNG image and does not return any value.
    """
    def save_and_show_plot(file_path, show_flag):
        plt.savefig(file_path)
        if show_flag:
            plt.show()

    report_data = [
        (f'{model_name} Train', y_train, y_train_preds),
        (f'{model_name} Test', y_test, y_test_preds)
    ]

    total_lines = 0  # To keep track of total lines across all reports

    for title, y, y_preds in report_data:
        report_str = str(classification_report(y, y_preds))
        num_lines = len(report_str.split('\n'))
        total_lines += num_lines + 2  # Adding 2 for the title and an empty line

    os.makedirs("results", exist_ok=True)  # Create the results directory if it does not exist

    for title, y, y_preds in report_data:
        plt.figure(figsize=(30, total_lines * 0.2))  # Adjust the figure size based on the total lines

        y_position = 1.0
        plt.text(0.01, y_position, title, {'fontsize': 12})
        y_position -= 0.2  # Move down for the title

        report_str = str(classification_report(y, y_preds))
        for line in report_str.split('\n'):
            plt.text(0.01, y_position, line, {'fontsize': 10})
            y_position -= 0.2  # Move down for the next line

        plt.axis('off')
        save_and_show_plot(os.path.join("results", f'Classification_report_{title.replace(" ", "_")}.png'), show_flag=False)
        plt.close()  # Close the current plot to free resources


def evaluate_models(lrc_model, rfc_model,
                    x_train, x_test,
                    y_train, y_test,
                    y_train_preds_lr, y_train_preds_rf,
                    y_test_preds_lr, y_test_preds_rf):
    """
    Evaluate the performance of Logistic Regression and Random Forest models.

    This function performs a detailed evaluation of the Logistic Regression and
    Random Forest models using various metrics and plots. It generates
    classification reports, ROC curves, and feature importance plots, and saves
    these visualizations as PNG images.

    Parameters
    ----------
    lrc_model : object
        Trained Logistic Regression model.

    rfc_model : object
        Trained Random Forest model.

    x_train, x_test : DataFrame or array-like
        Feature matrix for the training and test datasets.

    y_train, y_test : array-like
        Ground truth labels for the training and test datasets.

    y_train_preds_lr, y_test_preds_lr : array-like
        Predicted labels by Logistic Regression for the training and test datasets.

    y_train_preds_rf, y_test_preds_rf : array-like
        Predicted labels by Random Forest for the training and test datasets.

    Returns
    -------
    None
        The function saves evaluation metrics and plots as PNG images but does not
        return any value.
    """

    logging.info("Evaluating Model Performance")
    # Generate classification reports (placeholder function)
    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf)

    logging.info("Generating ROC Curves")
    # Compute ROC curves for both models
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_test_preds_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_test_preds_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    # Plot ROC curves
    plt.figure(figsize=(15, 8))
    plt.plot(
        fpr_lr,
        tpr_lr,
        label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join("./results", 'ROC_curves.png'))
    plt.close()

    logging.info("Generating Feature Importance Plot")
    # Generate feature importance plot (placeholder function)
    feature_importance_plot(
        model=rfc_model,
        feature_data=x_train,
        output_path=f"./{CONFIG['paths']['results']}")

    logging.info("Generating SHAP Summary Plot")
    # Generate and save SHAP Summary Plot
    explainer = shap.TreeExplainer(rfc_model)
    shap_values = explainer.shap_values(x_test)
    plt.figure(figsize=(150, 10))
    shap.summary_plot(shap_values, x_test, plot_type="bar")
    plt.savefig(f"./{CONFIG['paths']['results']}/SHAP.png")


def predict_models(model, predict_input):
    """
    Use the given model to make predictions based on the input data.

    This function takes a pre-trained machine learning model and an input dataset,
    then returns the model's predictions.

    Parameters
    ----------
    model : object
        The pre-trained machine learning model to use for making predictions.

    predict_input : array-like, shape (n_samples, n_features)
        The input data for which to make predictions.

    Returns
    -------
    array-like, shape (n_samples,)
        The predictions generated by the model for the given input data.
    """

    # Use the model to make predictions on the provided input
    predictions = model.predict(predict_input)

    return predictions


def train_models(x_train, y_train):
    """
    Train multiple machine learning models and store the best-performing models.

    This function iterates over a list of pre-defined models and their parameter spaces,
    performing randomized search cross-validation. It then saves the best-performing model
    for each type in a specified directory.

    Parameters
    ----------
    x_train : array-like, shape (n_samples, n_features)
        The feature matrix used for training.

    y_train : array-like, shape (n_samples,)
        The target vector used for training.

    Returns
    -------
    None

    Notes
    -----
    The function saves the best-performing model for each type in the './models/' directory.

    Models Trained:
        - Random Forest Classifier
        - Logistic Regression
    """

    # Define the list of models and their parameter spaces for hyperparameter
    # tuning
    models_params = [
        {
            'model': RandomForestClassifier(random_state=42, max_features='sqrt'),
            'params': {
                'n_estimators': [200, 500],
                'max_depth': [4, 5, 100],
                'criterion': ['gini', 'entropy']
            },
            'name': 'rfc_model'
        },
        {
            'model': LogisticRegression(solver='lbfgs', max_iter=3000),
            'params': {
                'C': np.logspace(-4, 4, 20),
                'penalty': ['l2']
            },
            'name': 'logistic_model'
        }
    ]

    # Loop through the list of models to perform randomized search and save
    # the best model
    for model_param in tqdm(models_params, desc="Training Models"):
        model = model_param['model']
        params = model_param['params']
        name = model_param['name']

        # Perform randomized search cross-validation
        random_search = RandomizedSearchCV(estimator=model, param_distributions=params,
                                           n_iter=2, cv=3, random_state=42)
        random_search.fit(x_train, y_train)

        # Save the best-performing model to disk
        joblib.dump(random_search.best_estimator_, f'./models/{name}.pkl')


def classification_report_image(
        y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf):
    """
    Generate and store classification reports for both training and test datasets.

    This function takes actual and predicted response values for both the training and test datasets
    and generates classification reports using two types of models: Logistic Regression and
    Random Forest. These reports are then saved as images in a designated folder.

    Parameters
    ----------
    y_train : array-like, shape (n_samples,)
        The actual response values for the training dataset.

    y_test : array-like, shape (n_samples,)
        The actual response values for the test dataset.

    y_train_preds_lr : array-like, shape (n_samples,)
        The predicted response values for the training dataset from the Logistic Regression model.

    y_train_preds_rf : array-like, shape (n_samples,)
        The predicted response values for the training dataset from the Random Forest model.

    y_test_preds_lr : array-like, shape (n_samples,)
        The predicted response values for the test dataset from the Logistic Regression model.

    y_test_preds_rf : array-like, shape (n_samples,)
        The predicted response values for the test dataset from the Random Forest model.

    Returns
    -------
    None

    Notes
    -----
    The function saves the classification reports as images in a designated folder.
    It uses the helper function `generate_classification_report` to actually create the reports.
    """

    # Generate and save the classification report for Logistic Regression
    generate_classification_report('Logistic_Regression',
                                   y_train,
                                   y_test,
                                   y_train_preds_lr,
                                   y_test_preds_lr)

    # Generate and save the classification report for Random Forest
    generate_classification_report('Random_Forest',
                                   y_train,
                                   y_test,
                                   y_train_preds_rf,
                                   y_test_preds_rf)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Importing Data")
    dataset = import_data(f'./{CONFIG["paths"]["data"]}/bank_data.csv')

    logging.info("Perform EDA")
    perform_eda(dataset, show_flag=False)

    logging.info("Feature Engineering")
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        dataset, response='Churn')

    logging.info("Train Models")
    train_models(x_train, y_train)

    logging.info("Load Trained Models")
    rfc_model = joblib.load(f'./{CONFIG["paths"]["models"]}/rfc_model.pkl')
    lr_model = joblib.load(f'./{CONFIG["paths"]["models"]}/logistic_model.pkl')

    logging.info("Predict")
    y_train_preds_rf = predict_models(rfc_model, x_train)
    y_test_preds_rf = predict_models(rfc_model, x_test)

    y_train_preds_lr = predict_models(lr_model, x_train)
    y_test_preds_lr = predict_models(lr_model, x_test)

    logging.info("Evaluate")
    evaluate_models(lr_model, rfc_model,
                    x_train, x_test,
                    y_train, y_test,
                    y_train_preds_lr, y_train_preds_rf,
                    y_test_preds_lr, y_test_preds_rf)
