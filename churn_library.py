"""
Description: Helper functions to assist with the customer churn classification
Author: V.Manousakis-Kokorakis
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

# Local imports
from constant import CAT_COLUMNS, KEEP_COLUMNS

IMAGES_PATH = "./images/"
MODELS_PATH = "./models/"

# Common function to save and optionally show plot
# def save_show_plot(filename, show_flag):
#     plt.tight_layout()
#     plt.savefig(os.path.join(IMAGES_PATH, filename))
#     if show_flag:
#         plt.show()
#     plt.close()

# Importing the data
def import_data(pth):
    return pd.read_csv(pth)

# EDA functions
def save_show_plot(filepath, show_flag):
    # Your save or show plot code here
    plt.savefig(filepath)
    if show_flag:
        plt.show()

def perform_eda(data_frame, show_flag=True):
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    
    plots = [
        ('Churn', lambda: data_frame['Churn'].hist(), 'test.png'),
        ('Customer_Age', lambda: data_frame['Customer_Age'].hist(), 'Customer_Age.png'),
        ('Marital_Status', lambda: data_frame['Marital_Status'].value_counts('normalize').plot(kind='bar'), 'marital_status.png'),
        ('Total_Trans_Ct', lambda: sns.histplot(data_frame['Total_Trans_Ct'], stat='density', kde=True), 'total_trans_hist.png'),
        ('Heatmap', lambda: sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2), 'corr_heatmap.png')
    ]
    
    for title, plot_fn, filename in plots:
        plt.figure(figsize=(20, 10))
        plot_fn()  # Removed 'kind=bar' from here
        plt.title(title)
        save_show_plot(os.path.join("eda", filename), show_flag)

def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    # Save figure to output_pth
    plt.tight_layout()
    plt.savefig(os.path.join(output_pth, 'feature_importance.png'))

    # display feature importance figure
    plt.show()
    plt.close()

# Feature Engineering
def encoder_helper(data_frame, category_lst, response):
    for category in category_lst:
        avg_churn = data_frame.groupby(category).mean()[response]
        data_frame[f'{category}_Churn'] = data_frame[category].map(avg_churn)
    return data_frame

def perform_feature_engineering(data_frame, response='Churn'):
    y = data_frame[response]
    data_frame = encoder_helper(data_frame, CAT_COLUMNS, response)
    X = data_frame[KEEP_COLUMNS]
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Reporting
def generate_classification_report(model_name, y_train, y_test, y_train_preds, y_test_preds):
    report_data = [
        (f'{model_name} Train', y_train, y_train_preds),
        (f'{model_name} Test', y_test, y_test_preds)
    ]
    plt.figure(figsize=(5, 5))
    for title, y, y_preds in report_data:
        plt.text(0.01, 1.25, title, {'fontsize': 10})
        plt.text(0.01, 0.05, str(classification_report(y, y_preds)), {'fontsize': 10})
    plt.axis('off')
    save_show_plot(os.path.join("results", f'Classification_report_{model_name}.png'), show_flag=False)

def evaluate_models(lrc_model, rfc_model,
                    x_train, x_test,
                    y_train, y_test,
                    y_train_preds_lr, y_train_preds_rf,
                    y_test_preds_lr, y_test_preds_rf):
    '''
    evaluate model results: images + scores
    input:
              lrc_model: logarithmic regression model
              rfc_model: random forest model
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
              y_train_preds_lr: y predicted data by log regression on training
              y_train_preds_rf: y predicted data by log regression on training
              y_test_preds_lr: y predicted data by log regression on test
              y_test_preds_rf: y predicted data by log regression on test
    output:
              None
    '''
    # Generate classification reports (placeholder)
    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf)

    # Compute ROC curve and ROC area for Logistic Regression
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_test_preds_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)

    # Compute ROC curve and ROC area for Random Forest
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_test_preds_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    # Plot ROC curves
    plt.figure(figsize=(15, 8))
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Random classifier
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join("./results", 'ROC_curves.png'))
    plt.close()

    # Feature importance plot (placeholder)
    feature_importance_plot(model=rfc_model,
                            x_data=x_train,
                            output_pth="./results")

    # SHAP Summary Plot
    explainer = shap.TreeExplainer(rfc_model)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar")
    plt.savefig('./results/SHAP.png')

def predict_models(model, predict_input):
        '''
        train, store model results: images + scores, and store models
        input:
                model: model with which to do the prediciton
                predict_input: input for prediction
        output:
                prediction
        '''
        return model.predict(predict_input)

def train_models(x_train, y_train):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              y_train: y training data
    output:
              None
    '''
    # List of models and their parameter spaces
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

    # Loop through the list of models
    for model_param in tqdm(models_params):
        model = model_param['model']
        params = model_param['params']
        name = model_param['name']

        random_search = RandomizedSearchCV(estimator=model, param_distributions=params, 
                                           n_iter=2, cv=3, random_state=42)
        random_search.fit(x_train, y_train)

        # Save the best model
        joblib.dump(random_search.best_estimator_, f'./models/{name}.pkl')

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    generate_classification_report('Logistic_Regression',
                                   y_train,
                                   y_test,
                                   y_train_preds_lr,
                                   y_test_preds_lr)

    generate_classification_report('Random_Forest',
                                   y_train,
                                   y_test,
                                   y_train_preds_rf,
                                   y_test_preds_rf)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Importing Data")
    dataset = import_data("./data/bank_data.csv")
    logging.info("Perform EDA")
    perform_eda(dataset, show_flag=False)
    logging.info("Feature Engineering")
    x_train, x_test, y_train, y_test = perform_feature_engineering(dataset, response='Churn')
    logging.info("Train Models")
    train_models(x_train, y_train)
    logging.info("Load Trained Models")
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

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