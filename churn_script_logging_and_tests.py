"""
Description: Helper functions to assist with function's sanity 
Author: V.Manousakis-Kokorakis
Date: 11-09-2023
"""

# Standard libraries
import logging
import os

# Third-party libraries
import joblib
import yaml

# Local application / own libraries
import churn_library as cls
from constant import CAT_COLUMNS

# Read the configuration file
with open('./config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)


logging.basicConfig(
    filename=f"./{CONFIG['paths']['logs']}/churn_library.log",
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    """
    Test the data import functionality.

    Parameters:
    - import_data (function): Function to import data.
    
    Returns:
    - DataFrame: Imported data if successful. Raises error otherwise.

    Raises:
    - FileNotFoundError: If the data file does not exist.
    - AssertionError: If the DataFrame is empty.
    """
    logging.info("Test Import")
    data_frame = None
    try:
        data_frame = import_data("./data/bank_data.csv")
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except FileNotFoundError as err:
        logging.error("Testing import_eda execution: The file wasn't found")
        raise err
    except AssertionError as err:
        logging.error(
            "Testing import_data functionnality: The file doesn't appear to have rows and columns")
        raise err
    else:
        logging.info("Testing import_data functionnality: SUCCESS")
        return data_frame


def test_eda(perform_eda, dataframe):
    """
    Test the Exploratory Data Analysis (EDA) functionality.

    Parameters:
    - perform_eda (function): Function to perform EDA.
    - dataframe (DataFrame): The data to analyze.
    
    Raises:
    - AssertionError: If the expected files are not created.
    - Exception: For any other errors.
    """
    logging.info("Test EDA")
    try:
        perform_eda(dataframe, False)
        assert os.path.exists(f"./{CONFIG['paths']['eda']}/test.png")
        assert os.path.exists(f"./{CONFIG['paths']['eda']}/Customer_Age.png")
    except AssertionError as err:
        logging.error("Testing perform_eda functionnality: Files not created")
        raise err
    except Exception as err:
        logging.error("Testing perform_eda execution: Issue encountered")
        raise err
    else:
        logging.info("Testing perform_eda functionnality: SUCCESS")


def test_encode_categorical_variables(encode_categorical_variables, data_frame, cat_column, response):
    """
    Test the encoder helper function.

    Parameters:
    - encode_categorical_variables (function): Function to encode categorical variables.
    - data_frame (DataFrame): The original data.
    - cat_column (list): List of categorical columns to encode.
    - response (str): Target variable.
    
    Raises:
    - AssertionError: If the output DataFrame is empty.
    - Exception: For any other errors.
    """
    logging.info("Test Encoder Helper")
    try:
        data_frame_out = encode_categorical_variables(data_frame, cat_column, response)
        assert len(data_frame) > 0
    except AssertionError as err:
        logging.error(
            "Testing encode_categorical_variables functionnality: output data frame empty")
        raise err
    except Exception as err:
        logging.error("Testing encode_categorical_variables execution: Issue encountered")
        raise err
    else:
        logging.info("Testing encode_categorical_variables functionnality: SUCCESS")


def test_perform_feature_engineering(perform_feature_engineering, dataset):
    """
    Test the feature engineering functionality.

    Parameters:
    - perform_feature_engineering (function): Function to perform feature engineering.
    - dataset (DataFrame): The original data.
    
    Returns:
    - x_train, x_test, y_train, y_test (DataFrame, DataFrame, Series, Series): Training and test sets.

    Raises:
    - AssertionError: If any of the data sets are empty.
    - Exception: For any other errors.
    """
    logging.info("Test Feature Engineering")
    x_train, x_test, y_train, y_test = None, None, None, None
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            dataset, response='Churn')
        assert len(x_train) > 0
        assert len(x_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering functionnality: one "
            "of the output data frame is empty")
        raise err
    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering execution: Issue encountered")
        raise err
    else:
        logging.info(
            "Testing perform_feature_engineering functionnality: SUCCESS")
        return x_train, x_test, y_train, y_test


def test_train_models(train_models, x_train, y_train):
    """
    Test the model training functionality.

    Parameters:
    - train_models (function): Function to train models.
    - x_train (DataFrame): Feature matrix for training.
    - y_train (Series): Target variable for training.
    
    Raises:
    - AssertionError: If the expected model files are not created.
    - Exception: For any other errors.
    """
    logging.info("Test Train Models")
    try:
        train_models(x_train, y_train)
        assert os.path.exists(f"./{CONFIG['paths']['models']}/rfc_model.pkl")
        assert os.path.exists(f"./{CONFIG['paths']['models']}/logistic_model.pkl")
    except AssertionError as err:
        logging.error(
            "Testing train_models functionnality: one of the output data "
            "frame is empty")
        raise err
    except Exception as err:
        logging.error(
            "Testing train_models execution: Issue encountered")
        raise err
    else:
        logging.info(
            "Testing train_models functionnality: SUCCESS")


def test_predict_models(predict_models, model, predict_input):
    """
    Test the model prediction functionality.

    Parameters:
    - predict_models (function): Function to make predictions.
    - model (object): Trained model for making predictions.
    - predict_input (DataFrame): Input features for prediction.
    
    Returns:
    - predict_values (Series or ndarray): The predicted values.
    
    Raises:
    - AssertionError: If length of the predicted values and input are not equal.
    - Exception: For any other errors.
    """
    predict_values = None
    logging.info("Test Predict Models")
    try:
        predict_values = predict_models(model, predict_input)
        assert len(predict_values)==len(predict_input)
    except AssertionError as err:
        logging.error(
            "Testing predict_models functionnality: one of the output data "
            "frame is empty")
        raise err
    except Exception as err:
        logging.error(
            "Testing predict_models execution: Issue encountered")
        raise err
    else:
        logging.info(
            "Testing predict_models functionnality: SUCCESS")
        return predict_values


def test_evaluate_models(evaluate_models, lr_model, rfc_model,
                    x_train, x_test,
                    y_train, y_test,
                    y_train_preds_lr, y_train_preds_rf,
                    y_test_preds_lr, y_test_preds_rf):
    """
    Test the model evaluation functionality.

    Parameters:
    - evaluate_models (function): Function to evaluate models.
    - lr_model, rfc_model (object): Trained logistic regression and random forest models.
    - x_train, x_test, y_train, y_test (DataFrame, DataFrame, Series, Series): Training and test sets.
    - y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf: Predicted values.
    
    Raises:
    - AssertionError: If the expected evaluation files are not created.
    - Exception: For any other errors.
    """
    logging.info("Test Evaluate Models")
    try:
        evaluate_models(lr_model, rfc_model,
                    x_train, x_test,
                    y_train, y_test,
                    y_train_preds_lr, y_train_preds_rf,
                    y_test_preds_lr, y_test_preds_rf)
        assert os.path.exists(
            f"{CONFIG['paths']['results']}/Classification_report_Logistic_Regression.png")
        assert os.path.exists(
            f".{CONFIG['paths']['results']}/Classification_report_Random_Forest.png")
        assert os.path.exists(f"{CONFIG['paths']['results']}/ROC_curves.png")
        assert os.path.exists(f"{CONFIG['paths']['results']}/feature_importance.png")
    except AssertionError as err:
        logging.error(
            "Testing evaluate_models functionnality: one of the output data "
            "frame is empty")
        raise err
    except Exception as err:
        logging.error(
            "Testing evaluate_models execution: Issue encountered")
        raise err
    else:
        logging.info(
            "Testing evaluate_models functionnality: SUCCESS")
        return x_train, x_test, y_train, y_test


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Testing data import
    logging.info("Starting test for data import.")
    data_frame = test_import(cls.import_data)
    logging.info("Completed test for data import.")

    # Testing EDA
    logging.info("Starting test for Exploratory Data Analysis (EDA).")
    test_eda(cls.perform_eda, data_frame)
    logging.info("Completed test for EDA.")

    # Testing Encoding of Categorical Variables
    logging.info("Starting test for encoding categorical variables.")
    test_encode_categorical_variables(cls.encode_categorical_variables,
                                    data_frame,
                                    CAT_COLUMNS,
                                    "Churn")
    logging.info("Completed test for encoding categorical variables.")

    # Testing Feature Engineering
    logging.info("Starting test for feature engineering.")
    x_train, x_test, y_train, y_test = test_perform_feature_engineering(
        cls.perform_feature_engineering,
        data_frame)
    logging.info("Completed test for feature engineering.")

    # Testing Model Training
    logging.info("Starting test for model training.")
    test_train_models(cls.train_models, x_train, y_train)
    logging.info("Completed test for model training.")

    # Loading Trained Models
    logging.info("Loading trained models.")
    rfc_model = joblib.load(f'./{CONFIG["paths"]["models"]}/rfc_model.pkl')
    lr_model = joblib.load(f'./{CONFIG["paths"]["models"]}/logistic_model.pkl')
    logging.info("Loaded trained models successfully.")

    # Testing Model Prediction for Random Forest
    logging.info("Starting test for model prediction using Random Forest.")
    y_train_preds_rf = test_predict_models(cls.predict_models, rfc_model, x_train)
    y_test_preds_rf = test_predict_models(cls.predict_models, rfc_model, x_test)
    logging.info("Completed test for model prediction using Random Forest.")

    # Testing Model Prediction for Logistic Regression
    logging.info("Starting test for model prediction using Logistic Regression.")
    y_train_preds_lr = test_predict_models(cls.predict_models, lr_model, x_train)
    y_test_preds_lr = test_predict_models(cls.predict_models, lr_model, x_test)
    logging.info("Completed test for model prediction using Logistic Regression.")

    # Testing Model Evaluation
    logging.info("Starting test for model evaluation.")
    test_evaluate_models(cls.evaluate_models,
                        lr_model, rfc_model,
                        x_train, x_test,
                        y_train, y_test,
                        y_train_preds_lr, y_train_preds_rf,
                        y_test_preds_lr, y_test_preds_rf)
    logging.info("Completed test for model evaluation.")
