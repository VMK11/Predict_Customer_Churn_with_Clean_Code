# Predict Customer Churn

- Project **Predict Customer Churn** is part of Udacity's ML DevOps Engineer Nanodegree Program

## Project Description

This project aims to transform a Proof of Concept (POC) into a production-ready application, implementing best practices covered in Udacity's MLOps Chapter 1. These practices include:
- Code Refactoring
- Commenting and DocString
- Modularity
- Testing Procedures
- Log Management
- Adherence to AutoPEP8 and Pylint standards

The initial project, provided by Udacity, predicts customer churn for bank clients.

Workflow:
The project progresses through the following stages:

- EDA (Exploratory Data Analysis): Initial data exploration and analysis
- Feature Engineering: Dataset transformation for model training
- Model Training: Training of two sklearn classification models (Random Forest and Logistic Regression)
- Post-Training Analysis: Utilization of the SHAP library to understand feature impact
- Data Storage: Saving the best-performing models and related metrics

## Project Structure

The project is organized into the following folders and files:
- PROJECT_FOLDER
    - data
        - bank_data.csv                   --> csv dataset
    - images                              --> contains model scores, confusion matrix, ROC curve
        - eda                             --> results of the EDA
        - results                         --> Training and evaluation results
    - logs                                --> log generated during testing
    - models                              --> contains saved models in .pkl format
    - churn_library.py                    --> Main entry file containing all the functions
    - churn_script_logging_and_testing.py --> testing script for churn_library.py
    - constant.py                         --> ontains constant informations such as columns to process
    - requirements.txt              --> requirements for the execution

All Python files comply with PEP8 standards and have the following Pylint ratings:

- churn_library.py = 7.36
- constant.py      = 10
- churn_script_logging_and_testing.py = 7.80/10


## Execution Guide
How do you run your files? What should happen when you run your files?

The following project were tested using python  3.6 and all listed packages inside the requirements.txt.

To install required packages, run the following command:

```
pip install -r requirements.txt
```

### How to run the project

You can initiate the project using one of the two entry points:
- Execute churn_library.py using
```
python churn_library.py
```


### Functionality Testing

Each function in the project comes with a dedicated testing module. To run these tests, execute:
```
python churn_script_logging_and_tests.py
```
The console will display the functions being tested. For more detailed logs, refer to the logs/churn_library.log file.
