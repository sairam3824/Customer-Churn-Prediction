# Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-blue)

This repository contains the code and resources for predicting customer churn using various machine learning models. The project aims to identify customers who are likely to stop using a service, which is crucial for businesses to retain customers and improve customer satisfaction.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Customer churn prediction is a common problem in the business world, where companies aim to identify customers who are likely to stop using their services. This project uses machine learning techniques to predict customer churn based on historical data. The models implemented include:
- Random Forest Classifier
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Logistic Regression
- XGBoost
- K-Nearest Neighbors (KNN)

## Dataset
The dataset used in this project is `Customer-Churn-Records.csv`, which contains various features related to customer behavior and demographics. The target variable is `Exited`, which indicates whether a customer has churned (1) or not (0).

### Dataset Columns:
- `RowNumber`
- `CustomerId`
- `Surname`
- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Exited`
- `Complain`
- `Satisfaction Score`
- `Card Type`
- `Point Earned`

## Models Used
The following machine learning models were implemented and evaluated:
1. **Random Forest Classifier**: Achieved an accuracy of 0.999.
2. **Support Vector Machine (SVM)**: Achieved an accuracy of 0.797.
3. **Decision Tree Classifier**: Achieved an accuracy of 0.9985.
4. **Logistic Regression**: Achieved an accuracy of 0.9985.
5. **XGBoost**: Achieved an accuracy of 0.999.
6. **K-Nearest Neighbors (KNN)**: Achieved an accuracy of 0.765.

## Results
The accuracy of each model is summarized below:

| Model                  | Accuracy |
|------------------------|----------|
| Random Forest          | 0.999    |
| XGBoost                | 0.999    |
| Decision Tree          | 0.9985   |
| Logistic Regression    | 0.9985   |
| SVM                    | 0.797    |
| K-Nearest Neighbors    | 0.765    |

A bar graph comparing the accuracies of the models is also provided in the notebook.

## Installation
To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sairam3824/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
2. **Install the required dependencies**:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn xgboost
3. **Run the Jupyter Notebook**:
  ```bash
   jupyter notebook Customer_Churn_Prediction.ipynb
  ```
## Usage

1. Open the Jupyter Notebook `Customer_Churn_Prediction.ipynb`.
2. Run each cell to load the dataset, preprocess the data, train the models, and evaluate their performance.
3. The notebook includes visualizations and comparisons of the models' accuracies.

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeatureName`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeatureName`).
5. Open a pull request.
