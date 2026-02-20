# House-Price-Prediction
This project focuses on predicting house prices using Machine Learning techniques. The goal is to build and compare multiple regression models to estimate property prices based on various input features.
The project includes data preprocessing, feature engineering, model training, hyperparameter tuning, and performance evaluation
# Objective
1) To analyze housing data
2) To preprocess and clean the dataset
3) To build multiple regression models
4) To evaluate and compare model performance
5) To select the best-performing model for price prediction
# Dataset
The dataset contains various features related to houses such as:
1) Numerical features (e.g., area, number of rooms, etc.)
2) Categorical features (e.g., location, type, etc.)
3) Target variable: House Price
California Housing Prices - `housing.csv`
Kaggle link: https://www.kaggle.com/datasets/camnugent/california-housing-prices
# Technologies Used
1) Python
2) NumPy
3) Pandas
4) Matplotlib
5) Seaborn
6) Scikit-learn
# Data Preprocessing
1) Handling missing values using SimpleImputer
2) Encoding categorical variables using OneHotEncoder
3) Feature scaling using StandardScaler
4) Column transformation using ColumnTransformer
5) Pipeline creation for clean workflow
# Models Implemented
1) Linear Regression
2) Ridge Regressio
3) Lasso Regression
4) Random Forest Regressor
5) HistGradientBoosting Regressor
# Model Evaluation
1) Train-Test Split
2) K-Fold Cross Validation
3) GridSearchCV for hyperparameter tuning
4) Performance metrics such as:
5) R² Score
6) Mean Squared Error (MSE)
7) Cross-validation scores
# Model Comparison
Multiple models were trained and evaluated to determine the best performing algorithm. Hyperparameter tuning was performed using GridSearchCV to improve model accuracy.
The final model was selected based on:
1)Higher R² score
2)Lower error metrics
3)Better generalization performance
# How to Run the Project
Step 1: Clone the Repository
git clone <repository-link>
cd <project-folder>
Step 2: Install Required Libraries
pip install numpy pandas matplotlib seaborn scikit-learn
Step 3: Run the Notebook
Open the Jupyter Notebook:
jupyter notebook House_Price_Predictions.ipyn
# Project Structure
House_Price_Predictions.ipynb
README.md
dataset.csv (if included)
# Key Features of This Project
1) Clean ML pipeline implementation
2) Multiple regression model comparison
3) Cross-validation for robust evaluation
4) Hyperparameter tuning using GridSearchCV
5) Data visualization using Matplotlib & Seaborn
# Future Improvements
Add feature selection techniques
Deploy the model using Flask / Streamlit
Improve feature engineering
Use advanced ensemble methods
