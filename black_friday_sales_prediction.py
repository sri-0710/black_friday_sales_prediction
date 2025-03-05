!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the Black Friday dataset
data = pd.read_csv('/content/BlackFridaySales.csv')

# Exploratory Data Analysis (EDA)
print(data.head())
print(data.describe())
print(data.info())

# Data Cleaning and Preprocessing
# Handle missing values (e.g., fill with mean or median)
data['Product_Category_2'].fillna(data['Product_Category_2'].mean(), inplace=True)
data['Product_Category_3'].fillna(data['Product_Category_3'].mean(), inplace=True)

# Encode categorical variables
data = pd.get_dummies(data, columns=['Gender', 'Age', 'City_Category', 'Occupation', 'Marital_Status'])

# Feature Selection
# Select relevant features based on domain knowledge and correlation analysis
features = ['Gender_F', 'Gender_M', 'Age_0-17', 'Age_18-25', 'Age_26-35',
            'Age_36-45', 'Age_46-50', 'Age_51-55', 'Age_55+', 'City_Category_A',
            'City_Category_B', 'City_Category_C', 'Occupation_0', 'Occupation_1',
            'Occupation_2', 'Occupation_3', 'Occupation_4', 'Occupation_5',
            'Occupation_6', 'Occupation_7', 'Occupation_8', 'Occupation_9',
            'Occupation_10', 'Occupation_11', 'Occupation_12', 'Occupation_13',
            'Occupation_14', 'Occupation_15', 'Occupation_16', 'Occupation_17',
            'Occupation_18', 'Occupation_19', 'Occupation_20', 'Marital_Status_0',
            'Marital_Status_1', 'Product_Category_1', 'Product_Category_2',
            'Product_Category_3']
target = 'Purchase'

X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualization (optional)
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Purchase')
plt.ylabel('Predicted Purchase')
plt.title('Actual vs. Predicted Purchase')
plt.show()

# Further Improvements
# Consider more advanced models like Random Forest, XGBoost
# Perform hyperparameter tuning
# Conduct more in-depth feature engineering and selection