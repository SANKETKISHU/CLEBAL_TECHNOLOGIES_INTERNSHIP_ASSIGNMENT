import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('D:/INTERNSHIP_CLEBAL_TECHNOLOGY/PROJECTS/week_6/titanic.csv')

# 1. Handling missing values
# Fill missing 'Age' with median
imputer = SimpleImputer(strategy='median')
data['Age'] = imputer.fit_transform(data[['Age']])

# Fill missing 'Embarked' with the most frequent value
imputer = SimpleImputer(strategy='most_frequent')
data['Embarked'] = imputer.fit_transform(data[['Embarked']])

# Fill missing 'Fare' with median
data['Fare'] = imputer.fit_transform(data[['Fare']])

# Drop 'Cabin' due to too many missing values
data = data.drop(columns=['Cabin'])

# 2. Encoding categorical variables
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['Embarked'] = label_encoder.fit_transform(data['Embarked'])

# 3. Scaling features
scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

# 4. Detecting outliers using Z-score
def detect_outliers_zscore(data, threshold=3):
    outliers = []
    mean = np.mean(data)
    std = np.std(data)
    for i in data:
        z_score = (i - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers

outliers_age = detect_outliers_zscore(data['Age'])
outliers_fare = detect_outliers_zscore(data['Fare'])

print("Outliers in Age: ", outliers_age)
print("Outliers in Fare: ", outliers_fare)

# Visualizing outliers using boxplot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data['Age'])
plt.title('Age Boxplot')

plt.subplot(1, 2, 2)
sns.boxplot(data['Fare'])
plt.title('Fare Boxplot')

plt.show()

# 5. Feature engineering
# Creating new feature 'FamilySize' from 'SibSp' and 'Parch'
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# Creating new feature 'IsAlone'
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

# Creating new feature 'Title' from 'Name'
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 
                                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data['Title'] = data['Title'].replace('Mlle', 'Miss')
data['Title'] = data['Title'].replace('Ms', 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')

# Encoding 'Title'
data['Title'] = label_encoder.fit_transform(data['Title'])

# Drop columns that won't be used for modeling
data = data.drop(columns=['Name', 'Ticket', 'PassengerId'])

# Save the cleaned and engineered dataset to a new CSV file
data.to_csv('D:/INTERNSHIP_CLEBAL_TECHNOLOGY/PROJECTS/week_6/titanic_cleaned.csv', index=False)

print("Data preprocessing and feature engineering completed successfully.")
