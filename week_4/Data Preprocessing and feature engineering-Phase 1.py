import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = 'd:/INTERNSHIP_CLEBAL_TECHNOLOGY/PROJECTS/week_4/titanic.csv'
data = pd.read_csv(file_path)

# Handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data.dropna(subset=['Embarked'], inplace=True)
data.drop(columns=['Cabin'], inplace=True)

# Impute missing values in 'Age' and 'Fare'
imputer = SimpleImputer(strategy='median')
data[['Age', 'Fare']] = imputer.fit_transform(data[['Age', 'Fare']])

# Encode categorical features
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_features = pd.DataFrame(encoder.fit_transform(data[['Sex', 'Embarked']]), columns=encoder.get_feature_names_out(['Sex', 'Embarked']))
data = pd.concat([data, encoded_features], axis=1)
data.drop(columns=['Sex', 'Embarked', 'Name', 'Ticket'], inplace=True)

# Create a new feature 'FamilySize'
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data.drop(columns=['SibSp', 'Parch'], inplace=True)

# Ensure no remaining missing values in the final dataset
data.dropna(inplace=True)

# Print the columns of the DataFrame to check the correct names
print("Columns after encoding and preprocessing:")
print(data.columns)

# Histogram of Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['Age'], bins=20, kde=True, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Assuming the column for male is 'Sex_male' after encoding, update the bar plot accordingly
plt.figure(figsize=(8, 6))
sns.barplot(x=data['Sex_male'], y=data['Survived'], palette='Set1')
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.xticks([0, 1], ['Female', 'Male'])  # Add labels for clarity
plt.grid(True)
plt.show()

# Define features and target
X = data.drop(columns=['Survived'])
y = data['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features after splitting the data
scaler = StandardScaler()
X_train[['Age', 'Fare']] = scaler.fit_transform(X_train[['Age', 'Fare']])
X_test[['Age', 'Fare']] = scaler.transform(X_test[['Age', 'Fare']])

# Train a Logistic Regression model with increased max_iter and different solver
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the Logistic Regression model: {accuracy:.2f}')
