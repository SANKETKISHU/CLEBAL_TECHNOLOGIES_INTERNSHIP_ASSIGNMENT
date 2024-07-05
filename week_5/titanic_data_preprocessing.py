import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the datasets
train_df = pd.read_csv('D:/INTERNSHIP_CLEBAL_TECHNOLOGY/PROJECTS/week_5/train.csv')
test_df = pd.read_csv('D:/INTERNSHIP_CLEBAL_TECHNOLOGY/PROJECTS/week_5/test.csv')

# Step 1: Data Cleaning
train_df_cleaned = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_df_cleaned = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Step 2: Handling Missing Values
train_df_cleaned['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df_cleaned['Age'].fillna(train_df['Age'].median(), inplace=True)

test_df_cleaned['Fare'].fillna(train_df['Fare'].median(), inplace=True)

train_df_cleaned['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
test_df_cleaned['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# Step 3: Transformation
train_df_cleaned['Sex'] = train_df_cleaned['Sex'].map({'male': 0, 'female': 1})
test_df_cleaned['Sex'] = test_df_cleaned['Sex'].map({'male': 0, 'female': 1})

train_df_cleaned = pd.get_dummies(train_df_cleaned, columns=['Embarked'], drop_first=True)
test_df_cleaned = pd.get_dummies(test_df_cleaned, columns=['Embarked'], drop_first=True)

# Step 4: Normalization/Scaling
scaler = StandardScaler()
train_df_cleaned[['Age', 'Fare']] = scaler.fit_transform(train_df_cleaned[['Age', 'Fare']])
test_df_cleaned[['Age', 'Fare']] = scaler.transform(test_df_cleaned[['Age', 'Fare']])

# Step 5: Feature Engineering
train_df_cleaned['FamilySize'] = train_df_cleaned['SibSp'] + train_df_cleaned['Parch'] + 1
test_df_cleaned['FamilySize'] = test_df_cleaned['SibSp'] + test_df_cleaned['Parch'] + 1

train_df_cleaned.drop(['SibSp', 'Parch'], axis=1, inplace=True)
test_df_cleaned.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# Display the first few rows of the cleaned datasets
print("Training dataset:")
print(train_df_cleaned.head())
print("\nTest dataset:")
print(test_df_cleaned.head())

# Step 6: Exploratory Data Analysis
print("\nTraining Dataset Descriptive Statistics:")
print(train_df_cleaned.describe())

print("\nTest Dataset Descriptive Statistics:")
print(test_df_cleaned.describe())

# Step 7: Model Training
# Split data into training and validation sets
X = train_df_cleaned.drop('Survived', axis=1)
y = train_df_cleaned['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'\nValidation Accuracy: {accuracy}')
