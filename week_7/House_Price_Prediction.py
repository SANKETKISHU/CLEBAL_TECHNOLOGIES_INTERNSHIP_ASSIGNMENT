import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

# Load datasets
train_data = pd.read_csv('D:/INTERNSHIP_CLEBAL_TECHNOLOGY/PROJECTS/week_7/train.csv')
test_data = pd.read_csv('D:/INTERNSHIP_CLEBAL_TECHNOLOGY/PROJECTS/week_7/test.csv')

# Separate features and target
X = train_data.drop(['SalePrice', 'Id'], axis=1)
y = train_data['SalePrice']
test_ids = test_data['Id']
X_test = test_data.drop('Id', axis=1)

# Combine train and test data for preprocessing
data = pd.concat([X, X_test], keys=['train', 'test'])

# Data exploration
def data_exploration(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['SalePrice'], kde=True)
    plt.title('SalePrice Distribution')
    plt.show()

    corr_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

data_exploration(train_data)

# Handle missing values
imputer = SimpleImputer(strategy='median')
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
data[num_cols] = imputer.fit_transform(data[num_cols])

# Encode categorical variables
cat_cols = data.select_dtypes(include=['object']).columns
data[cat_cols] = data[cat_cols].fillna('None')

# Apply label encoding to categorical features
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# Split data back into train and test sets
X = data.xs('train')
X_test = data.xs('test')

# Outlier detection and handling (optional)
def detect_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))

outliers = detect_outliers(X[num_cols])
X = X[~(outliers.any(axis=1))]
y = y[~(outliers.any(axis=1))]  # Ensure y is consistent with X after outlier removal

# Feature engineering (adjustment to avoid SettingWithCopyWarning)
X.loc[:, 'TotalSF'] = X['1stFlrSF'] + X['2ndFlrSF'] + X['TotalBsmtSF']
X_test.loc[:, 'TotalSF'] = X_test['1stFlrSF'] + X_test['2ndFlrSF'] + X_test['TotalBsmtSF']

# Model tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_features': ['sqrt', 'log2'],  # Valid options for max_features
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search.fit(X, y)
best_model = grid_search.best_estimator_

# Evaluate model
scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f'RMSE: {rmse_scores.mean()}')

# Fit model and make predictions
best_model.fit(X, y)
predictions = best_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': predictions
})
submission.to_csv('D:/INTERNSHIP_CLEBAL_TECHNOLOGY/PROJECTS/week_7/submission.csv', index=False)
