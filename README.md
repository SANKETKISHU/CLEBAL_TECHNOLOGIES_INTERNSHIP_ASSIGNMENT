
# Internship Assignments at Clebal Technologies

This repository contains Python scripts for various tasks and visualizations completed during my internship at Clebal Technologies.

## Week 1

### Lower and Upper Triangular Pyramids

The `Lower_Upper_Triangular_Pyramids.py` script generates and displays lower and upper triangular matrices.

#### Algorithm Details

- **Lower Triangular Matrix**: This is a type of square matrix where all elements above the main diagonal are zero. For example, in a 3x3 matrix, only elements a11, a21, a22, a31, a32, and a33 are non-zero.
- **Upper Triangular Matrix**: This is a type of square matrix where all elements below the main diagonal are zero. For example, in a 3x3 matrix, only elements a11, a12, a13, a22, a23, and a33 are non-zero.

#### Usage

To generate the pyramids, run the script and follow the prompts.

```sh
python Lower_Upper_Triangular_Pyramids.py
```

## Week 2

### Simple Calculator

The `Calculator_using_Python.py` script provides a simple calculator that can perform basic arithmetic operations.

#### Algorithm Details

- **Addition**: `result = num1 + num2`
- **Subtraction**: `result = num1 - num2`
- **Multiplication**: `result = num1 * num2`
- **Division**: `result = num1 / num2` (with a check to ensure `num2` is not zero to avoid division by zero errors)

#### Usage

To use the calculator, run the script and follow the prompts to enter the operation and the numbers.

```sh
python Calculator_using_Python.py
```

## Week 3

### Iris Data Visualization

The `Iris_Data_Visualization.py` script loads the Iris dataset and generates various visualizations including histograms, pairplots, box plots, and a heatmap.

#### Visualizations

- **Histograms**: Display the distribution of each feature.
- **Pairplot**: Visualize the pairwise relationships between features.
- **Box plots**: Show the distribution of each feature across different species.
- **Heatmap**: Display the correlation between different features.

#### Usage

To generate the visualizations, run the script and the plots will be saved as PNG files in your working directory.

```sh
python Iris_Data_Visualization.py
```

#### Example Output

The script will produce the following output files:
- `iris_histograms.png`
- `iris_pairplot.png`
- `sepal_length_boxplot.png`
- `sepal_width_boxplot.png`
- `petal_length_boxplot.png`
- `petal_width_boxplot.png`
- `iris_heatmap.png`

## Week 4

### Data Preprocessing and Feature Engineering - Phase 1

The `Data_Preprocessing_and_Feature_Engineering_Phase_1.py` script preprocesses the Titanic dataset, performs feature engineering, and trains a Logistic Regression model.

#### Tasks and Algorithm Details

- **Handle Missing Values**: Impute missing values using techniques like mean, median, or mode imputation.
- **Normalize Numerical Features**: Apply standard scaling to features like 'Age' and 'Fare' using `StandardScaler` from scikit-learn.
- **Encode Categorical Features**: Convert categorical variables to numerical form using techniques like one-hot encoding.
- **Create New Features**: Engineer new features such as 'FamilySize' from existing features like 'SibSp' and 'Parch'.
- **Train Logistic Regression Model**: Use the preprocessed data to train a Logistic Regression model from scikit-learn.

#### Usage

To preprocess the data and train the model, run the script.

```sh
python Data_Preprocessing_and_Feature_Engineering_Phase_1.py
```

### Example Output

The script will display the columns of the DataFrame after preprocessing and the accuracy of the Logistic Regression model.

## Week 5

### Titanic Dataset Preprocessing and Logistic Regression

The `Data_Preprocessing_and_Logistic_Regression.py` script preprocesses the Titanic dataset, performs feature engineering, and trains a Logistic Regression model for survival prediction.

#### Tasks and Algorithm Details

- **Data Cleaning**: Remove unnecessary columns and handle missing values.
- **Transformation**: Convert categorical variables (`Sex`, `Embarked`) into numerical form using one-hot encoding or label encoding.
- **Feature Engineering**: Create a new feature `FamilySize` based on `SibSp` and `Parch`.
- **Normalization/Scaling**: Standardize numerical features (`Age`, `Fare`) using `StandardScaler`.
- **Model Training**: Train a Logistic Regression model using scikit-learn to predict survival on the Titanic.

#### Usage

To preprocess the data and train the model, run the script.

```sh
python Data_Preprocessing_and_Logistic_Regression.py
```

## Week 6

### Data Preprocessing and Feature Engineering - Phase 2

The `Data_Preprocessing_and_feature_engineering-Phase-2.py` script continues the data preprocessing and feature engineering for the Titanic dataset.

#### Tasks and Algorithm Details

1. **Handling Missing Values**: Fill missing values in 'Age', 'Embarked', and 'Fare' using techniques like median imputation.
2. **Encoding Categorical Variables**: Convert 'Sex' and 'Embarked' into numerical form using one-hot encoding.
3. **Scaling Features**: Standardize 'Age' and 'Fare' using `StandardScaler`.
4. **Detecting Outliers Using Z-score**: Identify outliers in 'Age' and 'Fare' based on the Z-score method.
5. **Feature Engineering**: Create new features 'FamilySize', 'IsAlone', and 'Title' from existing features.
6. **Drop Unnecessary Columns**: Remove 'Cabin', 'Name', 'Ticket', and 'PassengerId'.
7. **Save the Cleaned and Engineered Dataset**: Save the processed dataset to a new CSV file.

#### Usage

To preprocess the data and save the cleaned dataset, run the script.

```sh
python Data_Preprocessing_and_feature_engineering-Phase-2.py
```

## Week 7

### House Price Prediction

The `House_Price_Prediction.py` script performs data preprocessing, feature engineering, model tuning, and prediction using the Kaggle House Prices dataset.

#### Tasks and Algorithm Details

1. **Load and Preprocess the Dataset**: Load `train.csv` and `test.csv`.
2. **Handle Missing Values**: Use median imputation for numerical features.
3. **Encode Categorical Variables**: Use Label Encoding for categorical features.
4. **Scale Numerical Features**: Standardize numerical features using `StandardScaler`.
5. **Detect and Handle Outliers**: (Optional) Remove outliers based on domain knowledge or statistical methods.
6. **Feature Engineering**: Create new features like `TotalSF` (Total Square Footage).
7. **Tune Random Forest Regressor Model**: Use `GridSearchCV` for hyperparameter tuning of the Random Forest Regressor.
8. **Evaluate the Model**: Use cross-validation to calculate RMSE (Root Mean Squared Error).
9. **Train the Best Model**: Train on the entire training dataset and make predictions on the test set.
10. **Generate Submission File**: Create a `submission.csv` file for the Kaggle competition.

#### Usage

To run the house price prediction script, use the following command:

```sh
python House_Price_Prediction.py
```

#### Example Output

- **RMSE Score**: Displays the RMSE score after cross-validation.
- **Submission File**: Generates a `submission.csv` file for the Kaggle competition.

## Week 8

### Loan Approval Prediction

The `Loan_Approval_Prediction.py` script performs data preprocessing, feature engineering, and model evaluation for the Loan Approval Prediction task using the Kaggle dataset.

#### Tasks and Algorithm Details

1. **Load and Preprocess the Dataset**: Load `Training Dataset.csv`.
2. **Handle Missing Values**: Impute missing values in categorical and numerical features.
3. **Encode Categorical Variables**: Use OneHotEncoding and LabelEncoding for categorical features.
4. **Scale Numerical Features**: Standardize numerical features using `StandardScaler`.
5. **Feature Engineering**: Create new features based on domain knowledge.
6. **Train and Evaluate Model**: Train a Logistic Regression model and evaluate its performance.
7. **Generate Classification Metrics**: Calculate accuracy, confusion matrix, and classification report.

#### Usage

To run the loan approval prediction script, use the following command:

```sh
python Loan_Approval_Prediction.py
```

#### Example Output

- **Accuracy**: Displays the accuracy of the model.
- **Confusion Matrix**: Shows the confusion matrix of the predictions.
- **Classification Report**: Provides precision, recall, and f1-score for each class.
