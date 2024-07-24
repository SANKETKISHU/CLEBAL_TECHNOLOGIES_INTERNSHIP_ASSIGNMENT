
# Internship Assignments at Clebal Technologies

This repository contains Python scripts for various tasks and visualizations completed during my internship at Clebal Technologies.

## Week 1

### Lower and Upper Triangular Pyramids

The `Lower_Upper_Triangular_Pyramids.py` script generates and displays lower and upper triangular matrices.

#### Usage

To generate the pyramids, run the script and follow the prompts.

```sh
python Lower_Upper_Triangular_Pyramids.py
```

## Week 2

### Simple Calculator

The `Calculator_using_Python.py` script provides a simple calculator that can perform the following operations:
- Addition
- Subtraction
- Multiplication
- Division

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
- **Pairplot**: Visualize the relationships between features.
- **Box plots**: Show the distribution of each feature by species.
- **Heatmap**: Display the correlation between features.

#### Usage

To generate the visualizations, run the script and the plots will be saved as PNG files in your working directory.

```sh
python Iris_Data_Visualization.py
```

### Example Output

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

#### Tasks

- Handle missing values
- Normalize numerical features
- Encode categorical features
- Create new features
- Train a Logistic Regression model

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

#### Tasks

- Data Cleaning: Remove unnecessary columns and handle missing values.
- Transformation: Convert categorical variables (`Sex`, `Embarked`) into numerical form.
- Feature Engineering: Create a new feature `FamilySize` based on `SibSp` and `Parch`.
- Normalization/Scaling: Standardize numerical features (`Age`, `Fare`).
- Model Training: Train a Logistic Regression model to predict survival on the Titanic.

#### Usage

To preprocess the data and train the model, run the script.

```sh
python Data_Preprocessing_and_Logistic_Regression.py
```

## Week 6

### Data Preprocessing and Feature Engineering - Phase 2

The `Data_Preprocessing_and_feature_engineering-Phase-2.py` script continues the data preprocessing and feature engineering for the Titanic dataset.

#### Tasks

1. Handling missing values: Fill missing values in 'Age', 'Embarked', and 'Fare'.
2. Encoding categorical variables: Convert 'Sex' and 'Embarked' into numerical form.
3. Scaling features: Standardize 'Age' and 'Fare' using StandardScaler.
4. Detecting outliers using Z-score: Identify outliers in 'Age' and 'Fare'.
5. Feature engineering: Create new features 'FamilySize', 'IsAlone', and 'Title' from existing features.
6. Drop unnecessary columns: Remove 'Cabin', 'Name', 'Ticket', and 'PassengerId'.
7. Save the cleaned and engineered dataset to a new CSV file.

#### Usage

To preprocess the data and save the cleaned dataset, run the script.

```sh
python Data_Preprocessing_and_feature_engineering-Phase-2.py
```
## Week 7

### House Price Prediction

The `House_Price_Prediction.py` script performs data preprocessing, feature engineering, model tuning, and prediction using the Kaggle House Prices dataset.

#### Tasks

1. Load and preprocess the dataset (`train.csv`, `test.csv`).
2. Handle missing values using median imputation for numerical features.
3. Encode categorical variables using Label Encoding.
4. Scale numerical features using StandardScaler.
5. Detect and handle outliers (optional).
6. Perform feature engineering (e.g., creating new features like `TotalSF`).
7. Tune a Random Forest Regressor model using GridSearchCV.
8. Evaluate the model using cross-validation and calculate RMSE.
9. Train the best model and make predictions on the test set.
10. Generate a submission file (`submission.csv`) for the Kaggle competition.

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

The Loan_Approval_Prediction.py script performs data preprocessing, feature engineering, and model evaluation for the Loan Approval Prediction task using the Kaggle dataset.

#### Tasks

1. *Load and preprocess the dataset* (Training Dataset.csv).
2. *Handle missing values* in categorical and numerical features.
3. *Encode categorical variables* using OneHotEncoding and LabelEncoding.
4. *Scale numerical features* using StandardScaler.
5. *Feature engineering* to create new features.
6. *Train and evaluate* a Logistic Regression model.
7. *Generate classification metrics* including accuracy, confusion matrix, and classification report.

#### Usage

To run the loan approval prediction script, use the following command:

```sh
python Loan_Approval_Prediction.py
```
## Example Output
*Accuracy:* Displays the accuracy of the model.

*Confusion Matrix:* Shows the confusion matrix of the predictions.

*Classification Report:* Provides precision, recall, and f1-score for each class.
