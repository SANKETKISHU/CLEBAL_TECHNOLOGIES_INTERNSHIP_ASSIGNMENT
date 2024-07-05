
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

