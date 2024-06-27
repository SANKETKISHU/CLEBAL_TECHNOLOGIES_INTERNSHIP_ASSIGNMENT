import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(iris.head())

# Display summary statistics of the dataset
print("\nSummary statistics of the dataset:")
print(iris.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(iris.isnull().sum())

# Display the distribution of the species
print("\nDistribution of species:")
print(iris['species'].value_counts())

# Plot histograms for each feature
iris.hist(bins=20, figsize=(10, 10))
plt.suptitle('Histograms of Iris Features', fontsize=16)
plt.savefig('iris_histograms.png')
plt.show()

# Pairplot to visualize the relationships between features
sns.pairplot(iris, hue='species')
plt.suptitle('Pairplot of Iris Features', y=1.02, fontsize=16)
plt.savefig('iris_pairplot.png')
plt.show()

# Box plots to visualize the distribution of each feature by species
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='sepal_length', data=iris)
plt.title('Sepal Length by Species')
plt.savefig('sepal_length_boxplot.png')
plt.show()

sns.boxplot(x='species', y='sepal_width', data=iris)
plt.title('Sepal Width by Species')
plt.savefig('sepal_width_boxplot.png')
plt.show()

sns.boxplot(x='species', y='petal_length', data=iris)
plt.title('Petal Length by Species')
plt.savefig('petal_length_boxplot.png')
plt.show()

sns.boxplot(x='species', y='petal_width', data=iris)
plt.title('Petal Width by Species')
plt.savefig('petal_width_boxplot.png')
plt.show()

# Heatmap to show the correlation between features
plt.figure(figsize=(8, 6))
sns.heatmap(iris.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('iris_heatmap.png')
plt.show()
