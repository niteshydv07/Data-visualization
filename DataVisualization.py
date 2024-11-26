# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris = pd.read_csv(url)

# Check if the dataset is loaded correctly
print("Dataset loaded successfully")
print(iris.head())

# Data Overview
print("\nData Overview:\n")
iris_info = iris.info()
print(iris_info)

print("\nSummary Statistics:\n")
iris_describe = iris.describe()
print(iris_describe)

# Pairplot
plt.figure(figsize=(10, 6))
sns.pairplot(iris, hue='species', markers=["o", "s", "D"])
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()

# Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(data=iris, x='species', y='sepal_length', ci=None)
plt.title('Average Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Sepal Length')
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris, x='sepal_length', y='petal_length', hue='species', style='species', palette='deep', s=100)
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.show()

# Heatmap
plt.figure(figsize=(10, 6))
correlation_matrix = iris.drop(columns=['species']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Iris Dataset')
plt.show()

# Insights
print("\nInsights based on the visualizations:\n")
print("1. Pairplot: The pairplot shows clear distinctions between the species in terms of their sepal and petal measurements. Setosa species have distinctively smaller petal lengths and widths.")
print("2. Bar Chart: The bar chart indicates that the average sepal length varies across species. Specifically, the Virginica species has the highest average sepal length.")
print("3. Scatter Plot: The scatter plot shows a positive correlation between sepal length and petal length across all species, with distinct clustering for each species.")
print("4. Heatmap: The heatmap reveals strong positive correlations between petal length and petal width, as well as between sepal length and petal length. Sepal width is less correlated with the other features.")
