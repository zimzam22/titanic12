import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

titanic = pd.read_csv("D:\\Assignment\\CPSC_4800\\titanic.csv")

titanic.info()

titanic['Pclass'] = titanic['Pclass'].astype('category')
titanic['Survived'] = titanic['Survived'].astype('category')
titanic['Embarked'] = titanic['Embarked'].astype('category')
titanic['Sex'] = titanic['Sex'].astype('category')

missing_values = titanic.isnull().sum()

# Display the missing values
print("Missing values in each column:")
print(missing_values)

# For Age, fill missing values with the mean
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].mean())

# For Cabin, fill missing values with the mode (most frequent value)
titanic['Cabin'] = titanic['Cabin'].fillna(titanic['Cabin'].mode()[0])

# For Embarked, fill missing values with the mode (most frequent value)
titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])

missing_values = titanic.isnull().sum()

# Display the missing values
print("Missing values in each column:")
print(missing_values)

# Summary statistics for numerical columns
summary = titanic.describe()

# Display the summary
print("Summary Statistics for Numerical Columns:")
print(summary)

# Include both numeric and categorical columns in the summary
summary = titanic.describe()

# Print the summary
print(summary)


sex_summary = titanic['Sex'].value_counts()
survived_summary = titanic['Survived'].value_counts()
pclass_summary = titanic['Pclass'].value_counts()
embarked_summary = titanic['Embarked'].value_counts()
# Print the summary
print(sex_summary)
print(survived_summary)
print(pclass_summary)
print(embarked_summary)

# Create the 'Age Group' column if not already created
age_bins = [0, 12, 18, 30, 40, 50, 60, 100]
age_labels = ['0-12', '13-18', '19-30', '31-40', '41-50', '51-60', '60+']
titanic['Age Group'] = pd.cut(titanic['Age'], bins=age_bins, labels=age_labels, right=False)

# Calculate the number of people in each age group
age_group_counts = titanic['Age Group'].value_counts().sort_index()

# Display the result
print(age_group_counts)


# Ensure 'Survived' is of numeric type (in case it's not)
titanic['Survived'] = titanic['Survived'].astype(float)

# 1. Calculate Survival Rate Based on Pclass
# Ensure 'Pclass' is treated as a categorical or object type
titanic['Pclass'] = titanic['Pclass'].astype(str)
survival_rate_by_class = titanic.groupby('Pclass', observed=False)['Survived'].mean().reset_index()
survival_rate_by_class.rename(columns={'Survived': 'survival_rate'}, inplace=True)

# 2. Calculate Survival Rate Based on Gender (Sex)
# Ensure 'Sex' is treated as a categorical or object type
survival_rate_by_gender = titanic.groupby('Sex', observed=False)['Survived'].mean().reset_index()
survival_rate_by_gender.rename(columns={'Survived': 'survival_rate'}, inplace=True)

# 3. Calculate Survival Rate Based on Age (using bins for age groups)
# Define bins for age groups (you can adjust these bins as needed)
age_bins = [0, 12, 18, 30, 40, 50, 60, 100]
age_labels = ['0-12', '13-18', '19-30', '31-40', '41-50', '51-60', '60+']

# Create an 'Age Group' column
titanic['Age Group'] = pd.cut(titanic['Age'], bins=age_bins, labels=age_labels, right=False)

# Calculate survival rate based on the age group
survival_rate_by_age = titanic.groupby('Age Group', observed=False)['Survived'].mean().reset_index()
survival_rate_by_age.rename(columns={'Survived': 'survival_rate'}, inplace=True)

# Display the results
print("Survival Rate by Pclass:")
print(survival_rate_by_class)

print("\nSurvival Rate by Gender (Sex):")
print(survival_rate_by_gender)

print("\nSurvival Rate by Age Group:")
print(survival_rate_by_age)


# Set plot style
sns.set(style="whitegrid")

# 1. Plot the number of people in each age group
# Calculate the count of people in each Age Group and Sex, with observed=False
age_group_counts_gender = titanic.groupby(['Age Group', 'Sex'], observed=False).size().reset_index(name='Count')

# Plot the number of people in each Age Group by Gender
plt.figure(figsize=(10, 6))
sns.barplot(x='Age Group', y='Count', hue='Sex', data=age_group_counts_gender, palette='coolwarm')

# Add title and labels
plt.title('Number of People in Each Age Group by Gender')
plt.xlabel('Age Group')
plt.ylabel('Number of People')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()

# 2. Plot the survival rate based on Pclass
# Calculate survival rate based on both Pclass and Gender
survival_rate_by_class_gender = titanic.groupby(['Pclass', 'Sex'], observed=False)['Survived'].mean().reset_index()

# Plot the survival rate based on Pclass and Gender
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=survival_rate_by_class_gender, palette='Blues_d')

# Add title and labels
plt.title('Survival Rate Based on Pclass and Gender')
plt.xlabel('Pclass')
plt.ylabel('Survival Rate')
plt.show()

# 3. Plot the survival rate based on Sex
survival_rate_by_gender = titanic.groupby('Sex', observed=False)['Survived'].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(x='Sex', y='Survived', data=survival_rate_by_gender, hue='Sex', palette='Set2', legend=False)
plt.title('Survival Rate Based on Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.show()

# 4. Plot the survival rate based on Age Group
# Calculate survival rate based on Age Group and Sex
survival_rate_by_age_gender = titanic.groupby(['Age Group', 'Sex'], observed=False)['Survived'].mean().reset_index()

# Plot the survival rate based on Age Group and Sex
plt.figure(figsize=(10, 6))
sns.barplot(x='Age Group', y='Survived', data=survival_rate_by_age_gender, hue='Sex', palette='coolwarm')

# Add title and labels
plt.title('Survival Rate Based on Age Group and Gender')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()



# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = titanic['Age'].quantile(0.25)
Q3 = titanic['Age'].quantile(0.75)

# Calculate IQR (Interquartile Range)
IQR = Q3 - Q1

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = titanic[(titanic['Age'] < lower_bound) | (titanic['Age'] > upper_bound)]

# Print the number of outliers and the outlier values
print(f"Number of outliers: {outliers.shape[0]}")
print(outliers[['Age']])

# Optionally, you can visualize the distribution of Age and highlight outliers
plt.figure(figsize=(8, 5))
sns.boxplot(x=titanic['Age'])
plt.title('Boxplot of Age with Outliers')
plt.show()

# Create a boxplot for the 'Fare' column
plt.figure(figsize=(8, 6))
sns.boxplot(x=titanic['Fare'])

# Add a title and labels
plt.title('Boxplot of Fare')
plt.xlabel('Fare')

# Show the plot
plt.show()


# Calculate the average age in each Pclass based on Gender, explicitly setting observed=True
avg_age_by_class_gender = titanic.groupby(['Pclass', 'Sex'], observed=True)['Age'].mean().reset_index()

# Plot the average age in each Pclass based on Gender
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Age', hue='Sex', data=avg_age_by_class_gender, palette='coolwarm')

# Add title and labels
plt.title('Average Age in Each Pclass Based on Gender')
plt.xlabel('Pclass')
plt.ylabel('Average Age')
plt.show()


# Convert 'Sex' column to numeric values for correlation
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})  # 'male' becomes 0 and 'female' becomes 1

# Select relevant columns for correlation
columns_for_corr = ['Survived', 'Pclass', 'Sex', 'Age']

# Compute the correlation matrix
corr_matrix = titanic[columns_for_corr].corr()

# Plot the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Plot scatter plots for numerical columns
sns.pairplot(titanic[columns_for_corr])
plt.suptitle("Scatter Plots of Selected Variables", y=1.02)
plt.show()