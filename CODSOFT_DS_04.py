#!/usr/bin/env python
# coding: utf-8

# # TASK 4

# ### AIM
# 
# To predict sales based on advertising expenditure across different media channels (TV, Radio, Newspaper).

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from IPython.display import display
from tabulate import tabulate


# In[2]:


# Load the dataset
file_path = r"C:\Users\rosha\Downloads\advertising.csv"
data = pd.read_csv(file_path)


# In[3]:


# Display the first few rows of the dataset in a tabular format
print("First few rows of the dataset:")
print(tabulate(data.head(), headers='keys', tablefmt='psql'))


# In[4]:


# Display basic information about the dataset
print("\nBasic information about the dataset:")
print(data.info())


# In[5]:


# Display summary statistics
print("\nSummary statistics of the dataset:")
print(data.describe())


# In[6]:


# Check for missing values
print("\nChecking for missing values:")
print(data.isnull().sum())


# In[7]:


import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn.axisgrid")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Visualize the relationships between features and sales
print("\nVisualizing data relationships:")
sns.pairplot(data)
plt.show()


# In[8]:


# Heatmap of correlations
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[9]:


# Distribution of each feature
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.histplot(data['TV'], bins=30, kde=True, ax=axs[0])
axs[0].set_title('Distribution of TV Advertising Spend')
sns.histplot(data['Radio'], bins=30, kde=True, ax=axs[1])
axs[1].set_title('Distribution of Radio Advertising Spend')
sns.histplot(data['Newspaper'], bins=30, kde=True, ax=axs[2])
axs[2].set_title('Distribution of Newspaper Advertising Spend')
plt.show()


# In[10]:


# Box plots to identify outliers
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(y=data['TV'], ax=axs[0])
axs[0].set_title('Box Plot of TV Advertising Spend')
sns.boxplot(y=data['Radio'], ax=axs[1])
axs[1].set_title('Box Plot of Radio Advertising Spend')
sns.boxplot(y=data['Newspaper'], ax=axs[2])
axs[2].set_title('Box Plot of Newspaper Advertising Spend')
plt.show()


# In[11]:


# Pairwise relationships with regression lines
sns.pairplot(data, kind='reg', plot_kws={'line_kws':{'color':'red'}})
plt.show()


# In[12]:


# Define features and target variable
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']


# In[13]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[15]:


# Make predictions
y_pred = model.predict(X_test)


# In[16]:


# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[17]:


# Print the performance metrics
print(f"\nMean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[18]:


# Visualize the actual vs predicted sales
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()


# ### Conclusion
# 
# The linear regression model, based on advertising expenditures, provides a reasonable prediction of sales, with TV and Radio being more significant predictors compared to Newspaper. The visualization and metrics suggest that while the model captures the general trend, there is room for improvement, potentially by exploring more complex models or additional features.
# 
# This analysis enables businesses to make informed decisions regarding their advertising budget allocation to maximize sales potential.
