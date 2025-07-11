# supermart_grocery_sales
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Step 2: Load Dataset
file_path = 'supermart_grocery_sales.csv'  # Replace with your dataset path
sales_df = pd.read_csv(file_path)

# Step 3: Initial Data Exploration
print("Data Shape:", sales_df.shape)
print("\nData Types:\n", sales_df.dtypes)
print("\nMissing Values:\n", sales_df.isnull().sum())
print("\nFirst 5 Records:\n", sales_df.head())

# Step 4: Data Cleaning
sales_df.drop_duplicates(inplace=True)
sales_df.dropna(inplace=True)

# Convert 'Order Date' to datetime
sales_df['Order Date'] = pd.to_datetime(sales_df['Order Date'], errors='coerce')

# Extract new date features
sales_df['Order Day'] = sales_df['Order Date'].dt.day
sales_df['Order Month'] = sales_df['Order Date'].dt.month
sales_df['Order Year'] = sales_df['Order Date'].dt.year

# Step 5: Label Encoding Categorical Columns
le = LabelEncoder()
for col in ['Category', 'Sub Category', 'City', 'Region', 'State', 'Month']:
    if col in sales_df.columns:
        sales_df[col] = le.fit_transform(sales_df[col].astype(str))

# Step 6: SQL Integration
conn = sqlite3.connect('supermart_sales.db')
sales_df.to_sql('supermart_sales', conn, if_exists='replace', index=False)

# Unique SQL Query: Top 3 Categories by Profit
query = """
SELECT Category, SUM(Profit) as Total_Profit
FROM supermart_sales
GROUP BY Category
ORDER BY Total_Profit DESC
LIMIT 3;
"""
top_categories_profit = pd.read_sql(query, conn)
print("\nTop 3 Categories by Profit:\n", top_categories_profit)

# Unique SQL Query: Yearly Sales Summary
query_yearly = """
SELECT Order_Year, SUM(Sales) as Yearly_Sales
FROM supermart_sales
GROUP BY Order_Year
ORDER BY Order_Year;
"""
yearly_sales_sql = pd.read_sql(query_yearly, conn)
print("\nYearly Sales from SQL:\n", yearly_sales_sql)

# Step 7: Exploratory Data Analysis (EDA)
# Sales by Category
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Sales', data=sales_df, ci=None, palette='coolwarm')
plt.title('Sales by Category')
plt.xlabel('Category')
plt.ylabel('Sales')
plt.show()

# Monthly Sales Trend
plt.figure(figsize=(12, 6))
monthly_trend = sales_df.groupby('Order Month')['Sales'].sum().reset_index()
sns.lineplot(x='Order Month', y='Sales', data=monthly_trend, marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.grid()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 6))
corr_matrix = sales_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')
plt.title('Correlation Heatmap')
plt.show()

# Step 8: Feature Selection and Model Preparation
X = sales_df.drop(columns=['Order ID', 'Customer Name', 'Order Date', 'Sales'])
y = sales_df['Sales']

# Data Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: Model Training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Step 10: Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:\nMean Squared Error: {mse:.2f}\nR-Squared: {r2:.4f}")

# Step 11: Visualization - Actual vs Predicted Sales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.title('Actual vs Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.show()

# Step 12: Export Cleaned Data to Excel
export_path = 'Supermart_Grocery_Sales_Cleaned_Unique.xlsx'
sales_df.to_excel(export_path, index=False)
print(f"\nCleaned dataset exported successfully to: {export_path}")
