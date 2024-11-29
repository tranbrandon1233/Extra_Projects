import pandas as pd
import numpy as np

# Sample DataFrame of sales data with some inconsistencies
data = {
    'Product': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
    'Price': [10, 20, 15, 10, 20, 15, np.nan, 20],  # Missing Price for 'A'
    'Quantity': [100, 150, 200, 110, 160, 190, 100, 150],
    'Discount': [5, 0, 5, np.nan, 0, 5, 5, np.nan],  # Missing Discount for 'A' and 'B'
    'Date': ['2023-01-01', '2023/01/01', '01-02-2023', '2023/01/02', '2023-01/03', '2023-01-03', '2023-01-04', '04/01/2023'],
    'Product_Category': ['Electronics', 'Furniture', None, 'Electronics', 'Furniture', 'Toys', 'Electronics', None]
}

df = pd.DataFrame(data)

# Handle missing and inconsistent data

# Forward fill for Price and fill Discount with median, ensuring no NaN remains
df['Price'] = df['Price'].fillna(method='ffill')
df['Discount'] = df['Discount'].fillna(df['Discount'].median())

# Ensure uniform date format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Interpolating missing category data and default to 'Unknown'
df['Product_Category'] = df['Product_Category'].fillna('Unknown')

# Remove negative quantities, interpret as returns or errors
df = df[df['Quantity'] >= 0]

# Calculate Total Sales
df['Total_Sales'] = df['Price'] * df['Quantity']

# Apply Discounted Price
df['Discounted_Price'] = df['Price'] - df['Discount']
df['Discounted_Sales'] = df['Discounted_Price'] * df['Quantity']

# Calculate Profit Margin, ensuring Total_Sales is valid
df['Profit_Margin'] = df['Total_Sales'] - df['Discounted_Sales']
df['Profit_Margin_Percentage'] = np.where(df['Total_Sales'] > 0, (df['Profit_Margin'] / df['Total_Sales']) * 100, 0)

# Calculate Sales Growth with improved handling for zero or low values
df['Sales_Growth'] = df.groupby('Product')['Total_Sales'].pct_change().replace([np.inf, -np.inf, np.nan], 0) * 100

# Drop duplicates on Product and Date
df = df.drop_duplicates(subset=['Product', 'Date'])

# Aggregate data by Product and Date
agg_data = df.groupby(['Product', 'Date']).agg(
    Total_Sales=('Total_Sales', 'sum'),
    Discounted_Sales=('Discounted_Sales', 'sum'),
    Average_Price=('Price', 'mean'),
    Total_Quantity=('Quantity', 'sum'),
    Profit_Margin=('Profit_Margin', 'sum'),
    Profit_Margin_Percentage=('Profit_Margin_Percentage', 'mean'),
    Sales_Growth=('Sales_Growth', 'mean')  # Recommend reviewing strategy if consistent low sales
).reset_index()

# Weighted average calculation for price based on quantity sold
agg_data['Weighted_Avg_Price'] = agg_data.apply(
    lambda row: np.average(
        df[df['Product'] == row['Product']]['Price'], 
        weights=df[df['Product'] == row['Product']]['Quantity']
    ) if not df[df['Product'] == row['Product']].empty else np.nan,
    axis=1
)

print("Sales Data with Calculations:")
print(df)

print("\nAggregated Data (Product-wise and Date-wise):")
print(agg_data)

# Find the Product with the Highest Profit Margin Percentage
try:
    max_profit_margin_product = df.loc[df['Profit_Margin_Percentage'].idxmax()]
    print(f"\nProduct with the Highest Profit Margin Percentage: {max_profit_margin_product['Product']}")
except ValueError:
    print("No product found with a valid profit margin percentage.")

# Find the Product with the Highest Sales Growth
try:
    max_sales_growth_product = df.loc[df['Sales_Growth'].idxmax()]
    print(f"Product with the Highest Sales Growth: {max_sales_growth_product['Product']}")
except ValueError:
    print("No product found with valid sales growth.")

# Additional Aggregation by Product_Category if needed
category_agg = df.groupby('Product_Category').agg(
    Total_Sales=('Total_Sales', 'sum'),
    Average_Price=('Price', 'mean'),
    Total_Quantity=('Quantity', 'sum')
).reset_index()

print("\nCategory-wise Aggregation:")
print(category_agg)