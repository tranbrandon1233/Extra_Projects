import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
df = pd.read_csv('diamonds.csv')

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['cut', 'color', 'clarity'])

# Separate features and target variable
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Initialize and train the Random Forest model
model = RandomForestRegressor(n_estimators=150, random_state=23)
model.fit(X_train, y_train)

# Make predictions on the testing set
predictions = model.predict(X_test)

# Create a DataFrame for the test set including predictions
df_test = X_test.copy()
df_test['Price_real'] = y_test.values
df_test['Price_predict'] = predictions

# Display the first 10 rows
print(df_test.head(10))