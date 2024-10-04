import pandas as pd

data = pd.read_json('data.json')

# Edit a value in the first row
data.loc[0, 'col1'] = 32

# Filter: keep rows where age > 30 and city is 'New York'
filtered_data = data[(data['col1'] > 2)]

# Removing a column
filtered_data_short = filtered_data.drop('col2', axis=1)

# Save the filtered and modified data to a new file
filtered_data_short.to_json('modified_data.json', orient='records', indent=2) 