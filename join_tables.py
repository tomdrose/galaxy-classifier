import pandas as pd

csv1_path = 'gz2_filename_mapping.csv'
csv2_path = 'GalaxyZoo1_DR_table2.csv'

# Load CSV files
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

#I don't want 'uncertain' galaxies for training
df2 = df2[df2[df2.columns[15]] == 0]

# Get the first column name from each file (assumes header exists)
key1 = df1.columns[0]
key2 = df2.columns[0]

# Perform inner join on the first column of each table
merged_df = pd.merge(df1, df2, left_on=key1, right_on=key2, how='inner')

#drop duplicate column
merged_df.drop(columns=[key2], inplace=True)

# Save the result to a new CSV
merged_df.to_csv('joined_galaxies_table.csv', index=False)

print("Join complete. Output saved to 'joined_output.csv'")
