import pandas as pd
import numpy as np
import os

# get the path to the current script's directory

file_path = '../data/swiggy.csv'

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: '{file_path}' not found. Please make sure the file exists at that location.")
    exit()

# Data Cleaning

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Handling Specific Columns

# Function to split city and area
def split_city_area(location):
    #Splits location string into city and area.
    if isinstance(location, str):
        if ',' in location:
            parts = [part.strip() for part in location.split(',')]
            # Assuming format "Area, City"
            if len(parts) == 2:
                return parts[1], parts[0]
            else:
                return parts[-1], ", ".join(parts[:-1])
        else:
            # If no comma, city and area are the same
            city = location.strip()
            return city, city
    return None, None # Return None for non-string or invalid inputs

# Apply the function to the 'city' column to create 'city' and 'area' columns
df[['city', 'area']] = df['city'].apply(lambda x: pd.Series(split_city_area(x)))


# Process 'rating_count' column
def process_rating_count(value):
    """Converts rating count strings to a numerical value."""
    if not isinstance(value, str):
        return value

    value_lower = value.lower().strip()

    if 'too few ratings' in value_lower:
        return np.nan

    # Clean the string from non-numeric parts like 'ratings' or 'rating'
    cleaned_str = value_lower.replace('ratings', '').replace('rating', '').strip()

    try:
        if 'k+' in cleaned_str:
            # Handle cases like '1k+' -> 1000
            num_str = cleaned_str.replace('k+', '').strip()
            return int(float(num_str) * 1000)
        elif '+' in cleaned_str:
            # Handle cases like '50+' -> 51
            num_str = cleaned_str.replace('+', '').strip()
            return int(num_str) + 1
        else:
            # Handle plain numbers like '50'
            return int(cleaned_str)
    except (ValueError, TypeError):
        # If any conversion fails after cleaning, return NaN
        return np.nan

df['rating_count'] = df['rating_count'].apply(process_rating_count)


# Data Type Conversion and Final Cleaning 


df['cost'] = df['cost'].astype(str).str.replace('₹', '').str.replace(',', '').str.strip()

# Convert columns to numeric, coercing errors to NaN
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')


# Drop rows with any remaining NaN values in critical columns
df.dropna(subset=['rating', 'rating_count', 'cost', 'city', 'area', 'cuisine'], inplace=True)


# Save the cleaned file back into the 'data' directory
output_path = '../data/cleaned_swiggy.csv'
try:
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data cleaning and preprocessing complete. The cleaned data is saved as '{output_path}'.")
except Exception as e:
    print(f"An error occurred while saving the file: {e}")

# Display the first few rows of the cleaned dataframe
print("\n--- First 5 rows of the cleaned data: ---")
print(df.head())
print("\n--- Data Info ---")
df.info()
