# Import the necessary libraries
import pandas as pd
import pickle
import os

# scikit-learn is the main library for machine learning in Python
# We need these specific tools from it:
#  OneHotEncoder: To turn text categories (like 'Chennai') into numbers for the model.
# MinMaxScaler: To scale all number columns to be between 0 and 1.
# ColumnTransformer: A tool to apply different preparations to different columns.
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

# Load our Cleaned Data 
print("Starting the feature engineering process...")

try:
    # We build the path to the file. '../' means 'go up one directory'.
    # So from 'src', we go up to the main 'Swiggy' folder, then into 'data'.
    file_path = '../data/cleaned_swiggy.csv'
    df = pd.read_csv(file_path)
    print("Successfully loaded 'cleaned_swiggy.csv'.")
except FileNotFoundError:
    print("Error: 'cleaned_swiggy.csv' not found. Please run 'datacleaning.py' first.")
    exit()


# We tell our model which columns are numbers and which are text categories.
# The model will treat them differently.
numerical_features = ['rating', 'cost', 'rating_count']
categorical_features = ['city', 'area', 'cuisine']
all_model_features = numerical_features + categorical_features


# We create a "recipe" that tells the model exactly how to prepare our data.
# For numerical columns: We scale them to be between 0 and 1 using MinMaxScaler.
# For categorical columns: We convert them into numbers using OneHotEncoder.
# The 'handle_unknown='ignore'' part tells the encoder to not throw an error
# if it sees a new category in the future (e.g., a new city).
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop' # We drop any columns we didn't explicitly mention.
)

# "Fitting" means the preprocessor learns from our data. It learns all the unique
# cities, areas, and cuisines for one-hot encoding.
print("Fitting the preprocessor to the data...")
preprocessor.fit(df[all_model_features])
print("Preprocessor has been fitted successfully.")

# "Transforming" means we apply the learned recipe to our data.
print("Transforming the data into its numerical (encoded) format...")
encoded_matrix = preprocessor.transform(df[all_model_features])
print("Data has been transformed.")


# This new step creates the encoded_data.csv file you requested.
# We convert the sparse matrix from the transform step into a full DataFrame.
print("Creating the encoded data DataFrame...")
# Get the new column names from the preprocessor
encoded_column_names = preprocessor.get_feature_names_out()
encoded_df = pd.DataFrame(encoded_matrix.toarray(), columns=encoded_column_names)

# Define the output path for the new CSV file
output_path_encoded = '../data/encoded_data.csv'
try:
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path_encoded), exist_ok=True)
    encoded_df.to_csv(output_path_encoded, index=False)
    print(f"NEW: Encoded data saved to '{output_path_encoded}'.")
except Exception as e:
    print(f"Error saving encoded data CSV: {e}")


# We still save the preprocessor (for encoding user input in the app)
# and the original cleaned data (for displaying results).

# Create the 'models' directory if it doesn't already exist.
os.makedirs('../models', exist_ok=True)


# This is essential for the app to encode the user's selections.
with open('../models/encoder.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
print("Preprocessor model saved to '../models/encoder.pkl'.")


# The app needs this to filter by and show the final recommendations.
df.to_pickle('../models/restaurant_data.pkl')
print("Complete restaurant data saved to '../models/restaurant_data.pkl'.")


print("\n--- All Done! ---")
print("The 'models' folder has been updated, and 'encoded_data.csv' has been created in the 'data' folder.")
