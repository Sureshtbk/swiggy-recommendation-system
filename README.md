# Swiggy Restaurant Recommendation System

This project is a content-based recommendation system that suggests restaurants to users based on their preferences. The application is built with Python and Streamlit, providing an interactive and user-friendly interface to help users discover new places to eat.

## Features

- **Interactive UI**: A clean and intuitive web interface built with Streamlit.
- **Personalized Recommendations**: Get restaurant suggestions based on City, Area, Cuisine, Cost, and Rating.
- **Content-Based Filtering**: Uses cosine similarity to find and rank the most relevant restaurants based on a multi-feature profile.
- **Dynamic Filtering**: The "Area" dropdown dynamically updates based on the selected city for a seamless user experience.

## How It Works

The project follows a standard three-step machine learning workflow:

### 1. Data Cleaning (`src/datacleaning.py`)
The initial `swiggy.csv` dataset is thoroughly cleaned and preprocessed. This involves:
- Handling duplicates
- Normalizing location data into distinct city and area columns
- Converting `rating_count` (e.g., "1K+") and `cost` (e.g., "₹200") into numerical formats
- Handling missing values

The final clean dataset is saved as `data/cleaned_swiggy.csv`.

### 2. Feature Engineering (`src/feature_engineering.py`)
The cleaned data is transformed into a numerical format suitable for machine learning:
- Numerical features (`rating`, `cost`, `rating_count`) are scaled using `MinMaxScaler`.
- Categorical features (`city`, `area`, `cuisine`) are encoded using `OneHotEncoder`.

Outputs:
- `models/encoder.pkl` — Fitted preprocessor model
- `models/restaurant_data.pkl` — Cleaned restaurant data
- `data/encoded_data.csv` — Fully encoded dataset

### 3. Recommendation Application (`src/app.py`)
The Streamlit app loads the preprocessor and datasets, and performs the following:
- Filters the dataset to create a candidate pool based on selected City, Area, and Cuisine.
- Ranks candidates using cosine similarity based on full user preference (cost + rating).
- Displays the top 5 matches in a friendly interface.

## Tech Stack

- **Backend & Logic**: Python
- **Data Manipulation**: Pandas
- **Machine Learning**: Scikit-learn
- **Web Framework**: Streamlit
- **Version Control**: Git & GitHub

## Project Structure

```
.
├── data/
│   ├── swiggy.csv              # Raw dataset
│   ├── cleaned_swiggy.csv      # Cleaned dataset
│   ├── encoded_data.csv        # Feature engineered dataset
├── models/
│   ├── encoder.pkl             # Fitted preprocessor
│   ├── restaurant_data.pkl     # Data for app recommendations
├── src/
│   ├── datacleaning.py         # Cleans and processes raw data
│   ├── feature_engineering.py  # Feature encoding and scaler
│   ├── app.py                  # Streamlit app
├── .gitignore
├── README.md                   # This file
```

## How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Sureshtbk/swiggy-recommendation-system.git
cd swiggy-recommendation-system
```

### 2. Create a Virtual Environment and Install Dependencies

```bash
# Create a virtual environment
python -m venv venv

# Activate it (on Windows)
.\venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# Install required libraries
pip install pandas scikit-learn streamlit
```

### 3. Run Data Processing Scripts

```bash
python src/datacleaning.py
python src/feature_engineering.py
```

### 4. Launch the Streamlit App

```bash
streamlit run src/app.py
```

The application should now be running in your browser! Enjoy exploring restaurant recommendations.
