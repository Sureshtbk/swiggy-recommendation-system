import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os

# Build Paths and Load All Necessary Files 
try:
    # Define base directory to make paths reliable
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
    MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')

    # Paths to all required files
    PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'encoder.pkl')
    RESTAURANT_DATA_PATH = os.path.join(MODELS_DIR, 'restaurant_data.pkl')
    ENCODED_DATA_PATH = os.path.join(DATA_DIR, 'encoded_data.csv')

    # Load the files
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    
    restaurant_data = pd.read_pickle(RESTAURANT_DATA_PATH)
    encoded_data = pd.read_csv(ENCODED_DATA_PATH)
    
    
except FileNotFoundError as e:
    st.error(e.filename + " not found. Please ensure the data and model files are correctly set up.")
    st.info("Please run `datacleaning.py` and then the updated `feature_engineering.py` script.")
    st.stop()


# Restaurant Recommendation Function 
def get_recommendations(user_input_df, n_recommendations=5):
    """
    Filters restaurants by city, AREA, and cuisine, then ranks them by similarity.
    """
    # Extract all user preferences from the 
    selected_city = user_input_df['city'].values[0]
    selected_area = user_input_df['area'].values[0] 
    selected_cuisine = user_input_df['cuisine'].values[0]
    min_rating = user_input_df['rating'].values[0]

    # Filter the original data to find candidate restaurants.
    candidate_restaurants = restaurant_data[
        (restaurant_data['city'] == selected_city) &
        (restaurant_data['area'] == selected_area) & 
        (restaurant_data['cuisine'].str.contains(selected_cuisine, case=False)) &
        (restaurant_data['rating'] >= min_rating)
    ].copy()

    if candidate_restaurants.empty:
        return pd.DataFrame() # Return empty if no restaurants match the initial filter

    # Get the indices of these candidates. This is the key to alignment.
    candidate_indices = candidate_restaurants.index
    
    # Select the corresponding encoded vectors from our pre-loaded encoded_data.
    candidate_vectors = encoded_data.values[candidate_indices]

    # Encode the user's input using the loaded preprocessor.
    user_vector = preprocessor.transform(user_input_df)

    # Calculate cosine similarity between the user and all candidates.
    similarity_scores = cosine_similarity(user_vector, candidate_vectors)

    # Get the top N most similar candidates.
    # We get the indices of the top scores from the similarity matrix.
    top_candidate_indices = similarity_scores[0].argsort()[-n_recommendations:][::-1]
    
    # Map these indices back to the `candidate_restaurants` DataFrame to get the final recommendations.
   
    recommended_indices = candidate_restaurants.index[top_candidate_indices]
    final_recommendations = restaurant_data.loc[recommended_indices]

    return final_recommendations

# Streamlit UI part
st.set_page_config(layout="wide")
st.title("Swiggy’s Restaurant Recommendation System ")
st.markdown("Find top 5 restaurant based on Rating & Cost!")


col1, col2 = st.columns(2)

with col1:
    st.subheader("Location & Cuisine")
    # Get unique, sorted lists for dropdowns
    cities = sorted(restaurant_data['city'].unique())
    cuisines = sorted(restaurant_data['cuisine'].unique())
    
    selected_city = st.selectbox("Select a City:", cities)
    
    # Dynamically update areas based on selected city for a better UX
    areas = sorted(restaurant_data[restaurant_data['city'] == selected_city]['area'].unique())
    selected_area = st.selectbox("Select an Area:", areas)
    
    selected_cuisine = st.selectbox("Select a Cuisine:", cuisines)

with col2:
    st.subheader("Cost & Rating")
    # Define cost range based on the data for the selected city
    min_cost_city = int(restaurant_data[restaurant_data['city'] == selected_city]['cost'].min())
    max_cost_city = int(restaurant_data[restaurant_data['city'] == selected_city]['cost'].max())

    selected_cost = st.slider("Select your preferred cost for two:", min_value=min_cost_city, max_value=max_cost_city, value=min_cost_city, step=50)
    selected_rating = st.slider("Select the minimum preferred rating:", min_value=1.0, max_value=5.0, value=4.0, step=0.1)

# Find restaurants by button click
if st.button("Find Restaurants", use_container_width=True):
    user_input = pd.DataFrame({
        'rating': [selected_rating],
        'cost': [selected_cost],
        'rating_count': [0], 
        'city': [selected_city],
        'area': [selected_area],
        'cuisine': [selected_cuisine]
    })

    with st.spinner('Finding the best matches for you...'):
        recommendations = get_recommendations(user_input, n_recommendations=5)
#display top 5 restaurants
    if not recommendations.empty:
        st.success("Top 5 recommended restaurants!")
        for index, row in recommendations.iterrows():
            st.markdown(f"""
            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 15px; background-color: #f9f9f9;">
                <h4 style="color: #FC8019;">{row['name']}</h4>
                <p style="color: #333333;"><strong>Cuisine:</strong> {row['cuisine']}</p>
                <p style="color: #333333;"><strong>Location:</strong> {row['area']}, {row['city']}</p>
                <p style="color: #333333;"><strong>Rating:</strong> {row['rating']} ⭐ ({row['rating_count']} ratings)</p>
                <p style="color: #333333;"><strong>Cost for Two:</strong> ₹{row['cost']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Sorry, we couldn't find any restaurants that match your criteria. Please try a different combination!")
