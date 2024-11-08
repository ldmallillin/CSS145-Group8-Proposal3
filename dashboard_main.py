#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#######################
# Page configuration
st.set_page_config(
    page_title="Dashboard Template", # Replace this with your Project's Title
    page_icon="☕", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('Dashboard Template')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. BUNAG, Annika\n2. CHUA, Denrick Ronn\n3. MALLILLIN, Loragene\n4. SIBAYAN, Gian Eugene\n5. UMALI, Ralph Dwayne")

#######################
# Data

# Load data
dataset = pd.read_csv("data/coffee_analysis.csv")
df_initial = dataset

#######################

# Pages

###################################################################
# About Page ######################################################
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    # Your content for the ABOUT page goes here

###################################################################
# Dataset Page ####################################################
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.write("Coffee Reviews Dataset by `@schmoyote`")
    st.write("")

    # Your content for your DATASET page goes here

###################################################################
# Data Cleaning Page ##############################################
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    # Display the initial dataset
    st.subheader("Initial Dataset", divider=True)
    st.write(df_initial.head())

    # Making a copy before pre-processing
    df = df_initial.copy()

    ##### Removing nulls
    st.subheader("Removing nulls, duplicates, etc.", divider=True)
    missing_values = round((df.isnull().sum()/df.shape[0])*100, 2)
    st.write(missing_values)

    # Drop NA
    df = df.dropna()
    st.write("Missing values after dropping NA:")
    st.write(round((df.isnull().sum()/df.shape[0])*100, 2))

    # Check for duplicates
    duplicate_rows = df[df.duplicated()]
    num_duplicate_rows = len(duplicate_rows)
    st.write(f"Number of duplicate rows: {num_duplicate_rows}")

    ##### Removing Outliers
    st.subheader("Removing Outliers", divider=True)
    plt.figure(figsize=(10, 4))
    plt.boxplot(df['100g_USD'], vert=False)
    plt.ylabel('Price Column (100g_USD)')
    plt.xlabel('Price Values')
    plt.title('Distribution of Coffee Price per 100g (USD)')
    st.pyplot(plt)

    Q1 = df['100g_USD'].quantile(0.25)
    Q3 = df['100g_USD'].quantile(0.75)
    IQR = Q3 - Q1
    price_lower_bound = max(0, Q1 - 1.5 * IQR)
    price_upper_bound = Q3 + 1.5 * IQR

    
    st.write('Lower Bound:', price_lower_bound)
    st.write('Upper Bound:', price_upper_bound)

    # Filter the dataset to remove outliers
    df = df[(df['100g_USD'] >= price_lower_bound) & (df['100g_USD'] <= price_upper_bound)]

    # Display price statistics
    st.markdown("**Price Statistics after Outlier Removal:**")
    st.write(df['100g_USD'].describe())

    # Rating Outliers
    st.subheader("Rating Outliers")
    plt.figure(figsize=(10, 4))
    plt.boxplot(df['rating'], vert=False)
    plt.ylabel('Rating Column')
    plt.xlabel('Rating Values')
    plt.title('Distribution of Coffee Ratings')
    st.pyplot(plt)

    Q1 = df['rating'].quantile(0.25)
    Q3 = df['rating'].quantile(0.75)
    IQR = Q3 - Q1
    rating_lower_bound = max(0, Q1 - 1.5 * IQR)
    rating_upper_bound = Q3 + 1.5 * IQR

    st.write('Lower Bound:', rating_lower_bound)
    st.write('Upper Bound:', rating_upper_bound)

    # Filter the dataset to remove outliers
    df = df[(df['rating'] >= rating_lower_bound) & (df['rating'] <= rating_upper_bound)]

    # Display rating statistics
    st.markdown("**Rating Statistics after Outlier Removal:**")
    st.write(df['rating'].describe())

    ##### Text pre-processing
    st.subheader("Coffee review text pre-processing", divider=True)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    st.markdown("`df[['desc_1', 'desc_2', 'desc_3']].head()`")
    st.write(df[['desc_1', 'desc_2', 'desc_3']].head())

    # Function to preprocess a single description column
    def preprocess_text(text):
        # 1. Lowercase
        text = text.lower()

        # 2. Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # 3. Tokenize
        tokens = word_tokenize(text)

        # 4. Remove stopwords
        tokens = [word for word in tokens if word not in stop_words]

        # 5. Lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        return ' '.join(tokens)

    # Apply the preprocessing
    df['desc_1_processed'] = df['desc_1'].apply(preprocess_text)
    df['desc_2_processed'] = df['desc_2'].apply(preprocess_text)
    df['desc_3_processed'] = df['desc_3'].apply(preprocess_text)

    # Display the processed DataFrame
    st.write(df[['desc_1_processed', 'desc_1', 'desc_2_processed', 'desc_2', 'desc_3_processed', 'desc_3']].head())

    ##### Encoding object columns
    st.subheader("Encoding object columns", divider=True)

    # Display DataFrame info
    info_buffer = st.empty()  

    # Create a summary DataFrame for display
    info_data = {
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum(),
        "Dtype": df.dtypes
    }

    info_df = pd.DataFrame(info_data)

    st.dataframe(info_df)

    # Initialize the LabelEncoder
    encoder = LabelEncoder()

    # Encode object columns
    df['name_encoded'] = encoder.fit_transform(df['name'])
    df['roaster_encoded'] = encoder.fit_transform(df['roaster'])
    df['roast_encoded'] = encoder.fit_transform(df['roast'])
    df['loc_country_encoded'] = encoder.fit_transform(df['loc_country'])
    df['origin_1_encoded'] = encoder.fit_transform(df['origin_1'])
    df['origin_2_encoded'] = encoder.fit_transform(df['origin_2'])

    # Display encoded columns
    st.subheader("Encoded Columns Preview")
    st.write(df[['name', 'name_encoded', 'roast', 'roast_encoded', 'roaster', 'roaster_encoded']].head())
    st.write(df[['loc_country', 'loc_country_encoded', 'origin_1', 'origin_1_encoded', 'origin_2', 'origin_2_encoded']].head())

    # Create and display summary mapping DataFrames
    def display_summary_mapping(column_name, encoded_column_name):
        unique_summary = df[column_name].unique()
        unique_summary_encoded = df[encoded_column_name].unique()
        summary_mapping_df = pd.DataFrame({column_name: unique_summary, f'{column_name}_encoded': unique_summary_encoded})
        st.write(f"{column_name} Summary Mapping")
        st.dataframe(summary_mapping_df)

    # Display mappings for each encoded column
    display_summary_mapping('name', 'name_encoded')
    display_summary_mapping('roast', 'roast_encoded')
    display_summary_mapping('roaster', 'roaster_encoded')
    display_summary_mapping('loc_country', 'loc_country_encoded')
    display_summary_mapping('origin_1', 'origin_1_encoded')
    display_summary_mapping('origin_2', 'origin_2_encoded')

###################################################################
# EDA Page ########################################################
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")


    col = st.columns((1.5, 4.5, 2), gap='medium')

    # Your content for the EDA page goes here

    with col[0]:
        st.markdown('#### Graphs Column 1')


    with col[1]:
        st.markdown('#### Graphs Column 2')
        
    with col[2]:
        st.markdown('#### Graphs Column 3')

###################################################################
# Machine Learning Page ###########################################
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here

###################################################################
# Prediction Page #################################################
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here

###################################################################
# Conclusions Page ################################################
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here