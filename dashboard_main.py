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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')
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

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"
        
    if st.button("Coffee Price Prediction", use_container_width=True, on_click=set_page_selection, args=('coffee_price_prediction',)):
        st.session_state.page_selection = "coffee_price_prediction"
    
    if st.button("Description to Rating", use_container_width=True, on_click=set_page_selection, args=('description_to_rating',)):
        st.session_state.page_selection = "description_to_rating"
    
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

    # Combining all the preprocessing techniques in one function
    def preprocess_text(text):
        # 1. Lowercase
        text = text.lower()

        # 2. Remove punctuations
        text = text.translate(str.maketrans('', '', string.punctuation))

        # 3. Tokenize
        tokens = word_tokenize(text)

        # 4. Stopword Removal
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        # 5. Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        return tokens

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

    # Store the cleaned DataFrame in session state
    st.session_state.df = df

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

    # Check if the cleaned DataFrame exists in session state
    if 'df' in st.session_state:
        df = st.session_state.df

        col = st.columns((2, 3, 3), gap='medium')

        def create_pie_chart(df, names_col, values_col, width, height, key, title):
            # Generate a pie chart
            fig = px.pie(
                df,
                names=names_col,
                values=values_col,
                title=title
            )
            # Adjust the layout for the pie chart
            fig.update_layout(
                width=width,  
                height=height,  
                showlegend=True
            )
            # Display the pie chart in Streamlit
            st.plotly_chart(fig, use_container_width=True, key=f"pie_chart_{key}")

        with col[0]:
            with st.expander('Legend', expanded=True):
                st.write('''
                    - Data: [Coffee Reviews Dataset](https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset).
                    - :orange[**Pie Chart**]: TBA.
                    ''')

            ##### Roast types chart #####
            roast_counts = df['roast'].value_counts().reset_index()
            roast_counts.columns = ['Roast', 'Count']
            create_pie_chart(roast_counts, 'Roast', 'Count', width=400, height=400, key='coffee_roast', title='Distribution of Roast Types')     

        with col[1]:
            ##### Roasters pie chart #####
            top_n = 20

            roaster_counts = df['roaster'].value_counts()
            top_roasters = roaster_counts.nlargest(top_n)
            other_roasters = roaster_counts.iloc[top_n:].sum()

            roaster_counts_top = pd.concat([top_roasters, pd.Series({'Other': other_roasters})]).reset_index()
            roaster_counts_top.columns = ['Roaster', 'Count']

            create_pie_chart(roaster_counts_top, 'Roaster', 'Count', width=400, height=400, key='top_roasters', title='Distribution of Top 20 Coffee Roasters')

            ##### Bean Origins 1 Pie Chart #####
            origin_counts = df['origin_1'].value_counts()
            top_origins = origin_counts.nlargest(top_n)
            other_origins = origin_counts.iloc[top_n:].sum()

            origin_counts_top = pd.concat([top_origins, pd.Series({'Other': other_origins})]).reset_index()
            origin_counts_top.columns = ['Origin', 'Count']

            create_pie_chart(origin_counts_top, 'Origin', 'Count', width=400, height=400, key='bean_origins', title=f'Distribution of Top {top_n} Bean Origins in "origin_1"')

        with col[2]:
            ##### Roaster Locations Pie Chart #####
            loc_counts = df['loc_country'].value_counts()

            loc_counts_df = loc_counts.reset_index()
            loc_counts_df.columns = ['Location', 'Count']
            create_pie_chart(loc_counts_df, 'Location', 'Count', width=400, height=400, key='roaster_locations', title='Distribution of Roaster Locations')

            ##### Bean Origins 2 Pie Chart #####
            origin_counts_2 = df['origin_2'].value_counts()
            top_origins_2 = origin_counts_2.nlargest(top_n)
            other_origins_2 = origin_counts_2.iloc[top_n:].sum()

            origin_counts_top_2 = pd.concat([top_origins_2, pd.Series({'Other': other_origins_2})]).reset_index()
            origin_counts_top_2.columns = ['Origin', 'Count']

            create_pie_chart(origin_counts_top_2, 'Origin', 'Count', width=400, height=400, key='bean_origins_2', title=f'Distribution of Top {top_n} Bean Origins in "origin_2"')
            

    else:
        st.error("DataFrame not found. Please go to the Data Cleaning page first. (It is required that you scroll to the bottom)")

###################################################################
# Machine Learning Page ###########################################
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here
    # Coffee Price Prediction Page ################################################
    elif st.session_state.page_selection == "coffee_price_prediction":
    st.header("☕ Coffee Price Prediction")

    # Create a copy of the original DataFrame with a unique name for regression
    df_coffeeprice_regression = st.session_state.df.copy()  # Use cleaned DataFrame stored in session state

    # Select relevant columns and rename them if pre-processed (already cleaned)
    df_coffeeprice_regression = df_coffeeprice_regression[['roast', 'origin_2', 'loc_country', '100g_USD']]
    df_coffeeprice_regression = df_coffeeprice_regression.rename(columns={
        'origin_2': 'origin_2_processed',
        'loc_country': 'loc_processed'
    })

    # Drop rows with missing values if any
    df_coffeeprice_regression = df_coffeeprice_regression.dropna()

    # Handling Outliers using IQR for '100g_USD'
    Q1 = df_coffeeprice_regression['100g_USD'].quantile(0.25)
    Q3 = df_coffeeprice_regression['100g_USD'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_coffeeprice_regression = df_coffeeprice_regression[(df_coffeeprice_regression['100g_USD'] >= lower_bound) & 
                                                          (df_coffeeprice_regression['100g_USD'] <= upper_bound)]

    # Encoding Categorical Variables
    le_roast = LabelEncoder()
    le_origin_2 = LabelEncoder()
    le_location = LabelEncoder()

    df_coffeeprice_regression['roast'] = le_roast.fit_transform(df_coffeeprice_regression['roast'])
    df_coffeeprice_regression['origin_2_processed'] = le_origin_2.fit_transform(df_coffeeprice_regression['origin_2_processed'])
    df_coffeeprice_regression['loc_processed'] = le_location.fit_transform(df_coffeeprice_regression['loc_processed'])

    # Define features (X) and target (y)
    X = df_coffeeprice_regression[['roast', 'origin_2_processed', 'loc_processed']]
    y = df_coffeeprice_regression['100g_USD']

    # No need for SMOTE since this is a regression problem
    # Split the data into training (70%) and testing (30%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_pred)
    st.write(f"Random Forest - Mean Squared Error: {rf_mse:.2f}")

    # Decision Tree Regressor
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_mse = mean_squared_error(y_test, dt_pred)
    st.write(f"Decision Tree - Mean Squared Error: {dt_mse:.2f}")

    # Visualize Actual vs Predicted Prices for Random Forest
    st.write("### Actual vs Predicted Prices (Random Forest)")
    fig, ax = plt.subplots()
    ax.scatter(y_test, rf_pred, label='Random Forest', alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Actual Price (USD)')
    ax.set_ylabel('Predicted Price (USD)')
    ax.legend()
    st.pyplot(fig)

    # Feature Importance Analysis for Random Forest
    st.write("### Feature Importance (Random Forest)")
    rf_importance = rf_model.feature_importances_
    for i, score in enumerate(rf_importance):
        st.write(f"Feature: {X.columns[i]}, Score: {score:.4f}")

###################################################################

# Description to Rating Page ################################################
elif st.session_state.page_selection == "description_to_rating":
    st.header("📊 Description to Rating")

    # Initialize SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    # Define function to extract sentiment
    def extract_sentiment(text):
        sentiment_score = sia.polarity_scores(text)
        return sentiment_score['compound']  # Compound score as overall sentiment

    # Access the cleaned data from session state
    if 'df' not in st.session_state:
        st.error("Please process the data in the Data Cleaning page first")
        st.stop()

    df = st.session_state.df  # Use cleaned DataFrame stored in session state

    # Add tabs for different features
    tab1, tab2 = st.tabs(["Describe to Rate The Coffee", "Model Insights"])

    with tab1:
        # Check if sentiment score columns exist; if not, create them
        if not all(col in df.columns for col in ['sentiment_score_1', 'sentiment_score_2', 'sentiment_score_3']):
            # Create sentiment score columns based on existing descriptions
            df['sentiment_score_1'] = df['desc_1'].apply(extract_sentiment)
            df['sentiment_score_2'] = df['desc_2'].apply(extract_sentiment)
            df['sentiment_score_3'] = df['desc_3'].apply(extract_sentiment)

        # Feature columns for training (using the three individual sentiment scores)
        X = df[['sentiment_score_1', 'sentiment_score_2', 'sentiment_score_3']]
        y = df['rating']

        # Split the data into training (70%) and testing (30%) sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize the Random Forest Regressor
        rf = RandomForestRegressor(random_state=42)

        # Fit the model
        rf.fit(X_train, y_train)

        # Input field for coffee name
        coffee_names = df['name'].unique() 
        selected_coffee = st.selectbox("Select Coffee Name", coffee_names)

        # Get the existing descriptions for the selected coffee
        existing_desc = df[df['name'] == selected_coffee].iloc[0]
        desc_1 = existing_desc['desc_1']
        desc_2 = existing_desc['desc_2']
        desc_3 = existing_desc['desc_3']

        # Input fields for new coffee descriptions
        st.subheader("Describe This Coffee")
        new_desc_1 = st.text_area("New Description 1", "")
        new_desc_2 = st.text_area("New Description 2", "")
        new_desc_3 = st.text_area("New Description 3", "")

        if st.button("Predict Rating"):
            # Calculate sentiment scores for the new descriptions
            sentiment_score_1 = extract_sentiment(new_desc_1)
            sentiment_score_2 = extract_sentiment(new_desc_2)
            sentiment_score_3 = extract_sentiment(new_desc_3)

            # Prepare the input for prediction
            new_data = pd.DataFrame([[sentiment_score_1, sentiment_score_2, sentiment_score_3]], 
                                    columns=['sentiment_score_1', 'sentiment_score_2', 'sentiment_score_3'])

            # Predict the new rating
            predicted_rating = rf.predict(new_data)

            # Display the sentiment scores and predicted rating
            st.write("Sentiment Scores:")
            st.write(f"Description 1 Sentiment Score: {sentiment_score_1:.2f}")
            st.write(f"Description 2 Sentiment Score: {sentiment_score_2:.2f}")
            st.write(f"Description 3 Sentiment Score: {sentiment_score_3:.2f}")
            st.success(f"Predicted Rating for the provided descriptions: {predicted_rating[0]:.2f}")

    with tab2:
        st.subheader("Feature Importance Analysis")
        
        # Calculate the average sentiment score for each coffee description (mean of the 3 sentiment scores)
        df['average_sentiment'] = df[['sentiment_score_1', 'sentiment_score_2', 'sentiment_score_3']].mean(axis=1)

        # Rescale sentiment scores from range [0, 1] to range [84, 98]
        min_rating = 84
        max_rating = 98
        df['rescaled_sentiment'] = min_rating + (df['average_sentiment'] * (max_rating - min_rating))

        # Create hexbin plot using Plotly
        hexbin_fig = go.Figure(data=go.Histogram2d(
            x=df['rescaled_sentiment'],
            y=df['rating'],
            colorscale='Blues',
            colorbar=dict(title='Frequency'),
            histfunc='count',
        ))
        
        # Update layout for better visualization
        hexbin_fig.update_layout(
            title='Hexbin Plot of Sentiment Scores vs Ratings',
            xaxis_title='Sentiment Scores',
            yaxis_title='Ratings',
            plot_bgcolor='rgba(0,0,0,0)',
        )

        # Show the plot in Streamlit
        st.plotly_chart(hexbin_fig, use_container_width=True)

        st.write("""
        The hexbin plot shows the relationship between "Sentiment Scores" and "Ratings." The plot suggests a positive correlation between "Rescaled Sentiment Scores" and "Ratings." As the sentiment scores increase, the ratings tend to also increase.
        """)

        # Prepare data for scatter plot
        X = df[['sentiment_score_1', 'sentiment_score_2', 'sentiment_score_3']]
        y = df['rating']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize and fit the Random Forest Regressor
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)

        # Make predictions
        y_pred = rf.predict(X_test)

        # Create scatter plot using Plotly Express
        scatter_fig = px.scatter(
            df,
            x='rescaled_sentiment',
            y='rating',
            title='Scatter Plot of Rescaled Sentiment Scores vs Ratings',
            labels={'rescaled_sentiment': 'Rescaled Sentiment Scores', 'rating': 'Ratings'},
            color='rating',  # Optional: color by rating
            hover_data=['rescaled_sentiment', 'rating']  # Optional: show data on hover
        )
        
        # Update layout for better visualization
        scatter_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
        )
        
        # Show the plot in Streamlit
        st.plotly_chart(scatter_fig, use_container_width=True)

        # Calculate and display MAE and RMSE
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

        st.write(f"The model's Mean Absolute Error (MAE) is 1.1728734991620522 and the Root Mean Squared Error (RMSE) is 1.5956235031050126. These metrics indicate the average magnitude of the model's prediction errors. A lower MAE suggests smaller average errors, while a lower RMSE suggests smaller average squared errors, with more weight given to larger errors.")

        st.markdown("""
        ### Understanding the Rating Prediction Model
        
        This rating prediction model utilizes multiple features derived from coffee descriptions to estimate the overall rating:
        
        1. **Description Sentiment Analysis**: Analyzes the sentiment of the coffee descriptions using Natural Language Processing (NLP) techniques to quantify the emotional tone and context.
        
        2. **Sentiment Scores**: Incorporates individual sentiment scores from multiple descriptions to capture different perspectives on the coffee's characteristics.
        
        3. **Model Training**: Utilizes a Random Forest Regressor to learn from historical data, identifying patterns and relationships between sentiment scores and actual ratings.
        
        The bar chart above represents the contribution of each sentiment score to the rating prediction.
        """)


# Prediction Page #################################################
elif st.session_state.page_selection == "prediction":
    st.header("☕ Coffee Recommendation System")
    
    # Check if the cleaned DataFrame exists in session state
    if 'df' not in st.session_state:
        st.error("Please process the data in the Data Cleaning page first")
        st.stop()
        
    df = st.session_state.df
    
    # Add tabs for different features
    tab1, tab2 = st.tabs(["Get Recommendations", "Model Insights"])
    
    with tab1:
        # Create sidebar for filters
        with st.sidebar:
            st.subheader("Filter Options")
            
            # Price range filter
            price_range = st.slider(
                "Price Range (USD/100g)",
                float(df['100g_USD'].min()),
                float(df['100g_USD'].max()),
                (float(df['100g_USD'].min()), float(df['100g_USD'].max()))
            )
            
            # Roast type filter
            roast_types = ['All'] + list(df['roast'].unique())
            selected_roast = st.selectbox("Roast Type", roast_types)
            
            # Origin filter
            origins = ['All'] + list(df['origin_1'].unique())
            selected_origin = st.selectbox("Origin", origins)

        # Main content area
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Select Your Coffee")
            
            # Filter dataset based on selections
            filtered_df = df.copy()
            if selected_roast != 'All':
                filtered_df = filtered_df[filtered_df['roast'] == selected_roast]
            if selected_origin != 'All':
                filtered_df = filtered_df[filtered_df['origin_1'] == selected_origin]
            filtered_df = filtered_df[
                (filtered_df['100g_USD'] >= price_range[0]) &
                (filtered_df['100g_USD'] <= price_range[1])
            ]
            
            # Coffee selection
            selected_coffee = st.selectbox(
                "Choose a coffee to get recommendations",
                options=filtered_df['name'].unique()
            )

            if selected_coffee:
                coffee_info = filtered_df[filtered_df['name'] == selected_coffee].iloc[0]
                st.write("**Selected Coffee Details:**")
                st.write(f"🫘 Roast: {coffee_info['roast']}")
                st.write(f"📍 Origin: {coffee_info['origin_1']}")
                st.write(f"💰 Price: ${coffee_info['100g_USD']:.2f}/100g")
                st.write(f"⭐ Rating: {coffee_info['rating']}")

        with col2:
            if selected_coffee:
                st.subheader("Recommended Coffees")
                
                # Create recommendation dataframe
                df_reco = filtered_df.copy()
                columns_of_interest = ['name', 'roast', 'origin_1', 'origin_2', '100g_USD', 
                                     'desc_1_processed', 'desc_2_processed', 'desc_3_processed']
                df_reco = df_reco[columns_of_interest]

                # Combine descriptions
                df_reco['combined_desc'] = df_reco.apply(
                    lambda row: ' '.join(str(x) for x in [
                        row['desc_1_processed'],
                        row['desc_2_processed'],
                        row['desc_3_processed']
                    ]), 
                    axis=1
                )

                # Create TF-IDF matrix
                tfidf = TfidfVectorizer(stop_words='english')
                tfidf_matrix = tfidf.fit_transform(df_reco['combined_desc'])
                
                # Calculate cosine similarity
                cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

                # Get recommendations
                idx = df_reco[df_reco['name'] == selected_coffee].index[0]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:6]  # Get top 5 similar coffees
                
                # Create recommendations with details
                for i, (idx, score) in enumerate(sim_scores, 1):
                    coffee = df_reco.iloc[idx]
                    with st.container():
                        st.markdown(
                            f"""
                            <div style='padding: 10px; border-radius: 5px; background-color: rgba(255, 255, 255, 0.05); margin-bottom: 10px;'>
                                <h3>{i}. {coffee['name']}</h3>
                                <p><strong>Similarity Score:</strong> {score:.2%}</p>
                                <p>🫘 <strong>Roast:</strong> {coffee['roast']}</p>
                                <p>📍 <strong>Origin:</strong> {coffee['origin_1']}</p>
                                <p>💰 <strong>Price:</strong> ${coffee['100g_USD']:.2f}/100g</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )

    with tab2:
        st.subheader("Feature Importance Analysis")
        
        # Calculate and display feature importance
        def calculate_permutation_importance(df, baseline_sim_matrix, feature_columns):
            importance_scores = {}
            for feature in feature_columns:
                df_permuted = df.copy()
                df_permuted[feature] = np.random.permutation(df_permuted[feature].values)
                
                if feature in ['desc_1_processed', 'desc_2_processed', 'desc_3_processed']:
                    df_permuted['combined_desc'] = df_permuted.apply(
                        lambda row: ' '.join(str(x) for x in [
                            row['desc_1_processed'],
                            row['desc_2_processed'],
                            row['desc_3_processed']
                        ]), 
                        axis=1
                    )
                    tfidf_matrix_permuted = tfidf.fit_transform(df_permuted['combined_desc'])
                    permuted_sim_matrix = cosine_similarity(tfidf_matrix_permuted, tfidf_matrix_permuted)
                else:
                    permuted_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

                sim_diff = np.abs(baseline_sim_matrix - permuted_sim_matrix).mean()
                importance_scores[feature] = sim_diff
            
            return sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

        feature_columns = columns_of_interest
        importance_scores = calculate_permutation_importance(df_reco, cosine_sim, feature_columns)
        
        # Create and display bar chart
        importance_df = pd.DataFrame(importance_scores, columns=['Feature', 'Importance'])
        fig = px.bar(
            importance_df, 
            x='Feature', 
            y='Importance',
            title='Feature Importance in Coffee Recommendations',
            labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'}
        )
        fig.update_layout(
            xaxis_title="Features",
            yaxis_title="Importance Score",
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        ### Understanding the Model
        
        This recommendation system uses several features to find similar coffees:
        
        1. **Description Analysis**: Processes the coffee descriptions using NLP techniques
        2. **Roast Type**: Considers the roasting style of the coffee
        3. **Origin**: Takes into account where the coffee beans come from
        4. **Price**: Considers the price point of the coffee
        
        The bar chart above shows how much each feature contributes to the recommendations.
        Higher bars indicate more important features in determining coffee similarity.
        """)




    # Your content for the PREDICTION page goes here

###################################################################
# Conclusions Page ################################################
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here
