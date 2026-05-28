import pickle
import streamlit as st
import pandas as pd
import wikipedia
import os
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression

# --- 1.  POSTER FETCHING (WIKIPEDIA) ---
def fetch_poster(movie_title):
    try:
        # Search using "film" to target the specific movie page over character profiles
        search_results = wikipedia.search(movie_title + " film")
        if not search_results:
            return "https://via.placeholder.com/500x750?text=No+Search+Result"
        
        page = wikipedia.page(search_results[0], auto_suggest=False)
        all_images = page.images
        
        # Pass 1: Look for an explicit poster, cover, or infobox image layout asset
        for img_url in all_images:
            if isinstance(img_url, str) and img_url.startswith("http"):
                normalized_url = img_url.lower()
                if any(ext in normalized_url for ext in ['.jpg', '.jpeg', '.png']):
                    if "poster" in normalized_url or "cover" in normalized_url or "infobox" in normalized_url:
                        return img_url
                        
        # Pass 2: Fallback to the first standard movie image layout asset if no explicit keywords match
        for img_url in all_images:
            if isinstance(img_url, str) and img_url.startswith("http"):
                normalized_url = img_url.lower()
                # Skip administrative wiki icons or generic template images
                if any(bad_word in normalized_url for bad_word in ['stub', 'folder', 'ambox', 'commons']):
                    continue
                if any(ext in normalized_url for ext in ['.jpg', '.jpeg', '.png']):
                    return img_url
                    
        return "https://via.placeholder.com/500x750?text=Poster+Not+Found"
    except Exception:
        return "https://via.placeholder.com/500x750?text=Image+Unavailable"

# --- 2. RECOMMENDATION LOGIC ---
def recommend(movie):
    try:
        index = movies[movies['title'] == movie].index[0]
    except IndexError:
        return [], [], [], []
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    names, posters, years, ratings = [], [], [], []
    for i in distances[1:6]:
        full_title = movies.iloc[i[0]].title
        clean_name = full_title.split('(')[0].strip()
        posters.append(fetch_poster(clean_name))
        names.append(full_title)
        years.append(movies.iloc[i[0]].year)
        ratings.append(movies.iloc[i[0]].vote_average)
    return names, posters, years, ratings

# --- 3. UI LAYOUT & SIDEBAR NAVIGATION ---
st.set_page_config(layout="wide", page_title="AI Movie Analytics Platform")

st.sidebar.title("📊 Project Modules")
app_mode = st.sidebar.radio("Navigate to:", ["Movie Recommendations", "Sentiment Analysis", "Power BI Dashboard"])

st.sidebar.markdown("---")

# --- 4. DATA LOADING ---
try:
    movies_dict = pickle.load(open('artifacts/movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('artifacts/similarity.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found! Ensure 'artifacts/' folder is correct.")
    st.stop()

# --- 5. MODULE 1: RECOMMENDATION SYSTEM ---
if app_mode == "Movie Recommendations":
    st.header(' AI-Driven Movie Recommendation System')
    movie_list = movies['title'].values
    selected_movie = st.selectbox("Select a movie to get recommendations:", movie_list)

    if st.button('Show Recommendations'):
        with st.spinner('Analyzing metadata similarity...'):
            res_names, res_posters, res_years, res_ratings = recommend(selected_movie)
        
        if res_names:
            cols = st.columns(5)
            for i, col in enumerate(cols):
                with col:
                    st.subheader(res_names[i].split('(')[0].strip())
                    # Fixed deprecation warning syntax by replacing use_container_width
                    st.image(res_posters[i], width="stretch")
                    year = res_years[i]
                    st.caption(f"Year: {int(year)}" if pd.notna(year) else "Year: N/A")
                    st.caption(f"Rating: {res_ratings[i]:.1f} ⭐")

# --- 6. MODULE 2: SENTIMENT ANALYSIS ---
elif app_mode == "Sentiment Analysis":
    st.header("🎭 Movie Review Sentiment Analysis")
    st.write("This module uses Natural Language Processing (NLP) to detect if a review is Positive, Negative, or Neutral.")
    
    user_review = st.text_area("Paste a movie review or audience comment here:", height=150)
    
    if st.button("Analyze Sentiment"):
        if user_review.strip():
            analyzer = SentimentIntensityAnalyzer()
            sentiment_dict = analyzer.polarity_scores(user_review)
            compound_score = sentiment_dict['compound']
            
            st.markdown("### Analysis Results:")
            if compound_score >= 0.05:
                st.success(f"**Positive Sentiment** (Score: {compound_score}) 😄")
                st.balloons()
            elif compound_score <= -0.05:
                st.error(f"**Negative Sentiment** (Score: {compound_score}) 😡")
            else:
                st.warning(f"**Neutral Sentiment** (Score: {compound_score}) 😐")
            
            # Show raw breakdown
            with st.expander("See Sentiment Breakdown"):
                st.write(f"Positive: {sentiment_dict['pos']}")
                st.write(f"Neutral: {sentiment_dict['neu']}")
                st.write(f"Negative: {sentiment_dict['neg']}")
        else:
            st.warning("Please enter a review first.")

# --- 7. MODULE 3: POWER BI DASHBOARD ---
elif app_mode == "Power BI Dashboard":
    st.header("📈 Interactive Data Analytics Dashboard")
    st.write("This module integrates Microsoft Power BI for real-time visualization of the TMDB 5000 dataset.")
    
    power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=daeb3778-5d97-4fd2-9ad0-4e178754fb58&autoAuth=true&ctid=fe1ccc9f-0b9b-4c7b-b377-9763d4148e63"
    
    # Display the dashboard using an iframe component
    st.components.v1.iframe(power_bi_url, height=800, scrolling=True)
    
    st.markdown("---")
    st.info("💡 **BI Integration:** This demonstrates the bridge between Python-based Machine Learning and enterprise Business Intelligence tools.")