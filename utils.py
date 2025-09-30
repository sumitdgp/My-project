import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import streamlit as st

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class CourseRecommender:
    def __init__(self):
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
    def load_data(self, file_path):
        """Load course data from CSV"""
        self.df = pd.read_csv(file_path)
        return f"Loaded {len(self.df)} courses"
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_data(self):
        """Preprocess the course data"""
        # Fill missing values
        self.df['description'] = self.df['description'].fillna('')
        self.df['category'] = self.df['category'].fillna('Unknown')
        self.df['skills'] = self.df['skills'].fillna('')
        self.df['difficulty'] = self.df['difficulty'].fillna('Beginner')
        self.df['rating'] = self.df['rating'].fillna(4.0)
        
        # Preprocess text fields
        self.df['title_clean'] = self.df['title'].apply(self.preprocess_text)
        self.df['description_clean'] = self.df['description'].apply(self.preprocess_text)
        self.df['category_clean'] = self.df['category'].apply(self.preprocess_text)
        self.df['skills_clean'] = self.df['skills'].apply(self.preprocess_text)
        
        # Create combined feature for TF-IDF
        self.df['combined_features'] = (
            self.df['title_clean'] + ' ' +
            self.df['description_clean'] + ' ' +
            self.df['category_clean'] + ' ' +
            self.df['skills_clean']
        )
        
    def build_model(self):
        """Build TF-IDF model and compute similarity matrix"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['combined_features'])
        return "TF-IDF model built successfully"
        
    def get_recommendations(self, user_skills, user_interests, completed_courses=[], top_n=10):
        """Get course recommendations based on user profile"""
        if self.df is None or self.tfidf_matrix is None:
            raise ValueError("Model not trained. Call load_data and build_model first.")
        
        # Create user profile
        user_profile = ' '.join(user_skills) + ' ' + ' '.join(user_interests)
        user_profile_clean = self.preprocess_text(user_profile)
        
        # Transform user profile to TF-IDF
        user_tfidf = self.tfidf_vectorizer.transform([user_profile_clean])
        
        # Compute cosine similarity
        cosine_sim = cosine_similarity(user_tfidf, self.tfidf_matrix).flatten()
        
        # Get similarity scores
        similarity_scores = list(enumerate(cosine_sim))
        
        # Sort courses by similarity score
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Filter out completed courses and get top recommendations
        recommendations = []
        for idx, score in similarity_scores:
            course_id = self.df.iloc[idx]['course_id']
            if course_id not in completed_courses:
                course_data = self.df.iloc[idx].to_dict()
                course_data['similarity_score'] = float(score)
                recommendations.append(course_data)
            
            if len(recommendations) >= top_n:
                break
        
        return recommendations
    
    def get_similar_courses(self, course_id, top_n=5):
        """Get similar courses based on a given course"""
        if self.df is None or self.tfidf_matrix is None:
            raise ValueError("Model not trained. Call load_data and build_model first.")
        
        # Find course index
        course_idx = self.df[self.df['course_id'] == course_id].index
        if len(course_idx) == 0:
            return []
        
        course_idx = course_idx[0]
        
        # Compute similarity with all courses
        cosine_sim = cosine_similarity(
            self.tfidf_matrix[course_idx:course_idx+1], 
            self.tfidf_matrix
        ).flatten()
        
        # Get similar courses (excluding the input course)
        similar_indices = cosine_sim.argsort()[::-1][1:top_n+1]
        
        similar_courses = []
        for idx in similar_indices:
            course_data = self.df.iloc[idx].to_dict()
            course_data['similarity_score'] = float(cosine_sim[idx])
            similar_courses.append(course_data)
        
        return similar_courses

def initialize_session_state():
    """Initialize session state variables"""
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': '',
            'skills': [],
            'interests': [],
            'completed_courses': [],
            'bookmarks': []
        }
    if 'recommender' not in st.session_state:
        st.session_state.recommender = CourseRecommender()
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

def display_course_card(course, show_similarity=False):
    """Display a course in a card format"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(course['title'])
        st.write(f"**Platform:** {course['platform']} | **Category:** {course['category']} | **Difficulty:** {course['difficulty']}")
        st.write(f"**Rating:** ⭐ {course['rating']} | **Duration:** {course['duration']} hours | **Price:** {course['price']}")
        
        # Display skills
        skills = course['skills'].split(',')
        skill_chips = " ".join([f"`{skill.strip()}`" for skill in skills[:4]])
        st.write(f"**Skills:** {skill_chips}")
        
        st.write(course['description'])
    
    with col2:
        if show_similarity:
            st.metric("Match Score", f"{course['similarity_score']*100:.1f}%")
        
        # Bookmark button
        bookmark_key = f"bookmark_{course['course_id']}"
        is_bookmarked = course['course_id'] in st.session_state.user_profile['bookmarks']
        
        if st.button("⭐ Bookmark" if not is_bookmarked else "★ Bookmarked", 
                    key=bookmark_key, use_container_width=True):
            if course['course_id'] not in st.session_state.user_profile['bookmarks']:
                st.session_state.user_profile['bookmarks'].append(course['course_id'])
                st.success("Course bookmarked!")
            else:
                st.session_state.user_profile['bookmarks'].remove(course['course_id'])
                st.info("Bookmark removed!")
        
        # Enroll button
        if st.button("Enroll Now", key=f"enroll_{course['course_id']}", 
                    use_container_width=True, type="primary"):
            st.markdown(f"[Go to Course]({course['enroll_link']})")
    
    st.divider()

def get_available_skills(df):
    """Extract all unique skills from the dataset"""
    all_skills = []
    for skills_str in df['skills'].dropna():
        skills = [skill.strip() for skill in skills_str.split(',')]
        all_skills.extend(skills)
    return sorted(list(set(all_skills)))
