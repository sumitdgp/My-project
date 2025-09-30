import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
import plotly.express as px

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class CourseRecommender:
    def __init__(self):
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
    def load_data(self, file_path='courses.csv'):
        """Load course data from CSV"""
        try:
            self.df = pd.read_csv(file_path)
            return f"✅ Loaded {len(self.df)} courses successfully!"
        except Exception as e:
            # If CSV file not found, create sample data
            sample_data = [
                {
                    'course_id': 1, 'title': 'Python for Beginners', 
                    'description': 'Learn Python programming from scratch with hands-on projects',
                    'category': 'Programming', 'platform': 'Udemy', 'difficulty': 'Beginner', 
                    'rating': 4.5, 'duration': 10, 'skills': 'Python Programming Basic Syntax',
                    'enroll_link': 'https://www.udemy.com/python-beginners', 'price': '₹499'
                },
                {
                    'course_id': 2, 'title': 'Machine Learning Fundamentals', 
                    'description': 'Introduction to ML algorithms and practical implementation',
                    'category': 'Data Science', 'platform': 'Coursera', 'difficulty': 'Intermediate', 
                    'rating': 4.7, 'duration': 15, 'skills': 'Machine Learning Python Statistics',
                    'enroll_link': 'https://www.coursera.org/ml-fundamentals', 'price': 'Free'
                },
                {
                    'course_id': 3, 'title': 'Web Development Bootcamp', 
                    'description': 'Full-stack web development with modern technologies',
                    'category': 'Web Development', 'platform': 'Udemy', 'difficulty': 'Intermediate', 
                    'rating': 4.6, 'duration': 20, 'skills': 'HTML CSS JavaScript React',
                    'enroll_link': 'https://www.udemy.com/web-dev-bootcamp', 'price': '₹1299'
                }
            ]
            self.df = pd.DataFrame(sample_data)
            return f"⚠️ CSV not found! Loaded {len(self.df)} sample courses."
        
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
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['combined_features'])
        return "✅ AI Model built successfully!"
        
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

def initialize_session_state():
    """Initialize session state variables"""
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': '',
            'skills': [],
            'interests': [],
            'completed_courses': [],
            'bookmarks': [],
            'career_goals': ''
        }
    if 'recommender' not in st.session_state:
        st.session_state.recommender = CourseRecommender()
        # Load and initialize the model
        message = st.session_state.recommender.load_data('courses.csv')
        st.session_state.recommender.preprocess_data()
        build_message = st.session_state.recommender.build_model()
        
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
        skills = course['skills'].split()
        skill_chips = " ".join([f"`{skill}`" for skill in skills[:4]])
        st.write(f"**Skills:** {skill_chips}")
        
        st.write(course['description'])
    
    with col2:
        if show_similarity and 'similarity_score' in course:
            similarity_percent = course['similarity_score'] * 100
            color = "green" if similarity_percent > 70 else "orange" if similarity_percent > 50 else "red"
            st.metric("Match Score", f"{similarity_percent:.1f}%")
        
        # Bookmark button
        bookmark_key = f"bookmark_{course['course_id']}"
        is_bookmarked = course['course_id'] in st.session_state.user_profile['bookmarks']
        
        if st.button("⭐ Bookmark" if not is_bookmarked else "★ Bookmarked", 
                    key=bookmark_key, use_container_width=True):
            if course['course_id'] not in st.session_state.user_profile['bookmarks']:
                st.session_state.user_profile['bookmarks'].append(course['course_id'])
                st.success("✅ Course bookmarked!")
            else:
                st.session_state.user_profile['bookmarks'].remove(course['course_id'])
                st.info("📝 Bookmark removed!")
            st.rerun()
        
        # Enroll button
        if st.button("🎯 Enroll Now", key=f"enroll_{course['course_id']}", 
                    use_container_width=True, type="primary"):
            st.markdown(f"[📚 Go to Course Page]({course['enroll_link']})")
    
    st.divider()

def get_available_skills(df):
    """Extract all unique skills from the dataset"""
    all_skills = []
    for skills_str in df['skills'].dropna():
        skills = [skill.strip() for skill in skills_str.split()]
        all_skills.extend(skills)
    return sorted(list(set(all_skills)))

def show_home_page():
    """Display the home page"""
    st.title("🎓 Welcome to Course Recommendation System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Discover Your Perfect Learning Path
        
        This intelligent AI system helps you find the best courses based on:
        - Your **current skills** and knowledge level
        - Your **career interests** and goals  
        - Your **learning preferences** and style
        - **AI-powered** recommendations using machine learning
        
        ### 🚀 How it works:
        1. **Set up your profile** - Tell us about your skills and interests
        2. **Get personalized recommendations** - Our AI suggests perfect courses
        3. **Browse and bookmark** - Explore courses and save your favorites
        4. **Track your progress** - Mark courses as completed
        
        ### 📈 Features:
        - 🤖 AI-Powered Recommendations
        - 📚 20+ Courses across multiple platforms
        - ⭐ Smart Bookmarking System
        - 📊 Learning Analytics
        - 🎯 Personalized Matching
        """)
    
    with col2:
        st.markdown("""
        <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1e40af;'>
        <h3 style='color: #1e40af; margin-top: 0;'>🚀 Quick Start</h3>
        <ol style='color: #374151;'>
        <li>Go to <b>👤 User Profile</b></li>
        <li>Add your skills & interests</li>
        <li>Get <b>💡 AI Recommendations</b></li>
        <li>Browse and bookmark courses</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📞 Support")
        st.info("""
        **Need help?**
        - Check the user profile section
        - Ensure you select relevant skills
        - Use filters while browsing
        """)
    
    # Quick stats
    st.markdown("---")
    st.subheader("📊 System Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Courses", len(st.session_state.recommender.df))
    with col2:
        st.metric("Categories", st.session_state.recommender.df['category'].nunique())
    with col3:
        st.metric("Platforms", st.session_state.recommender.df['platform'].nunique())
    with col4:
        avg_rating = st.session_state.recommender.df['rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.1f} ⭐")

def show_profile_page():
    """Display user profile page"""
    st.title("👤 User Profile Management")
    
    with st.form("user_profile_form"):
        st.subheader("📝 Tell us about yourself")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input(
                "**Your Name**", 
                value=st.session_state.user_profile['name'],
                placeholder="Enter your full name"
            )
            
        with col2:
            career_goals = st.text_input(
                "**Career Goals**",
                value=st.session_state.user_profile.get('career_goals', ''),
                placeholder="e.g., Become a Data Scientist"
            )
        
        # Skills selection
        available_skills = get_available_skills(st.session_state.recommender.df)
        
        st.subheader("🛠️ Skills & Interests")
        col1, col2 = st.columns(2)
        
        with col1:
            skills = st.multiselect(
                "**Your Current Skills**",
                options=available_skills,
                default=st.session_state.user_profile['skills'],
                help="Select skills you already have experience with"
            )
        
        with col2:
            interests = st.multiselect(
                "**Your Learning Interests**",
                options=available_skills,
                default=st.session_state.user_profile['interests'],
                help="Select skills/topics you want to learn or improve"
            )
        
        # Completed courses
        st.subheader("📖 Learning History")
        completed_courses = st.multiselect(
            "**Completed Courses**",
            options=st.session_state.recommender.df['title'].tolist(),
            default=st.session_state.user_profile['completed_courses'],
            help="Select courses you have already completed"
        )
        
        # Submit button
        if st.form_submit_button("💾 Save Profile", use_container_width=True, type="primary"):
            st.session_state.user_profile.update({
                'name': name,
                'skills': skills,
                'interests': interests,
                'career_goals': career_goals,
                'completed_courses': completed_courses
            })
            st.success("🎉 Profile updated successfully!")
    
    # Display current profile
    st.markdown("---")
    st.subheader("👀 Current Profile Summary")
    
    if st.session_state.user_profile['name']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 Personal Information**")
            st.write(f"**Name:** {st.session_state.user_profile['name']}")
            if st.session_state.user_profile.get('career_goals'):
                st.write(f"**Career Goals:** {st.session_state.user_profile['career_goals']}")
            
            st.markdown("**🛠️ Technical Skills**")
            if st.session_state.user_profile['skills']:
                for skill in st.session_state.user_profile['skills']:
                    st.write(f"• {skill}")
            else:
                st.write("No skills added yet")
        
        with col2:
            st.markdown("**🎯 Learning Interests**")
            if st.session_state.user_profile['interests']:
                for interest in st.session_state.user_profile['interests']:
                    st.write(f"• {interest}")
            else:
                st.write("No interests added yet")
            
            st.markdown("**📊 Learning Stats**")
            st.write(f"**Completed Courses:** {len(st.session_state.user_profile['completed_courses'])}")
            st.write(f"**Bookmarked Courses:** {len(st.session_state.user_profile['bookmarks'])}")
    else:
        st.info("ℹ️ Please complete your profile to get personalized course recommendations!")

def show_recommendations_page():
    """Display recommendations page"""
    st.title("💡 AI-Powered Course Recommendations")
    
    if not st.session_state.user_profile['skills'] and not st.session_state.user_profile['interests']:
        st.warning("""
        ⚠️ **Please complete your profile first!** 
        
        Go to the **👤 User Profile** section and add your skills and interests to get personalized recommendations.
        """)
        if st.button("👤 Go to Profile", use_container_width=True):
            st.session_state.page = 'profile'
            st.rerun()
        return
    
    # Recommendation options
    st.subheader("🎛️ Recommendation Settings")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_recommendations = st.slider(
            "**Number of courses**", 
            min_value=5, 
            max_value=20, 
            value=10,
            help="How many courses would you like to see?"
        )
    
    with col2:
        use_interests = st.checkbox(
            "**Include interests**", 
            value=True,
            help="Consider your learning interests in recommendations"
        )
    
    with col3:
        filter_difficulty = st.selectbox(
            "**Difficulty Level**",
            options=["All", "Beginner", "Intermediate", "Advanced"],
            help="Filter by course difficulty"
        )
    
    # Get recommendations button
    if st.button("🎯 Get AI Recommendations", use_container_width=True, type="primary"):
        with st.spinner("🔍 Analyzing your profile and finding the best courses..."):
            user_skills = st.session_state.user_profile['skills']
            user_interests = st.session_state.user_profile['interests'] if use_interests else []
            completed_courses = st.session_state.user_profile['completed_courses']
            
            try:
                recommendations = st.session_state.recommender.get_recommendations(
                    user_skills=user_skills,
                    user_interests=user_interests,
                    completed_courses=completed_courses,
                    top_n=num_recommendations
                )
                
                # Apply difficulty filter
                if filter_difficulty != "All":
                    recommendations = [course for course in recommendations if course['difficulty'] == filter_difficulty]
                
                st.session_state.last_recommendations = recommendations
                st.success(f"✅ Found {len(recommendations)} courses matching your profile!")
                
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {str(e)}")
    
    # Display recommendations
    if 'last_recommendations' in st.session_state and st.session_state.last_recommendations:
        st.subheader(f"🎯 Recommended Courses for {st.session_state.user_profile['name'] or 'You'}")
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_match = np.mean([course['similarity_score'] for course in st.session_state.last_recommendations]) * 100
            st.metric("Average Match", f"{avg_match:.1f}%")
        with col2:
            platforms = len(set(course['platform'] for course in st.session_state.last_recommendations))
            st.metric("Platforms", platforms)
        with col3:
            categories = len(set(course['category'] for course in st.session_state.last_recommendations))
            st.metric("Categories", categories)
        
        st.markdown("---")
        
        # Display each recommended course
        for i, course in enumerate(st.session_state.last_recommendations, 1):
            st.markdown(f"#### #{i} Recommendation")
            display_course_card(course, show_similarity=True)
    
    elif 'last_recommendations' in st.session_state:
        st.info("🤔 No courses found matching your current filters. Try adjusting your preferences or browse all courses.")

def show_browse_page():
    """Display course browsing page"""
    st.title("📚 Browse All Courses")
    
    # Filters
    st.subheader("🔍 Filter Courses")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        category_filter = st.selectbox(
            "**Category**",
            ["All"] + sorted(st.session_state.recommender.df['category'].unique().tolist())
        )
    
    with col2:
        platform_filter = st.selectbox(
            "**Platform**",
            ["All"] + sorted(st.session_state.recommender.df['platform'].unique().tolist())
        )
    
    with col3:
        difficulty_filter = st.selectbox(
            "**Difficulty**",
            ["All"] + sorted(st.session_state.recommender.df['difficulty'].unique().tolist())
        )
    
    with col4:
        min_rating = st.slider("**Min Rating**", 3.0, 5.0, 3.5, 0.1)
    
    # Search
    search_query = st.text_input("🔎 Search courses by title, skills, or description")
    
    # Apply filters
    filtered_df = st.session_state.recommender.df.copy()
    
    if category_filter != "All":
        filtered_df = filtered_df[filtered_df['category'] == category_filter]
    
    if platform_filter != "All":
        filtered_df = filtered_df[filtered_df['platform'] == platform_filter]
    
    if difficulty_filter != "All":
        filtered_df = filtered_df[filtered_df['difficulty'] == difficulty_filter]
    
    filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
    
    if search_query:
        search_lower = search_query.lower()
        filtered_df = filtered_df[
            filtered_df['title'].str.lower().str.contains(search_lower) |
            filtered_df['skills'].str.lower().str.contains(search_lower) |
            filtered_df['description'].str.lower().str.contains(search_lower) |
            filtered_df['category'].str.lower().str.contains(search_lower)
        ]
    
    # Display courses
    st.subheader(f"📖 Found {len(filtered_df)} Courses")
    
    if len(filtered_df) == 0:
        st.info("🤷 No courses found matching your filters. Try adjusting your search criteria.")
        return
    
    # Sort options
    sort_option = st.selectbox(
        "**Sort by**",
        ["Default", "Rating (High to Low)", "Duration (Short to Long)", "Price (Low to High)"]
    )
    
    if sort_option == "Rating (High to Low)":
        filtered_df = filtered_df.sort_values('rating', ascending=False)
    elif sort_option == "Duration (Short to Long)":
        filtered_df = filtered_df.sort_values('duration', ascending=True)
    elif sort_option == "Price (Low to High)":
        # Extract numeric price for sorting
        def extract_price(price_str):
            if price_str == 'Free':
                return 0
            try:
                return int(price_str.replace('₹', '').replace(',', ''))
            except:
                return float('inf')
        
        filtered_df['price_numeric'] = filtered_df['price'].apply(extract_price)
        filtered_df = filtered_df.sort_values('price_numeric', ascending=True)
    
    # Display each course
    for _, course in filtered_df.iterrows():
        display_course_card(course.to_dict())

def show_bookmarks_page():
    """Display bookmarked courses"""
    st.title("⭐ My Bookmarked Courses")
    
    bookmarked_courses = []
    for course_id in st.session_state.user_profile['bookmarks']:
        course = st.session_state.recommender.df[
            st.session_state.recommender.df['course_id'] == course_id
        ]
        if not course.empty:
            bookmarked_courses.append(course.iloc[0].to_dict())
    
    if bookmarked_courses:
        st.success(f"🎉 You have {len(bookmarked_courses)} bookmarked courses!")
        
        # Bookmark stats
        col1, col2, col3 = st.columns(3)
        with col1:
            platforms = len(set(course['platform'] for course in bookmarked_courses))
            st.metric("Platforms", platforms)
        with col2:
            categories = len(set(course['category'] for course in bookmarked_courses))
            st.metric("Categories", categories)
        with col3:
            avg_rating = np.mean([course['rating'] for course in bookmarked_courses])
            st.metric("Avg Rating", f"{avg_rating:.1f}⭐")
        
        st.markdown("---")
        
        # Display bookmarked courses
        for course in bookmarked_courses:
            display_course_card(course)
            
        # Clear all bookmarks option
        if st.button("🗑️ Clear All Bookmarks", type="secondary"):
            st.session_state.user_profile['bookmarks'] = []
            st.success("All bookmarks cleared!")
            st.rerun()
            
    else:
        st.info("""
        📚 **You haven't bookmarked any courses yet!**
        
        Here's how to get started:
        1. Go to **💡 Get Recommendations** or **📚 Browse Courses**
        2. Find courses you're interested in
        3. Click the **⭐ Bookmark** button on any course card
        4. Your bookmarked courses will appear here!
        """)
        
        if st.button("📚 Browse Courses", use_container_width=True):
            st.session_state.page = 'browse'
            st.rerun()

def show_analytics_page():
    """Display analytics and insights"""
    st.title("📊 Learning Analytics & Insights")
    
    if not st.session_state.user_profile['name']:
        st.warning("Complete your profile to see personalized analytics!")
        return
    
    # User statistics
    st.subheader("👤 Your Learning Profile")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Skills", len(st.session_state.user_profile['skills']))
    with col2:
        st.metric("Interests", len(st.session_state.user_profile['interests']))
    with col3:
        st.metric("Completed", len(st.session_state.user_profile['completed_courses']))
    with col4:
        st.metric("Bookmarked", len(st.session_state.user_profile['bookmarks']))
    
    # Category distribution of bookmarked courses
    if st.session_state.user_profile['bookmarks']:
        st.subheader("📈 Your Learning Preferences")
        
        bookmarked_df = st.session_state.recommender.df[
            st.session_state.recommender.df['course_id'].isin(st.session_state.user_profile['bookmarks'])
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            if not bookmarked_df.empty:
                category_counts = bookmarked_df['category'].value_counts()
                fig1 = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Bookmarked Courses by Category"
                )
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Difficulty distribution
            if not bookmarked_df.empty:
                difficulty_counts = bookmarked_df['difficulty'].value_counts()
                fig2 = px.bar(
                    x=difficulty_counts.index,
                    y=difficulty_counts.values,
                    title="Bookmarked Courses by Difficulty",
                    labels={'x': 'Difficulty', 'y': 'Count'}
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    # Platform analytics
    st.subheader("🏢 Course Platform Distribution")
    platform_counts = st.session_state.recommender.df['platform'].value_counts()
    fig3 = px.bar(
        x=platform_counts.index,
        y=platform_counts.values,
        title="Courses by Platform",
        labels={'x': 'Platform', 'y': 'Number of Courses'}
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Category analytics
    st.subheader("📊 Course Category Distribution")
    category_counts = st.session_state.recommender.df['category'].value_counts()
    fig4 = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title="All Courses by Category"
    )
    st.plotly_chart(fig4, use_container_width=True)

def main():
    # Page configuration
    st.set_page_config(
        page_title="Course Recommendation System",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🎓 CourseRec AI")
        st.markdown("---")
        
        # Navigation
        page_options = [
            "🏠 Home", 
            "👤 User Profile", 
            "💡 Get Recommendations", 
            "📚 Browse Courses", 
            "⭐ Bookmarks",
            "📊 Analytics"
        ]
        
        selected_page = st.radio("Navigate to:", page_options)
        
        st.markdown("---")
        
        # Quick user info
        if st.session_state.user_profile['name']:
            st.markdown(f"**👋 Hello, {st.session_state.user_profile['name']}!**")
            st.markdown(f"**⭐ Bookmarks:** {len(st.session_state.user_profile['bookmarks'])}")
            st.markdown(f"**📝 Completed:** {len(st.session_state.user_profile['completed_courses'])}")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        **CourseRec AI** 🤖
        
        An intelligent course recommendation system that uses machine learning to suggest the best learning paths based on your profile.
        
        **Features:**
        - AI-Powered Recommendations
        - Smart Filtering
        - Progress Tracking
        - Multi-platform Courses
        """)
        
        st.markdown("---")
        st.markdown("### 🚀 Quick Actions")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
        
        if st.button("🗑️ Clear Session", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Page routing
    if selected_page == "🏠 Home":
        show_home_page()
    elif selected_page == "👤 User Profile":
        show_profile_page()
    elif selected_page == "💡 Get Recommendations":
        show_recommendations_page()
    elif selected_page == "📚 Browse Courses":
        show_browse_page()
    elif selected_page == "⭐ Bookmarks":
        show_bookmarks_page()
    elif selected_page == "📊 Analytics":
        show_analytics_page()

if __name__ == "__main__":
    main()
