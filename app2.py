import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Sample course data
def get_sample_courses():
    return [
        {
            'course_id': 1, 'title': 'Python for Beginners', 'description': 'Learn Python programming from scratch',
            'category': 'Programming', 'platform': 'Udemy', 'difficulty': 'Beginner', 'rating': 4.5, 'duration': 10,
            'skills': 'Python, Programming, Basics', 'enroll_link': 'https://www.udemy.com/python-beginners', 'price': '₹499'
        },
        {
            'course_id': 2, 'title': 'Machine Learning Fundamentals', 'description': 'Introduction to ML algorithms',
            'category': 'Data Science', 'platform': 'Coursera', 'difficulty': 'Intermediate', 'rating': 4.7, 'duration': 15,
            'skills': 'Machine Learning, Python, Statistics', 'enroll_link': 'https://www.coursera.org/ml-fundamentals', 'price': 'Free'
        },
        {
            'course_id': 3, 'title': 'Web Development Bootcamp', 'description': 'Full-stack web development course',
            'category': 'Web Development', 'platform': 'Udemy', 'difficulty': 'Intermediate', 'rating': 4.6, 'duration': 20,
            'skills': 'HTML, CSS, JavaScript, React', 'enroll_link': 'https://www.udemy.com/web-dev-bootcamp', 'price': '₹1299'
        },
        {
            'course_id': 4, 'title': 'Data Analysis with Python', 'description': 'Data analysis using pandas and numpy',
            'category': 'Data Science', 'platform': 'edX', 'difficulty': 'Intermediate', 'rating': 4.4, 'duration': 12,
            'skills': 'Python, Pandas, NumPy, Data Analysis', 'enroll_link': 'https://www.edx.org/data-analysis-python', 'price': '₹899'
        },
        {
            'course_id': 5, 'title': 'Advanced JavaScript', 'description': 'Deep dive into JavaScript concepts',
            'category': 'Programming', 'platform': 'Udemy', 'difficulty': 'Advanced', 'rating': 4.8, 'duration': 8,
            'skills': 'JavaScript, ES6, Async Programming', 'enroll_link': 'https://www.udemy.com/advanced-js', 'price': '₹799'
        },
        {
            'course_id': 6, 'title': 'React Native Mobile Development', 'description': 'Build mobile apps with React Native',
            'category': 'Mobile Development', 'platform': 'Coursera', 'difficulty': 'Intermediate', 'rating': 4.5, 'duration': 14,
            'skills': 'React Native, JavaScript, Mobile Development', 'enroll_link': 'https://www.coursera.org/react-native', 'price': 'Free'
        },
        {
            'course_id': 7, 'title': 'Cloud Computing Basics', 'description': 'Introduction to cloud platforms',
            'category': 'Cloud Computing', 'platform': 'Udemy', 'difficulty': 'Beginner', 'rating': 4.2, 'duration': 6,
            'skills': 'AWS, Cloud Computing', 'enroll_link': 'https://www.udemy.com/cloud-basics', 'price': '₹399'
        },
        {
            'course_id': 8, 'title': 'UI/UX Design Principles', 'description': 'User interface design fundamentals',
            'category': 'Design', 'platform': 'Coursera', 'difficulty': 'Beginner', 'rating': 4.3, 'duration': 8,
            'skills': 'UI Design, UX Research, Wireframing', 'enroll_link': 'https://www.coursera.org/ui-ux-design', 'price': 'Free'
        },
        {
            'course_id': 9, 'title': 'Database Management', 'description': 'SQL and database design concepts',
            'category': 'Databases', 'platform': 'Udemy', 'difficulty': 'Intermediate', 'rating': 4.4, 'duration': 12,
            'skills': 'SQL, Database Design, MySQL', 'enroll_link': 'https://www.udemy.com/database-management', 'price': '₹599'
        },
        {
            'course_id': 10, 'title': 'Data Science Bootcamp', 'description': 'Comprehensive data science course',
            'category': 'Data Science', 'platform': 'Udemy', 'difficulty': 'Intermediate', 'rating': 4.6, 'duration': 18,
            'skills': 'Python, Statistics, Machine Learning', 'enroll_link': 'https://www.udemy.com/data-science-bootcamp', 'price': '₹1699'
        }
    ]

class CourseRecommender:
    def __init__(self):
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
    def load_data(self, courses_data):
        self.df = pd.DataFrame(courses_data)
        return f"Loaded {len(self.df)} courses"
        
    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def preprocess_data(self):
        self.df['description'] = self.df['description'].fillna('')
        self.df['category'] = self.df['category'].fillna('Unknown')
        self.df['skills'] = self.df['skills'].fillna('')
        self.df['difficulty'] = self.df['difficulty'].fillna('Beginner')
        self.df['rating'] = self.df['rating'].fillna(4.0)
        
        self.df['title_clean'] = self.df['title'].apply(self.preprocess_text)
        self.df['description_clean'] = self.df['description'].apply(self.preprocess_text)
        self.df['category_clean'] = self.df['category'].apply(self.preprocess_text)
        self.df['skills_clean'] = self.df['skills'].apply(self.preprocess_text)
        
        self.df['combined_features'] = (
            self.df['title_clean'] + ' ' +
            self.df['description_clean'] + ' ' +
            self.df['category_clean'] + ' ' +
            self.df['skills_clean']
        )
        
    def build_model(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['combined_features'])
        return "Model built successfully"
        
    def get_recommendations(self, user_skills, user_interests, completed_courses=[], top_n=5):
        user_profile = ' '.join(user_skills) + ' ' + ' '.join(user_interests)
        user_profile_clean = self.preprocess_text(user_profile)
        
        user_tfidf = self.tfidf_vectorizer.transform([user_profile_clean])
        cosine_sim = cosine_similarity(user_tfidf, self.tfidf_matrix).flatten()
        
        similarity_scores = list(enumerate(cosine_sim))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
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
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': '', 'skills': [], 'interests': [], 'completed_courses': [], 'bookmarks': []
        }
    if 'recommender' not in st.session_state:
        st.session_state.recommender = CourseRecommender()
        courses_data = get_sample_courses()
        st.session_state.recommender.load_data(courses_data)
        st.session_state.recommender.preprocess_data()
        st.session_state.recommender.build_model()

def display_course_card(course, show_similarity=False):
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(course['title'])
        st.write(f"**Platform:** {course['platform']} | **Category:** {course['category']} | **Difficulty:** {course['difficulty']}")
        st.write(f"**Rating:** ⭐ {course['rating']} | **Duration:** {course['duration']} hours | **Price:** {course['price']}")
        st.write(f"**Skills:** {course['skills']}")
        st.write(course['description'])
    
    with col2:
        if show_similarity:
            st.metric("Match Score", f"{course['similarity_score']*100:.1f}%")
        
        bookmark_key = f"bookmark_{course['course_id']}"
        is_bookmarked = course['course_id'] in st.session_state.user_profile['bookmarks']
        
        if st.button("⭐ Bookmark" if not is_bookmarked else "★ Bookmarked", key=bookmark_key, use_container_width=True):
            if course['course_id'] not in st.session_state.user_profile['bookmarks']:
                st.session_state.user_profile['bookmarks'].append(course['course_id'])
                st.success("Course bookmarked!")
            else:
                st.session_state.user_profile['bookmarks'].remove(course['course_id'])
                st.info("Bookmark removed!")
        
        if st.button("Enroll Now", key=f"enroll_{course['course_id']}", use_container_width=True, type="primary"):
            st.markdown(f"[Go to Course]({course['enroll_link']})")
    
    st.divider()

def get_available_skills(df):
    all_skills = []
    for skills_str in df['skills'].dropna():
        skills = [skill.strip() for skill in skills_str.split(',')]
        all_skills.extend(skills)
    return sorted(list(set(all_skills)))

def main():
    st.set_page_config(page_title="Course Recommendation System", page_icon="🎓", layout="wide")
    initialize_session_state()
    
    with st.sidebar:
        st.title("🎓 CourseRec")
        st.markdown("---")
        page = st.radio("Navigate to:", ["🏠 Home", "👤 User Profile", "💡 Get Recommendations", "📚 Browse Courses", "⭐ Bookmarks"])
        st.markdown("---")
        st.info("AI-powered Course Recommendation System")
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "👤 User Profile":
        show_profile_page()
    elif page == "💡 Get Recommendations":
        show_recommendations_page()
    elif page == "📚 Browse Courses":
        show_browse_page()
    elif page == "⭐ Bookmarks":
        show_bookmarks_page()

def show_home_page():
    st.title("🎓 Welcome to Course Recommendation System")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Discover Your Perfect Learning Path
        This AI system helps you find the best courses based on your skills and interests.
        
        **How it works:**
        1. **Set up your profile** - Tell us about your skills and interests
        2. **Get personalized recommendations** - AI suggests perfect courses
        3. **Browse and bookmark** - Explore and save favorites
        """)
    
    with col2:
        st.markdown("""
        <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px;'>
        <h3 style='color: #1e40af;'>Quick Start</h3>
        <ol>
        <li>Go to <b>User Profile</b></li>
        <li>Add skills & interests</li>
        <li>Get <b>AI Recommendations</b></li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Courses", len(st.session_state.recommender.df))
    with col2: st.metric("Categories", st.session_state.recommender.df['category'].nunique())
    with col3: st.metric("Platforms", st.session_state.recommender.df['platform'].nunique())
    with col4: st.metric("Avg Rating", f"{st.session_state.recommender.df['rating'].mean():.1f} ⭐")

def show_profile_page():
    st.title("👤 User Profile")
    
    with st.form("user_profile_form"):
        st.subheader("Tell us about yourself")
        name = st.text_input("Your Name", value=st.session_state.user_profile['name'])
        
        available_skills = get_available_skills(st.session_state.recommender.df)
        skills = st.multiselect("Your Current Skills", options=available_skills, default=st.session_state.user_profile['skills'])
        interests = st.multiselect("Your Learning Interests", options=available_skills, default=st.session_state.user_profile['interests'])
        
        if st.form_submit_button("💾 Save Profile", use_container_width=True):
            st.session_state.user_profile.update({'name': name, 'skills': skills, 'interests': interests})
            st.success("Profile updated successfully!")
    
    st.markdown("---")
    st.subheader("Current Profile Summary")
    if st.session_state.user_profile['name']:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Name:** {st.session_state.user_profile['name']}")
            st.write(f"**Skills:** {', '.join(st.session_state.user_profile['skills']) or 'None'}")
        with col2:
            st.write(f"**Interests:** {', '.join(st.session_state.user_profile['interests']) or 'None'}")
            st.write(f"**Bookmarks:** {len(st.session_state.user_profile['bookmarks'])}")
    else:
        st.info("Please complete your profile to get personalized recommendations.")

def show_recommendations_page():
    st.title("💡 Personalized Course Recommendations")
    
    if not st.session_state.user_profile['skills'] and not st.session_state.user_profile['interests']:
        st.warning("⚠️ Please complete your profile first!")
        return
    
    col1, col2 = st.columns(2)
    with col1: num_recommendations = st.slider("Number of recommendations", 3, 10, 5)
    with col2: use_interests = st.checkbox("Include interests", value=True)
    
    if st.button("🎯 Get Recommendations", use_container_width=True):
        with st.spinner("Finding the best courses for you..."):
            user_skills = st.session_state.user_profile['skills']
            user_interests = st.session_state.user_profile['interests'] if use_interests else []
            recommendations = st.session_state.recommender.get_recommendations(
                user_skills=user_skills, user_interests=user_interests, top_n=num_recommendations
            )
            st.session_state.last_recommendations = recommendations
    
    if 'last_recommendations' in st.session_state:
        st.subheader(f"🎯 Recommended Courses for You")
        for course in st.session_state.last_recommendations:
            display_course_card(course, show_similarity=True)

def show_browse_page():
    st.title("📚 Browse All Courses")
    
    col1, col2, col3 = st.columns(3)
    with col1: category_filter = st.selectbox("Category", ["All"] + st.session_state.recommender.df['category'].unique().tolist())
    with col2: platform_filter = st.selectbox("Platform", ["All"] + st.session_state.recommender.df['platform'].unique().tolist())
    with col3: difficulty_filter = st.selectbox("Difficulty", ["All"] + st.session_state.recommender.df['difficulty'].unique().tolist())
    
    filtered_df = st.session_state.recommender.df.copy()
    if category_filter != "All": filtered_df = filtered_df[filtered_df['category'] == category_filter]
    if platform_filter != "All": filtered_df = filtered_df[filtered_df['platform'] == platform_filter]
    if difficulty_filter != "All": filtered_df = filtered_df[filtered_df['difficulty'] == difficulty_filter]
    
    st.subheader(f"📖 Found {len(filtered_df)} Courses")
    for _, course in filtered_df.iterrows():
        display_course_card(course.to_dict())

def show_bookmarks_page():
    st.title("⭐ Bookmarked Courses")
    bookmarked_courses = []
    for course_id in st.session_state.user_profile['bookmarks']:
        course = st.session_state.recommender.df[st.session_state.recommender.df['course_id'] == course_id]
        if not course.empty: bookmarked_courses.append(course.iloc[0].to_dict())
    
    if bookmarked_courses:
        st.write(f"You have {len(bookmarked_courses)} bookmarked courses")
        for course in bookmarked_courses: display_course_card(course)
    else:
        st.info("You haven't bookmarked any courses yet!")

if __name__ == "__main__":
    main()
