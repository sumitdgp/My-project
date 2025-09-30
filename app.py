import streamlit as st
import pandas as pd
import plotly.express as px
from utils import *

# Page configuration
st.set_page_config(
    page_title="Course Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Load and initialize recommender
    if st.session_state.recommender.df is None:
        with st.spinner("Loading course data and building recommendation model..."):
            st.session_state.recommender.load_data('courses.csv')
            st.session_state.recommender.preprocess_data()
            st.session_state.recommender.build_model()
    
    # Sidebar
    with st.sidebar:
        st.title("🎓 CourseRec")
        st.markdown("---")
        
        # Navigation
        page = st.radio("Navigate to:", 
                       ["🏠 Home", "👤 User Profile", "💡 Get Recommendations", 
                        "📚 Browse Courses", "⭐ Bookmarks", "📊 Analytics"])
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This Course Recommendation System uses AI to suggest the best learning courses based on your skills, interests, and learning goals.
        """)
    
    # Page routing
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
    elif page == "📊 Analytics":
        show_analytics_page()

def show_home_page():
    """Display the home page"""
    st.title("🎓 Welcome to Course Recommendation System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Discover Your Perfect Learning Path
        
        This intelligent system helps you find the best courses based on:
        - Your **current skills** and knowledge level
        - Your **career interests** and goals  
        - Your **learning preferences** and style
        - **AI-powered** recommendations using machine learning
        
        ### How it works:
        1. **Set up your profile** - Tell us about your skills and interests
        2. **Get personalized recommendations** - Our AI suggests perfect courses
        3. **Browse and bookmark** - Explore courses and save your favorites
        4. **Track your progress** - Mark courses as completed
        
        ### Get Started:
        Use the sidebar to navigate through different sections and start your learning journey!
        """)
    
    with col2:
        st.image("https://cdn.pixabay.com/photo/2018/09/27/09/22/artificial-intelligence-3706562_1280.jpg", 
                use_column_width=True)
    
    # Quick stats
    st.markdown("---")
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
    st.title("👤 User Profile")
    
    with st.form("user_profile_form"):
        st.subheader("Tell us about yourself")
        
        name = st.text_input("Your Name", 
                           value=st.session_state.user_profile['name'],
                           placeholder="Enter your name")
        
        # Skills selection
        available_skills = get_available_skills(st.session_state.recommender.df)
        skills = st.multiselect(
            "Your Current Skills",
            options=available_skills,
            default=st.session_state.user_profile['skills'],
            help="Select skills you already have"
        )
        
        # Interests selection
        interests = st.multiselect(
            "Your Learning Interests",
            options=available_skills,
            default=st.session_state.user_profile['interests'],
            help="Select skills/topics you want to learn"
        )
        
        # Career goals
        career_goals = st.text_area(
            "Career Goals",
            value=st.session_state.user_profile.get('career_goals', ''),
            placeholder="Describe your career goals and what you want to achieve..."
        )
        
        # Completed courses
        completed_courses = st.multiselect(
            "Completed Courses",
            options=st.session_state.recommender.df['title'].tolist(),
            default=st.session_state.user_profile['completed_courses'],
            help="Select courses you have already completed"
        )
        
        if st.form_submit_button("💾 Save Profile", use_container_width=True):
            st.session_state.user_profile.update({
                'name': name,
                'skills': skills,
                'interests': interests,
                'career_goals': career_goals,
                'completed_courses': completed_courses
            })
            st.success("Profile updated successfully!")
    
    # Display current profile
    st.markdown("---")
    st.subheader("Current Profile Summary")
    
    if st.session_state.user_profile['name']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Name:** {st.session_state.user_profile['name']}")
            st.write(f"**Skills:** {', '.join(st.session_state.user_profile['skills']) or 'None'}")
            st.write(f"**Interests:** {', '.join(st.session_state.user_profile['interests']) or 'None'}")
        
        with col2:
            st.write(f"**Completed Courses:** {len(st.session_state.user_profile['completed_courses'])}")
            st.write(f"**Bookmarked Courses:** {len(st.session_state.user_profile['bookmarks'])}")
            
            if st.session_state.user_profile.get('career_goals'):
                st.write(f"**Career Goals:** {st.session_state.user_profile['career_goals']}")
    else:
        st.info("Please complete your profile to get personalized recommendations.")

def show_recommendations_page():
    """Display recommendations page"""
    st.title("💡 Personalized Course Recommendations")
    
    if not st.session_state.user_profile['skills'] and not st.session_state.user_profile['interests']:
        st.warning("⚠️ Please complete your profile first to get personalized recommendations!")
        if st.button("Go to Profile Page"):
            st.session_state.page = 'profile'
        return
    
    # Recommendation options
    col1, col2 = st.columns(2)
    
    with col1:
        num_recommendations = st.slider("Number of recommendations", 5, 20, 10)
    
    with col2:
        use_interests = st.checkbox("Include interests in recommendations", value=True)
    
    # Get recommendations
    if st.button("🎯 Get Recommendations", use_container_width=True):
        with st.spinner("Finding the best courses for you..."):
            user_skills = st.session_state.user_profile['skills']
            user_interests = st.session_state.user_profile['interests'] if use_interests else []
            completed_courses = st.session_state.user_profile['completed_courses']
            
            recommendations = st.session_state.recommender.get_recommendations(
                user_skills=user_skills,
                user_interests=user_interests,
                completed_courses=completed_courses,
                top_n=num_recommendations
            )
            
            st.session_state.last_recommendations = recommendations
    
    # Display recommendations
    if 'last_recommendations' in st.session_state:
        st.subheader(f"🎯 Recommended Courses for You")
        st.write(f"Found {len(st.session_state.last_recommendations)} courses matching your profile")
        
        for course in st.session_state.last_recommendations:
            display_course_card(course, show_similarity=True)

def show_browse_page():
    """Display course browsing page"""
    st.title("📚 Browse All Courses")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        category_filter = st.selectbox(
            "Category",
            ["All"] + st.session_state.recommender.df['category'].unique().tolist()
        )
    
    with col2:
        platform_filter = st.selectbox(
            "Platform",
            ["All"] + st.session_state.recommender.df['platform'].unique().tolist()
        )
    
    with col3:
        difficulty_filter = st.selectbox(
            "Difficulty",
            ["All"] + st.session_state.recommender.df['difficulty'].unique().tolist()
        )
    
    with col4:
        min_rating = st.slider("Minimum Rating", 3.0, 5.0, 3.5, 0.1)
    
    # Apply filters
    filtered_df = st.session_state.recommender.df.copy()
    
    if category_filter != "All":
        filtered_df = filtered_df[filtered_df['category'] == category_filter]
    
    if platform_filter != "All":
        filtered_df = filtered_df[filtered_df['platform'] == platform_filter]
    
    if difficulty_filter != "All":
        filtered_df = filtered_df[filtered_df['difficulty'] == difficulty_filter]
    
    filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
    
    # Display courses
    st.subheader(f"📖 Found {len(filtered_df)} Courses")
    
    for _, course in filtered_df.iterrows():
        display_course_card(course.to_dict())

def show_bookmarks_page():
    """Display bookmarked courses"""
    st.title("⭐ Bookmarked Courses")
    
    bookmarked_courses = []
    for course_id in st.session_state.user_profile['bookmarks']:
        course = st.session_state.recommender.df[
            st.session_state.recommender.df['course_id'] == course_id
        ]
        if not course.empty:
            bookmarked_courses.append(course.iloc[0].to_dict())
    
    if bookmarked_courses:
        st.write(f"You have {len(bookmarked_courses)} bookmarked courses")
        
        for course in bookmarked_courses:
            display_course_card(course)
    else:
        st.info("You haven't bookmarked any courses yet. Browse courses and click the ⭐ button to bookmark them!")

def show_analytics_page():
    """Display analytics and insights"""
    st.title("📊 Learning Analytics")
    
    if not st.session_state.user_profile['name']:
        st.warning("Complete your profile to see personalized analytics!")
        return
    
    # User statistics
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
            category_counts = bookmarked_df['category'].value_counts()
            fig1 = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Bookmarked Courses by Category"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Difficulty distribution
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

if __name__ == "__main__":
    main()
