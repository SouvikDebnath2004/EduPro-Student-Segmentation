import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="EduPro Personalization Engine",
    page_icon="🎓",
    layout="wide"
)

# ----------------- DATA LOADING & PREPARATION -----------------
@st.cache_data
def load_data():
    learner_df = pd.read_csv("learner_features.csv", index_col="UserID")
    master_data = pd.read_csv("master_df.csv")
    courses_data = pd.read_csv("courses_clean.csv")
    
    # Compute content similarity
    courses_data['metadata'] = (
        courses_data['CourseCategory'].fillna('') + " " +
        courses_data['CourseType'].fillna('') + " " +
        courses_data['CourseLevel'].fillna('')
    )
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(courses_data['metadata'])
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    cid_to_idx = {cid: idx for idx, cid in enumerate(courses_data['CourseID'])}
    
    return learner_df, master_data, courses_data, sim_matrix, cid_to_idx

learner_features, master_df, courses_df, course_sim_matrix, course_id_to_idx = load_data()

# ----------------- RECOMMENDATION ENGINE -----------------
def recommend_courses(user_id, top_n=5, alpha=0.5, beta=0.3, gamma=0.2, category_filter="All", level_filter="All"):
    user_cluster = learner_features.loc[user_id, 'Cluster']
    enrolled_courses = master_df[master_df['UserID'] == user_id]['CourseID'].unique()
    
    cluster_users = learner_features[learner_features['Cluster'] == user_cluster].index
    cluster_transactions = master_df[master_df['UserID'].isin(cluster_users)]
    cluster_pop = cluster_transactions['CourseID'].value_counts().to_dict()
    max_pop = max(cluster_pop.values()) if cluster_pop else 1
    
    pool = []
    for _, row in courses_df.iterrows():
        cid = row['CourseID']
        if cid in enrolled_courses:
            continue
            
        # UI Filters
        if category_filter != "All" and row['CourseCategory'] != category_filter:
            continue
        if level_filter != "All" and row['CourseLevel'] != level_filter:
            continue
            
        c_idx = course_id_to_idx[cid]
        
        # Content score
        if len(enrolled_courses) > 0:
            past_indices = [course_id_to_idx[e] for e in enrolled_courses if e in course_id_to_idx]
            content_score = np.mean([course_sim_matrix[c_idx][p] for p in past_indices]) if past_indices else 0.0
        else:
            content_score = 0.0
            
        rating_score = (row['CourseRating'] / 5.0) if pd.notnull(row['CourseRating']) else 0.5
        pop_score = cluster_pop.get(cid, 0) / max_pop
        
        match_score = (alpha * content_score) + (beta * rating_score) + (gamma * pop_score)
        
        pool.append({
            'CourseID': cid,
            'CourseName': row['CourseName'],
            'Category': row['CourseCategory'],
            'Level': row['CourseLevel'],
            'Type': row['CourseType'],
            'Price ($)': row['CoursePrice'],
            'Rating': row['CourseRating'],
            'Match Score': round(match_score, 4)
        })
        
    rec_df = pd.DataFrame(pool).sort_values(by='Match Score', ascending=False)
    return rec_df.head(top_n)

# ----------------- UI LAYOUT -----------------
st.title("🎓 EduPro: Student Segmentation & Personalized Recommendation Engine")
st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "👤 Learner Profile & Recommendations", 
    "📊 Cluster Analytics Dashboard", 
    "📈 Segment Comparison Panel"
])

# TAB 1: USER EXPLORER & RECOMMENDATIONS
with tab1:
    col_user_select, col_filters = st.columns([1, 2])
    
    with col_user_select:
        user_list = learner_features.index.tolist()
        selected_user = st.selectbox("🔍 Select Learner (UserID):", user_list, index=0)
        
    with col_filters:
        f_col1, f_col2, f_col3 = st.columns(3)
        categories = ["All"] + sorted(courses_df['CourseCategory'].dropna().unique().tolist())
        levels = ["All"] + sorted(courses_df['CourseLevel'].dropna().unique().tolist())
        
        cat_filter = f_col1.selectbox("Filter by Category:", categories)
        lvl_filter = f_col2.selectbox("Filter by Level:", levels)
        top_k = f_col3.slider("Top Recommendations:", min_value=3, max_value=10, value=5)

    u_data = learner_features.loc[selected_user]
    
    # Profile Overview Cards
    st.markdown("### 📌 Learner Profile Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Segment", u_data["Segment"])
    m2.metric("Courses Enrolled", int(u_data["total_courses"]))
    m3.metric("Total Spending", f"${u_data['total_spending']:.2f}")
    m4.metric("Avg Course Rating", f"{u_data['avg_course_rating']:.2f}")
    m5.metric("Diversity Score", int(u_data["diversity_score"]))

    st.markdown("---")
    
    # Recommendations Table
    st.markdown(f"### 🎯 Recommended Learning Path for `{selected_user}`")
    recs = recommend_courses(
        selected_user, 
        top_n=top_k, 
        category_filter=cat_filter, 
        level_filter=lvl_filter
    )
    
    if not recs.empty:
        st.dataframe(recs, use_container_width=True, hide_index=True)
    else:
        st.info("No courses match the selected filters. Try broadening your criteria.")

# TAB 2: CLUSTER ANALYTICS
with tab2:
    st.subheader("Cluster Visualization & Distribution")
    
    c_col1, c_col2 = st.columns(2)
    
    with c_col1:
        fig_scatter = px.scatter(
            learner_features.reset_index(),
            x="total_spending",
            y="total_courses",
            color="Segment",
            size="diversity_score",
            hover_data=["UserID", "user_age"],
            title="Learner Segments: Spending vs Total Enrollments",
            labels={"total_spending": "Total Spend ($)", "total_courses": "Total Courses"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with c_col2:
        seg_counts = learner_features['Segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Learner Count']
        fig_pie = px.pie(
            seg_counts, 
            names='Segment', 
            values='Learner Count',
            title='Learner Distribution by Segment',
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# TAB 3: SEGMENT COMPARISON PANEL
with tab3:
    st.subheader("Comparative Analysis Across Learner Segments")
    
    summary_table = learner_features.groupby("Segment").agg({
        "total_courses": "mean",
        "total_spending": "mean",
        "avg_spending": "mean",
        "diversity_score": "mean",
        "avg_course_rating": "mean",
        "user_age": "mean"
    }).round(2).reset_index()
    
    summary_table.columns = [
        "Segment", "Avg Total Courses", "Avg Total Spend ($)", 
        "Avg Spend/Course ($)", "Avg Category Diversity", 
        "Avg Course Rating", "Avg Learner Age"
    ]
    st.table(summary_table)
    
    # Behavioral bar comparison
    fig_bar = px.bar(
        summary_table,
        x="Segment",
        y=["Avg Total Spend ($)", "Avg Spend/Course ($)"],
        barmode="group",
        title="Spending Dynamics Across Segments"
    )
    st.plotly_chart(fig_bar, use_container_width=True)