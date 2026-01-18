"""
Streamlit Frontend for Behavioral Model for Adaptive Learning System
"""

import streamlit as st
import requests
import json
import pandas as pd
from typing import Dict, Any, List, Optional
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Question Quest - Adaptive Learning System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = st.sidebar.text_input(
    "API Base URL",
    value="http://localhost:9010",
    help="Enter the base URL of the FastAPI backend"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("📚 Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    [
        "🏠 Home",
        "🎯 Competency Prediction",
        "❓ Question Recommendations",
        "🛤️ Learning Path Generation",
        "📊 Gap Analysis",
        "🔍 Misconception Detection",
        "🌐 Knowledge Graph Explorer"
    ]
)

# Helper functions
def check_api_health() -> bool:
    """Check if API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def make_api_request(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Optional[Dict]:
    """Make API request and handle errors."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=30)
        else:
            response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_BASE_URL}. Please ensure the API server is running.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Home Page
if page == "🏠 Home":
    st.markdown('<div class="main-header">📚 Question Quest - Adaptive Learning System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Adaptive Learning System!
    
    This system helps teachers and educators:
    - **Predict** student competency for questions
    - **Recommend** personalized questions based on student profiles
    - **Generate** optimal learning paths
    - **Analyze** competency gaps and identify root causes
    - **Detect** misconceptions from student responses
    - **Explore** the knowledge graph of educational concepts
    
    ### Getting Started
    
    1. Ensure the FastAPI backend is running (default: http://localhost:9010)
    2. Use the sidebar to navigate to different features
    3. Enter the required information in the forms
    4. View results and insights
    """)
    
    # API Health Check
    st.markdown("### 🔌 API Status")
    if check_api_health():
        st.success("✅ API is running and accessible")
    else:
        st.error(f"❌ Cannot connect to API at {API_BASE_URL}")
        st.info("Please start the API server using: `python -m src.api.app` or `uvicorn src.api.app:app --host 0.0.0.0 --port 9010`")

# Competency Prediction Page
elif page == "🎯 Competency Prediction":
    st.title("🎯 Competency Prediction")
    st.markdown("Predict student competency scores for questions")
    
    with st.form("competency_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            question_id = st.text_input("Question ID (optional)")
            question_text = st.text_area("Question Text (optional)", height=100)
            subject = st.selectbox("Subject", ["Mathematics", "Science", "English", "History", "Biology", "Chemistry", "Physics"])
            grade = st.selectbox("Grade", ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
            chapter = st.text_input("Chapter")
        
        with col2:
            learning_objective = st.text_input("Learning Objective")
            learning_unit = st.text_input("Learning Unit")
            bloom_tag = st.selectbox("Bloom's Taxonomy Tag", 
                                    ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"])
            three_pl = st.number_input("3PL Parameter", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        
        submitted = st.form_submit_button("Predict Competency", type="primary")
        
        if submitted:
            request_data = {
                "question_id": question_id if question_id else None,
                "question_text": question_text if question_text else None,
                "subject": subject,
                "grade": grade,
                "chapter": chapter if chapter else None,
                "learning_objective": learning_objective if learning_objective else None,
                "learning_unit": learning_unit if learning_unit else None,
                "bloom_tag": bloom_tag,
                "three_pl": three_pl
            }
            
            result = make_api_request("/predict/competency", method="POST", data=request_data)
            
            if result:
                competency = result.get("predicted_competency", 0)
                confidence = result.get("confidence")
                
                st.success("✅ Prediction successful!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Competency", f"{competency:.3f}")
                with col2:
                    if confidence:
                        st.metric("Confidence", f"{confidence:.3f}")
                
                # Visualize competency
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = competency * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Competency Score (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 60
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

# Question Recommendations Page
elif page == "❓ Question Recommendations":
    st.title("❓ Question Recommendations")
    st.markdown("Get personalized question recommendations for students")
    
    with st.form("question_recommendation_form"):
        st.subheader("Student Profile")
        
        col1, col2 = st.columns(2)
        with col1:
            avg_competency = st.number_input("Average Competency", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
            num_recommendations = st.number_input("Number of Recommendations", min_value=1, max_value=50, value=10)
        
        with col2:
            st.markdown("**Competency by Subject**")
            math_comp = st.number_input("Mathematics", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            science_comp = st.number_input("Science", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
        
        st.markdown("**Weak Areas**")
        weak_areas_input = st.text_input("Enter weak areas (comma-separated)", value="Fractions, Integers")
        weak_areas = [area.strip() for area in weak_areas_input.split(",")] if weak_areas_input else []
        
        target_objectives_input = st.text_input("Target Learning Objectives (optional, comma-separated)")
        target_objectives = [obj.strip() for obj in target_objectives_input.split(",")] if target_objectives_input else None
        
        submitted = st.form_submit_button("Get Recommendations", type="primary")
        
        if submitted:
            student_profile = {
                "avg_competency": avg_competency,
                "competency_by_subject": {
                    "Mathematics": math_comp,
                    "Science": science_comp
                },
                "weak_areas": {
                    "Mathematics": weak_areas
                }
            }
            
            request_data = {
                "student_profile": student_profile,
                "target_objectives": target_objectives,
                "num_recommendations": num_recommendations
            }
            
            result = make_api_request("/recommend/questions", method="POST", data=request_data)
            
            if result:
                recommendations = result.get("recommendations", [])
                reasoning = result.get("reasoning")
                
                st.success(f"✅ Found {len(recommendations)} recommendations!")
                
                if reasoning:
                    st.info(f"💡 Reasoning: {reasoning}")
                
                if recommendations:
                    # Display recommendations in a table
                    df = pd.DataFrame(recommendations)
                    st.dataframe(df, use_container_width=True)
                    
                    # Visualize recommendations
                    if "score" in df.columns or "competency" in df.columns:
                        score_col = "score" if "score" in df.columns else "competency"
                        fig = px.bar(df.head(10), x=score_col, y=df.index[:10], 
                                    orientation='h', title="Top Recommendations")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No recommendations found. Try adjusting the student profile.")

# Learning Path Generation Page
elif page == "🛤️ Learning Path Generation":
    st.title("🛤️ Learning Path Generation")
    st.markdown("Generate optimal learning paths for students")
    
    with st.form("learning_path_form"):
        st.subheader("Student Profile")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Competency by Chapter**")
            fractions_comp = st.number_input("Fractions", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
            integers_comp = st.number_input("Integers", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
            algebra_comp = st.number_input("Algebra", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        
        with col2:
            target_concept = st.text_input("Target Concept (optional)")
            target_chapter = st.text_input("Target Chapter (optional)", value="Algebra")
        
        submitted = st.form_submit_button("Generate Learning Path", type="primary")
        
        if submitted:
            student_profile = {
                "competency_by_chapter": {
                    "Fractions": fractions_comp,
                    "Integers": integers_comp,
                    "Algebra": algebra_comp
                }
            }
            
            request_data = {
                "student_profile": student_profile,
                "target_concept": target_concept if target_concept else None,
                "target_chapter": target_chapter if target_chapter else None
            }
            
            result = make_api_request("/generate/path", method="POST", data=request_data)
            
            if result:
                path = result.get("path", [])
                total_steps = result.get("total_steps", 0)
                
                st.success(f"✅ Generated learning path with {total_steps} steps!")
                
                if path:
                    st.subheader("Learning Path Steps")
                    for i, step in enumerate(path, 1):
                        with st.expander(f"Step {i}: {step.get('concept', step.get('chapter', 'Unknown'))}"):
                            st.json(step)
                    
                    # Visualize path
                    if isinstance(path[0], dict) and "concept" in path[0]:
                        concepts = [step.get("concept", "Unknown") for step in path]
                        fig = go.Figure(data=go.Scatter(
                            x=list(range(len(concepts))),
                            y=[1] * len(concepts),
                            mode='lines+markers+text',
                            text=concepts,
                            textposition="top center",
                            marker=dict(size=15, color='blue'),
                            line=dict(color='gray', width=2)
                        ))
                        fig.update_layout(
                            title="Learning Path Visualization",
                            xaxis_title="Step",
                            yaxis=dict(visible=False),
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)

# Gap Analysis Page
elif page == "📊 Gap Analysis":
    st.title("📊 Gap Analysis")
    st.markdown("Analyze competency gaps and identify root causes")
    
    with st.form("gap_analysis_form"):
        st.subheader("Student Responses")
        
        # Allow manual entry or file upload
        input_method = st.radio("Input Method", ["Manual Entry", "JSON Upload"])
        
        if input_method == "Manual Entry":
            num_responses = st.number_input("Number of Responses", min_value=1, max_value=20, value=3)
            student_responses = []
            
            for i in range(num_responses):
                with st.expander(f"Response {i+1}"):
                    response = {
                        "question_id": st.text_input(f"Question ID {i+1}", key=f"qid_{i}"),
                        "competency_score": st.number_input(f"Competency Score {i+1}", 
                                                          min_value=0.0, max_value=1.0, 
                                                          value=0.5, step=0.01, key=f"comp_{i}"),
                        "chapter": st.text_input(f"Chapter {i+1}", key=f"chap_{i}"),
                        "subject": st.text_input(f"Subject {i+1}", key=f"subj_{i}")
                    }
                    student_responses.append(response)
        else:
            uploaded_file = st.file_uploader("Upload JSON file", type=["json"])
            if uploaded_file:
                student_responses = json.load(uploaded_file)
            else:
                student_responses = []
        
        competency_threshold = st.number_input("Competency Threshold", 
                                              min_value=0.0, max_value=1.0, 
                                              value=0.6, step=0.01)
        
        submitted = st.form_submit_button("Analyze Gaps", type="primary")
        
        if submitted:
            if not student_responses:
                st.error("Please provide at least one student response")
            else:
                request_data = {
                    "student_responses": student_responses,
                    "competency_threshold": competency_threshold
                }
                
                result = make_api_request("/analyze/gaps", method="POST", data=request_data)
                
                if result:
                    gaps = result.get("gaps", {})
                    root_causes = result.get("root_causes", [])
                    remediation = result.get("remediation_strategy", {})
                    
                    st.success("✅ Gap analysis completed!")
                    
                    # Display gaps
                    st.subheader("Identified Gaps")
                    st.json(gaps)
                    
                    # Display root causes
                    if root_causes:
                        st.subheader("Root Causes")
                        for i, cause in enumerate(root_causes, 1):
                            st.write(f"{i}. {cause}")
                    
                    # Display remediation strategy
                    if remediation:
                        st.subheader("Remediation Strategy")
                        st.json(remediation)
                    
                    # Visualize gaps
                    if isinstance(gaps, dict) and "gaps_by_chapter" in gaps:
                        gaps_data = gaps["gaps_by_chapter"]
                        if gaps_data:
                            df = pd.DataFrame(list(gaps_data.items()), columns=["Chapter", "Gap Score"])
                            fig = px.bar(df, x="Chapter", y="Gap Score", 
                                       title="Competency Gaps by Chapter")
                            st.plotly_chart(fig, use_container_width=True)

# Misconception Detection Page
elif page == "🔍 Misconception Detection":
    st.title("🔍 Misconception Detection")
    st.markdown("Detect misconceptions from student responses")
    
    with st.form("misconception_detection_form"):
        st.subheader("Student Responses")
        
        input_method = st.radio("Input Method", ["Manual Entry", "JSON Upload"])
        
        if input_method == "Manual Entry":
            num_responses = st.number_input("Number of Responses", min_value=1, max_value=20, value=2)
            student_responses = []
            
            for i in range(num_responses):
                with st.expander(f"Response {i+1}"):
                    response = {
                        "question_id": st.text_input(f"Question ID {i+1}", key=f"mc_qid_{i}"),
                        "competency_score": st.number_input(f"Competency Score {i+1}", 
                                                          min_value=0.0, max_value=1.0, 
                                                          value=0.3, step=0.01, key=f"mc_comp_{i}"),
                        "question_text": st.text_area(f"Question Text {i+1}", 
                                                     height=80, key=f"mc_text_{i}"),
                        "student_response": st.text_area(f"Student Response {i+1}", 
                                                        height=80, key=f"mc_resp_{i}")
                    }
                    student_responses.append(response)
        else:
            uploaded_file = st.file_uploader("Upload JSON file", type=["json"])
            if uploaded_file:
                student_responses = json.load(uploaded_file)
            else:
                student_responses = []
        
        submitted = st.form_submit_button("Detect Misconceptions", type="primary")
        
        if submitted:
            if not student_responses:
                st.error("Please provide at least one student response")
            else:
                request_data = {
                    "student_responses": student_responses
                }
                
                result = make_api_request("/detect/misconceptions", method="POST", data=request_data)
                
                if result:
                    misconceptions = result.get("misconceptions", [])
                    total_identified = result.get("total_identified", 0)
                    remediation = result.get("remediation", {})
                    
                    st.success(f"✅ Detected {total_identified} misconception(s)!")
                    
                    if misconceptions:
                        st.subheader("Detected Misconceptions")
                        for i, mc in enumerate(misconceptions, 1):
                            with st.expander(f"Misconception {i}: {mc.get('misconception_id', 'Unknown')}"):
                                st.json(mc)
                        
                        # Visualize misconceptions
                        if len(misconceptions) > 0:
                            mc_data = []
                            for mc in misconceptions:
                                mc_data.append({
                                    "Misconception": mc.get("misconception_id", "Unknown"),
                                    "Confidence": mc.get("confidence", 0.5),
                                    "Severity": mc.get("severity", "Medium")
                                })
                            df = pd.DataFrame(mc_data)
                            fig = px.bar(df, x="Misconception", y="Confidence", 
                                       color="Severity", title="Misconception Analysis")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    if remediation:
                        st.subheader("Remediation Recommendations")
                        st.json(remediation)
                else:
                    st.warning("No misconceptions detected or API error occurred.")

# Knowledge Graph Explorer Page
elif page == "🌐 Knowledge Graph Explorer":
    st.title("🌐 Knowledge Graph Explorer")
    st.markdown("Explore the knowledge graph of educational concepts")
    
    explorer_tab = st.radio("Explorer Mode", 
                           ["Concept Details", "Prerequisites", "Learning Path Between Concepts"],
                           horizontal=True)
    
    if explorer_tab == "Concept Details":
        st.subheader("Get Concept Information")
        concept_id = st.text_input("Concept ID", value="concept_fractions")
        
        if st.button("Get Concept Details", type="primary"):
            result = make_api_request(f"/graph/concepts/{concept_id}")
            
            if result:
                concept_details = result.get("concept_details", {})
                st.success("✅ Concept details retrieved!")
                st.json(concept_details)
                
                # Display misconceptions if available
                if "misconceptions" in concept_details:
                    st.subheader("Associated Misconceptions")
                    st.write(concept_details["misconceptions"])
    
    elif explorer_tab == "Prerequisites":
        st.subheader("Find Prerequisites")
        concept_id = st.text_input("Concept ID", value="concept_algebra")
        max_depth = st.number_input("Maximum Depth", min_value=1, max_value=10, value=5)
        
        if st.button("Find Prerequisites", type="primary"):
            result = make_api_request(f"/graph/prerequisites/{concept_id}?max_depth={max_depth}")
            
            if result:
                prerequisites = result.get("prerequisites", [])
                st.success(f"✅ Found {len(prerequisites)} prerequisite(s)!")
                
                if prerequisites:
                    for i, prereq in enumerate(prerequisites, 1):
                        st.write(f"{i}. {prereq}")
                    
                    # Visualize prerequisites
                    fig = go.Figure(data=go.Scatter(
                        x=list(range(len(prerequisites))),
                        y=[1] * len(prerequisites),
                        mode='markers+text',
                        text=prerequisites,
                        textposition="top center",
                        marker=dict(size=15, color='green')
                    ))
                    fig.update_layout(
                        title="Prerequisites Chain",
                        xaxis_title="Order",
                        yaxis=dict(visible=False),
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No prerequisites found for this concept.")
    
    elif explorer_tab == "Learning Path Between Concepts":
        st.subheader("Find Learning Path")
        col1, col2 = st.columns(2)
        with col1:
            from_concept = st.text_input("From Concept", value="concept_fractions")
        with col2:
            to_concept = st.text_input("To Concept", value="concept_algebra")
        
        respect_prerequisites = st.checkbox("Respect Prerequisites", value=True)
        
        if st.button("Find Learning Path", type="primary"):
            result = make_api_request(
                f"/graph/path/{from_concept}/{to_concept}?respect_prerequisites={str(respect_prerequisites).lower()}"
            )
            
            if result:
                path = result.get("path", [])
                path_length = result.get("path_length", 0)
                
                st.success(f"✅ Found learning path with {path_length} step(s)!")
                
                if path:
                    st.subheader("Path Steps")
                    for i, concept in enumerate(path, 1):
                        st.write(f"{i}. {concept}")
                    
                    # Visualize path
                    fig = go.Figure(data=go.Scatter(
                        x=list(range(len(path))),
                        y=[1] * len(path),
                        mode='lines+markers+text',
                        text=path,
                        textposition="top center",
                        marker=dict(size=15, color='purple'),
                        line=dict(color='purple', width=2)
                    ))
                    fig.update_layout(
                        title="Learning Path Visualization",
                        xaxis_title="Step",
                        yaxis=dict(visible=False),
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No path found between these concepts.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 About")
st.sidebar.info("""
This is the frontend interface for the Behavioral Model for Adaptive Learning System.

**Features:**
- Competency Prediction
- Question Recommendations
- Learning Path Generation
- Gap Analysis
- Misconception Detection
- Knowledge Graph Exploration
""")
