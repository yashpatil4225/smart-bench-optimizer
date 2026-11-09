import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Smart Bench Optimizer", layout="wide")
st.title("IT Smart-Bench & Resource Allocation Optimizer")

# 1. Load Data
CSV_FILE = "employees.csv"

def load_data():
    return pd.read_csv(CSV_FILE)

df = load_data()

# ----------------- SIDEBAR: ADD NEW EMPLOYEE -----------------
st.sidebar.header("Add New Employee")
with st.sidebar.form(key="add_employee_form", clear_on_submit=True):
    new_name = st.text_input("Employee Name")
    new_skills = st.text_area("Skills Profile (comma separated)", placeholder="e.g. Python, Docker, FastApi, PostgreSQL")
    new_exp = st.number_input("Experience (Years)", min_value=0.0, max_value=30.0, value=2.0, step=0.5)
    new_status = st.selectbox("Current Status", ["Bench", "Allocated"])
    
    submit_btn = st.form_submit_button("Add to System")
    
    if submit_btn:
        if not new_name.strip() or not new_skills.strip():
            st.sidebar.error("Name आणि Skills भरणे आवश्यक आहे!")
        else:
            new_data = pd.DataFrame([{
                "employee_name": new_name.strip(),
                "skills_profile": new_skills.strip(),
                "experience_years": new_exp,
                "status": new_status
            }])
            # Append directly to CSV
            new_data.to_csv(CSV_FILE, mode='a', header=False, index=False)
            st.sidebar.success(f"{new_name} यशस्वीरीत्या सेव्ह झाला!")
            st.rerun()

# ----------------- MAIN PANEL: BENCH OVERVIEW -----------------
bench = df[df["status"] == "Bench"].copy()

col1, col2 = st.columns(2)
col1.metric("Total Employees", len(df))
col2.metric("Available on Bench", len(bench))

with st.expander("सर्व Bench Employees ची यादी पहा"):
    st.dataframe(bench[["employee_name", "skills_profile", "experience_years"]], use_container_width=True)

# ----------------- NLP RESOURCE MATCHING -----------------
st.subheader("Project Requirements Matching")
project_description = st.text_area("Project Requirements / Description", placeholder="e.g. Looking for a Python backend engineer with AWS and SQL database optimization skills")
required_experience = st.slider("Target Experience (years)", min_value=1, max_value=15, value=3)

if st.button("Find Best-Fit Resources"):
    if not project_description.strip():
        st.warning("कृपया project requirements टाका.")
    elif len(bench) == 0:
        st.error("सध्या बेंचवर एकही employee उपलब्ध नाही!")
    else:
        # TF-IDF + Cosine Similarity
        vectorizer = TfidfVectorizer()
        all_text = list(bench["skills_profile"]) + [project_description]
        tfidf_matrix = vectorizer.fit_transform(all_text)
        
        # Calculate skill similarity
        scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        bench["Skill_Match"] = scores
        
        # Experience fit calculation
        bench["Experience_Score"] = (bench["experience_years"] / required_experience).clip(upper=1.0)
        
        # Hybrid Ranking (70% Skill + 30% Experience)
        bench["Final_Score"] = (0.7 * bench["Skill_Match"]) + (0.3 * bench["Experience_Score"])
        
        # Sort Top 5
        result = bench.sort_values(by="Final_Score", ascending=False).head(5)
        
        st.success("Top Matched Profiles:")
        st.dataframe(
            result[["employee_name", "skills_profile", "experience_years", "Skill_Match", "Final_Score"]].round(2),
            use_container_width=True
        )