import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Configuration ---
st.set_page_config(
    page_title="SmartBench AI | Enterprise Resource Optimizer",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Styling (Corporate Dashboard Theme) ---
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #0e1117;
        border: 1px solid #1f2937;
        padding: 18px 24px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600;
        color: #9ca3af !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700;
        color: #60a5fa !important;
    }
    
    /* Headers & Section Titles */
    h1, h2, h3 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-weight: 600;
    }
    
    /* Top Banner Card */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 25px;
        border: 1px solid #1e40af;
        color: #ffffff;
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 1.8rem;
        color: #ffffff;
    }
    .hero-banner p {
        margin-top: 6px;
        margin-bottom: 0;
        color: #93c5fd;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

CSV_FILE = "employees.csv"

def load_data():
    return pd.read_csv(CSV_FILE)

df = load_data()

# ==================== SIDEBAR: RECRUITMENT / ONBOARDING ====================
with st.sidebar:
    st.markdown("### Resource Onboarding")
    st.caption("Add a newly inducted or unassigned engineer to the inventory.")
    
    with st.form(key="add_employee_form", clear_on_submit=True):
        new_name = st.text_input("Full Name", placeholder="e.g. Alex Johnson")
        new_skills = st.text_area(
            "Skills Inventory (Comma-separated)", 
            placeholder="e.g. Python, FastAPI, Docker, PostgreSQL, AWS"
        )
        new_exp = st.number_input(
            "Total Experience (Years)", 
            min_value=0.0, 
            max_value=30.0, 
            value=3.0, 
            step=0.5
        )
        new_status = st.selectbox("Allocation Status", ["Bench", "Allocated"])
        
        submitted = st.form_submit_button("Add Resource", use_container_width=True)
        
        if submitted:
            if not new_name.strip() or not new_skills.strip():
                st.error("Validation Error: Please provide both Name and Skills.")
            else:
                new_entry = pd.DataFrame([{
                    "employee_name": new_name.strip(),
                    "skills_profile": new_skills.strip(),
                    "experience_years": new_exp,
                    "status": new_status
                }])
                new_entry.to_csv(CSV_FILE, mode='a', header=False, index=False)
                st.success(f"Success: Added '{new_name}' to inventory.")
                st.rerun()

    st.divider()
    st.caption("SmartBench Optimizer Engine v2.4 | Enterprise MCA Edition")

# ==================== MAIN DASHBOARD ====================
st.markdown("""
<div class="hero-banner">
    <h1>SmartBench Optimizer</h1>
    <p>AI-Powered Talent Allocation & Semantic Bench Matching Engine</p>
</div>
""", unsafe_allow_html=True)

bench = df[df["status"] == "Bench"].copy()

# Key Metric Cards
m1, m2, m3 = st.columns(3)
m1.metric("Total Headcount", len(df))
m2.metric("Available on Bench", len(bench))
m3.metric("Bench Ratio", f"{(len(bench) / len(df) * 100):.1f}%" if len(df) > 0 else "0%")

# Expander for Bench Roster (In English)
with st.expander("View Active Bench Roster", expanded=False):
    st.dataframe(
        bench[["employee_name", "skills_profile", "experience_years"]].rename(columns={
            "employee_name": "Employee Name",
            "skills_profile": "Skills Profile",
            "experience_years": "Experience (Yrs)"
        }),
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# ==================== NLP MATCHING ENGINE ====================
st.subheader("Semantic Resource Matching")
st.caption("Leverages TF-IDF vectorization and cosine similarity to match bench skill profiles against project requirements.")

req_col, filter_col = st.columns([3, 1])

with req_col:
    project_description = st.text_area(
        "Project Requirements / Scope",
        placeholder="e.g. Seeking a backend developer proficient in Python, FastAPI, Docker, and relational databases like PostgreSQL for microservices architecture.",
        height=110
    )

with filter_col:
    required_experience = st.slider("Target Experience (Yrs)", min_value=1, max_value=15, value=4)
    top_n = st.selectbox("Max Recommendations", [3, 5, 10], index=1)

if st.button("Find Matching Resources", type="primary", use_container_width=True):
    if not project_description.strip():
        st.warning("Please supply project requirements to calculate semantic similarity.")
    elif len(bench) == 0:
        st.error("No resources currently marked as 'Bench'.")
    else:
        with st.spinner("Computing cosine similarity vectors..."):
            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(stop_words='english')
            all_documents = list(bench["skills_profile"]) + [project_description]
            tfidf_matrix = vectorizer.fit_transform(all_documents)
            
            # Semantic Similarity
            similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
            bench["Skill_Match"] = similarity_scores
            
            # Experience Fit Function
            bench["Experience_Score"] = (bench["experience_years"] / required_experience).clip(upper=1.0)
            
            # Composite Scoring (70% Semantic Skill Match, 30% Experience Proximity)
            bench["Composite_Score"] = (0.7 * bench["Skill_Match"]) + (0.3 * bench["Experience_Score"])
            
            results = bench.sort_values(by="Composite_Score", ascending=False).head(top_n)
            
            st.markdown("### Top Matched Candidates")
            
            display_df = results[[
                "employee_name", 
                "skills_profile", 
                "experience_years", 
                "Skill_Match", 
                "Composite_Score"
            ]].copy()
            
            display_df["Skill_Match"] = (display_df["Skill_Match"] * 100).round(1).astype(str) + "%"
            display_df["Composite_Score"] = (display_df["Composite_Score"] * 100).round(1).astype(str) + "%"
            
            st.dataframe(
                display_df.rename(columns={
                    "employee_name": "Employee Name",
                    "skills_profile": "Skills Profile",
                    "experience_years": "Experience (Yrs)",
                    "Skill_Match": "Skill Match",
                    "Composite_Score": "Overall Fit"
                }),
                use_container_width=True,
                hide_index=True
            )
