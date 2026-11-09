import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# --- Page Configuration ---
st.set_page_config(
    page_title="SmartBench AI | Enterprise Resource Optimizer",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Styling ---
st.markdown("""
<style>
    .metric-card {
        background-color: #111827;
        border: 1px solid #374151;
        padding: 16px 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #ffffff !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-number {
        font-size: 2.2rem;
        font-weight: 800;
        color: #60a5fa !important;
        line-height: 1.1;
    }
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

# Session State 
if "df" not in st.session_state:
    st.session_state.df = pd.read_csv(CSV_FILE)

df = st.session_state.df
# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### Resource Onboarding")
    st.caption("Add an unassigned or new engineer to the inventory.")
    
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
                
                # Session State मधील DataFrame मध्ये नवीन रो जोडणे
                st.session_state.df = pd.concat([st.session_state.df, new_entry], ignore_index=True)
                
                st.success(f"Added '{new_name}' to inventory.")
                st.rerun()

    st.divider()
    # Feature 4: Hyperparameter Weights Control in Sidebar
    st.markdown("### Model Weight Calibration")
    st.caption("Tune the composite scoring function balance:")
    skill_weight = st.slider("Skill Match Weight", min_value=0.1, max_value=0.9, value=0.7, step=0.05)
    exp_weight = round(1.0 - skill_weight, 2)
    st.info(f"Formula: **{int(skill_weight*100)}% Skill** + **{int(exp_weight*100)}% Experience**")

# ==================== MAIN DASHBOARD ====================
st.markdown("""
<div class="hero-banner">
    <h1>SmartBench Optimizer</h1>
    <p>AI-Powered Talent Allocation & Semantic Bench Matching Engine</p>
</div>
""", unsafe_allow_html=True)

bench = df[df["status"] == "Bench"].copy()

# Metric Cards
m1, m2, m3 = st.columns(3)
total_count = len(df)
bench_count = len(bench)
bench_ratio = f"{(bench_count / total_count * 100):.1f}%" if total_count > 0 else "0%"

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Headcount</div>
        <div class="metric-number">{total_count}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Available on Bench</div>
        <div class="metric-number">{bench_count}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Bench Ratio</div>
        <div class="metric-number">{bench_ratio}</div>
    </div>
    """, unsafe_allow_html=True)

# ==================== FEATURE 2: VISUAL ANALYTICS ====================
with st.expander("Bench Talent Analytics & Skill Distribution", expanded=False):
    chart_col1, chart_col2 = st.columns(2)
    
    # 1. Top Skills Chart
    all_skills_list = []
    for s in bench["skills_profile"].dropna():
        items = [x.strip().title() for x in str(s).split(",") if x.strip()]
        all_skills_list.extend(items)
    
    skill_counts = Counter(all_skills_list).most_common(8)
    if skill_counts:
        skills_df = pd.DataFrame(skill_counts, columns=["Skill", "Count"])
        fig_bar = px.bar(
            skills_df, 
            x="Count", 
            y="Skill", 
            orientation="h",
            title="Top Available Skills on Bench",
            color="Count",
            color_continuous_scale="Blues"
        )
        fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=300, margin=dict(l=20, r=20, t=40, b=20))
        chart_col1.plotly_chart(fig_bar, use_container_width=True)

    # 2. Experience Distribution Chart
    fig_hist = px.histogram(
        bench, 
        x="experience_years", 
        nbins=6, 
        title="Experience Distribution (Years)",
        color_discrete_sequence=["#3b82f6"]
    )
    fig_hist.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Years", yaxis_title="Count")
    chart_col2.plotly_chart(fig_hist, use_container_width=True)

# Active Roster Table
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

# Matching Execution
if st.button("Find Matching Resources", type="primary", use_container_width=True):
    if not project_description.strip():
        st.warning("Please supply project requirements to calculate semantic similarity.")
    elif len(bench) == 0:
        st.error("No resources currently marked as 'Bench'.")
    else:
        with st.spinner("Computing cosine similarity vectors..."):
            vectorizer = TfidfVectorizer(stop_words='english')
            all_documents = list(bench["skills_profile"]) + [project_description]
            tfidf_matrix = vectorizer.fit_transform(all_documents)
            
            similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
            bench["Skill_Match"] = similarity_scores
            
            # Experience scoring function
            bench["Experience_Score"] = (bench["experience_years"] / required_experience).clip(upper=1.0)
            
            # Feature 4: Dynamic Weights Application
            bench["Composite_Score"] = (skill_weight * bench["Skill_Match"]) + (exp_weight * bench["Experience_Score"])
            
            results = bench.sort_values(by="Composite_Score", ascending=False).head(top_n)
            
            st.markdown("### Top Matched Candidates")
            
            display_df = results[[
                "employee_name", 
                "skills_profile", 
                "experience_years", 
                "Skill_Match", 
                "Composite_Score"
            ]].copy()
            
            # Format percentage for display
            display_df["Skill_Match_Pct"] = (display_df["Skill_Match"] * 100).round(1).astype(str) + "%"
            display_df["Composite_Score_Pct"] = (display_df["Composite_Score"] * 100).round(1).astype(str) + "%"
            
            st.dataframe(
                display_df[[
                    "employee_name", 
                    "skills_profile", 
                    "experience_years", 
                    "Skill_Match_Pct", 
                    "Composite_Score_Pct"
                ]].rename(columns={
                    "employee_name": "Employee Name",
                    "skills_profile": "Skills Profile",
                    "experience_years": "Experience (Yrs)",
                    "Skill_Match_Pct": "Skill Match",
                    "Composite_Score_Pct": "Overall Fit"
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Feature 2 (Contd): Scatter plot showing Experience vs Match Score
            st.markdown("#### Candidate Fit Visualization")
            fig_scatter = px.scatter(
                results,
                x="experience_years",
                y="Composite_Score",
                text="employee_name",
                size=[12]*len(results),
                color="Composite_Score",
                color_continuous_scale="Viridis",
                labels={"experience_years": "Experience (Years)", "Composite_Score": "Overall Fit Score"},
                title="Top Candidates: Experience vs. Composite Fit"
            )
            fig_scatter.update_traces(textposition='top center')
            fig_scatter.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Feature 5: Export Allocation Report
            st.markdown("#### Export Results")
            export_csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Shortlist Report (CSV)",
                data=export_csv,
                file_name="smartbench_shortlist_report.csv",
                mime="text/csv",
                use_container_width=True
            )
