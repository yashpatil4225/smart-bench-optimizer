import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="SmartBench AI | Enterprise Resource Optimizer",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Universal Clean SaaS Styling (High-Contrast & Clean Cards) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer, header {visibility: hidden;}

    /* Professional Top Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 24px;
        color: #ffffff;
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 1.7rem;
        font-weight: 700;
        color: #ffffff;
    }
    .hero-banner p {
        margin-top: 6px;
        margin-bottom: 0;
        color: #94a3b8;
        font-size: 0.95rem;
    }

    /* Metric Cards - 100% High Contrast */
    .metric-card {
        background-color: #0f172a;
        border: 1px solid #334155;
        padding: 18px 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #f1f5f9 !important; /* Pure Bright White */
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-number {
        font-size: 2.2rem;
        font-weight: 800;
        color: #38bdf8 !important; /* Bright Neon Blue */
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 5px;
    }

    /* Force Sidebar Input Labels to be High Contrast & Readable */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] label p,
    div[data-testid="stForm"] label,
    div[data-testid="stForm"] label p {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.90rem !important;
    }

    /* Candidate Detail Card */
    .candidate-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .candidate-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
    }
    .candidate-exp {
        font-size: 0.85rem;
        color: #38bdf8;
        font-weight: 600;
    }
    .candidate-skills {
        font-size: 0.9rem;
        color: #cbd5e1;
        margin: 6px 0;
    }
</style>
""", unsafe_allow_html=True)

CSV_FILE = "employees.csv"

# ==================== DATA SESSION HANDLING ====================
if "df" not in st.session_state:
    try:
        st.session_state.df = pd.read_csv(CSV_FILE)
    except Exception:
        st.session_state.df = pd.DataFrame(columns=["employee_name", "skills_profile", "experience_years", "status"])

df = st.session_state.df

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## ⬡ SmartBench AI")
    st.caption("Enterprise Resource Allocation Engine")
    st.markdown("---")
    
    st.subheader("Resource Onboarding")
    with st.form(key="add_employee_form", clear_on_submit=True):
        new_name = st.text_input("Full Name", placeholder="e.g. Alex Johnson")
        new_skills = st.text_area("Skills Profile (Comma-separated)", placeholder="e.g. Python, SQL, Docker, AWS", height=80)
        new_exp = st.number_input("Experience (Years)", min_value=0.0, max_value=30.0, value=3.0, step=0.5)
        new_status = st.selectbox("Current Status", ["Bench", "Allocated"])
        
        submitted = st.form_submit_button("Add Resource to Inventory", use_container_width=True, type="primary")
        
        if submitted:
            if not new_name.strip() or not new_skills.strip():
                st.error("Please provide both Name and Skills.")
            else:
                new_entry = pd.DataFrame([{
                    "employee_name": new_name.strip(),
                    "skills_profile": new_skills.strip(),
                    "experience_years": float(new_exp),
                    "status": new_status
                }])
                st.session_state.df = pd.concat([st.session_state.df, new_entry], ignore_index=True)
                try:
                    new_entry.to_csv(CSV_FILE, mode='a', header=False, index=False)
                except Exception:
                    pass
                st.success(f"Added {new_name} successfully!")
                st.rerun()

    st.markdown("---")
    st.subheader("Algorithm Calibration")
    skill_weight = st.slider("Skill Match Weight", min_value=0.1, max_value=0.9, value=0.7, step=0.05)
    exp_weight = round(1.0 - skill_weight, 2)
    st.caption(f"Score Formula: **{int(skill_weight*100)}% Skills** + **{int(exp_weight*100)}% Experience**")

# ==================== MAIN PANEL ====================
st.markdown("""
<div class="hero-banner">
    <h1>SmartBench Resource Allocation Optimizer</h1>
    <p>Semantic NLP Talent Matching Engine powered by TF-IDF vectorization and cosine similarity scoring.</p>
</div>
""", unsafe_allow_html=True)

bench = df[df["status"] == "Bench"].copy()

# High-Contrast Metric Cards
total_count = len(df)
bench_count = len(bench)
bench_ratio = f"{(bench_count / total_count * 100):.1f}%" if total_count > 0 else "0.0%"

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Headcount</div>
        <div class="metric-number">{total_count}</div>
        <div class="metric-sub">Total Managed Talent</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Available on Bench</div>
        <div class="metric-number">{bench_count}</div>
        <div class="metric-sub">Ready for Projects</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Bench Utilization Ratio</div>
        <div class="metric-number" style="color: #a855f7;">{bench_ratio}</div>
        <div class="metric-sub">Optimization Target</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# Active Bench Roster Table
with st.expander("📋 View Complete Active Bench Roster", expanded=False):
    st.dataframe(
        bench[["employee_name", "skills_profile", "experience_years"]].rename(columns={
            "employee_name": "Employee Name",
            "skills_profile": "Skills Profile",
            "experience_years": "Experience (Years)"
        }),
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# ==================== NLP MATCHING ENGINE ====================
st.subheader("Semantic Matching Studio")
st.caption("Enter requirements to match candidate capabilities mathematically.")

query_col, filter_col = st.columns([3, 1])

with query_col:
    project_description = st.text_area(
        "Project Requirements / Scope",
        placeholder="e.g. Looking for a Python backend developer experienced with FastAPI, PostgreSQL, Docker, and AWS microservices.",
        height=100
    )

with filter_col:
    required_experience = st.slider("Target Experience (Yrs)", min_value=1, max_value=15, value=4)
    top_n = st.selectbox("Max Recommendations", [3, 5, 8], index=1)

if st.button("Find Matching Resources", type="primary", use_container_width=True):
    if not project_description.strip():
        st.warning("Please enter project requirements first.")
    elif len(bench) == 0:
        st.error("No resources currently available on the bench.")
    else:
        with st.spinner("Calculating semantic similarity..."):
            # TF-IDF Matching
            vectorizer = TfidfVectorizer(stop_words='english')
            corpus = list(bench["skills_profile"]) + [project_description]
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Compute Cosine Scores
            bench["Skill_Match"] = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
            bench["Experience_Score"] = (bench["experience_years"] / required_experience).clip(upper=1.0)
            bench["Composite_Score"] = (skill_weight * bench["Skill_Match"]) + (exp_weight * bench["Experience_Score"])
            
            results = bench.sort_values(by="Composite_Score", ascending=False).head(top_n).copy()
            
            st.markdown("### Top Recommended Candidates")
            
            # Interactive Visual Progress Cards
            for _, row in results.iterrows():
                fit_score = int(round(row["Composite_Score"] * 100))
                skill_score = int(round(row["Skill_Match"] * 100))
                
                st.markdown(f"""
                <div class="candidate-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="candidate-name">{row['employee_name']}</span>
                        <span class="candidate-exp">{row['experience_years']} Years Exp</span>
                    </div>
                    <div class="candidate-skills"><strong>Skills:</strong> {row['skills_profile']}</div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8; margin-top: 6px;">
                        <span>Overall Suitability: <strong style="color: #38bdf8;">{fit_score}%</strong> (Skill Match: {skill_score}%)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(fit_score / 100)
            
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            
            # Formatted Clean Table
            export_df = results[["employee_name", "skills_profile", "experience_years", "Composite_Score"]].copy()
            export_df["Overall Fit (%)"] = (export_df["Composite_Score"] * 100).round(1).astype(str) + "%"
            clean_table = export_df[["employee_name", "skills_profile", "experience_years", "Overall Fit (%)"]].rename(columns={
                "employee_name": "Employee Name",
                "skills_profile": "Skills Profile",
                "experience_years": "Experience (Years)"
            })
            
            st.dataframe(clean_table, use_container_width=True, hide_index=True)
            
            # ==================== CLEAN EXPORT ====================
            st.markdown("#### Export Allocation Report")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    label="📥 Download CSV Report",
                    data=clean_table.to_csv(index=False).encode('utf-8'),
                    file_name="SmartBench_Allocation.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with c2:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    clean_table.to_excel(writer, index=False, sheet_name='Allocations')
                st.download_button(
                    label="📊 Download Excel Spreadsheet (.xlsx)",
                    data=buffer.getvalue(),
                    file_name="SmartBench_Allocation.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
