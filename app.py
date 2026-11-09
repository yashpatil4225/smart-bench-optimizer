import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# ==============================================================================
# 1. PAGE SETUP
# ==============================================================================
st.set_page_config(
    page_title="SmartBench AI — Enterprise Resource Optimizer",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. BULLETPROOF HIGH-CONTRAST STYLING
# ==============================================================================
st.markdown("""
<style>
    /* Global Base */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    #MainMenu, footer, header {visibility: hidden;}

    /* Sidebar Background & Clean Layout */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #1e293b;
    }

    /* 🔥 सर्व इनपुट लेबल्स १००% पांढरे आणि ठळक 🔥 */
    section[data-testid="stSidebar"] label,
    div[data-testid="stForm"] label,
    .stSlider label,
    .stSelectbox label,
    .stTextArea label,
    .stTextInput label,
    .stNumberInput label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.02em;
    }

    /* इनपुट बॉक्सची बॉर्डर्स आणि टेक्स्ट */
    div[data-baseweb="input"] > div, 
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="select"] > div {
        background-color: #111827 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }

    /* Top Hero Header */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
    }
    .hero-title {
        color: #ffffff !important;
        font-size: 1.7rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #94a3b8 !important;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* KPI Cards (High Contrast - No Missmatch) */
    .kpi-box {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    }
    .kpi-title {
        color: #f1f5f9 !important; /* Pure Bright White */
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }
    .kpi-number {
        color: #38bdf8 !important; /* Neon Electric Blue */
        font-size: 2.3rem;
        font-weight: 800;
        line-height: 1;
    }
    .kpi-footer {
        color: #94a3b8 !important;
        font-size: 0.78rem;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Primary Action Buttons */
    button[kind="primary"] {
        background: #0284c7 !important;
        border: none !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
    }
    button[kind="primary"]:hover {
        background: #0369a1 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. DATA LOAD & REPO STATE
# ==============================================================================
CSV_FILE = "employees.csv"

if "df" not in st.session_state:
    try:
        st.session_state.df = pd.read_csv(CSV_FILE)
    except Exception:
        st.session_state.df = pd.DataFrame(columns=[
            "employee_name", "skills_profile", "experience_years", "status"
        ])

df = st.session_state.df
bench = df[df["status"] == "Bench"].copy()

# ==============================================================================
# 4. SIDEBAR (RESOURCE ONBOARDING & CONFIG)
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="padding-bottom: 12px; margin-bottom: 16px; border-bottom: 1px solid #1e293b;">
        <h3 style="color: #ffffff; margin: 0; font-size: 1.3rem;">⬡ SmartBench AI</h3>
        <span style="font-size: 0.7rem; color: #38bdf8; font-weight: 700; text-transform: uppercase;">Enterprise Edition</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='color:#e2e8f0; font-weight:600; font-size:1rem; margin-bottom:12px;'>Resource Onboarding</p>", unsafe_allow_html=True)

    with st.form(key="onboarding_form", clear_on_submit=True):
        new_name = st.text_input("Full Name", placeholder="e.g. Alex Johnson")
        new_skills = st.text_area("Skills Profile", placeholder="e.g. Python, Docker, FastAPI, PostgreSQL", height=80)
        new_exp = st.number_input("Experience (Years)", min_value=0.0, max_value=30.0, value=3.0, step=0.5)
        new_status = st.selectbox("Current Status", ["Bench", "Allocated"])
        
        submit_btn = st.form_submit_button("Add Resource", use_container_width=True, type="primary")
        
        if submit_btn:
            if not new_name.strip() or not new_skills.strip():
                st.error("Please provide both Name and Skills.")
            else:
                new_row = pd.DataFrame([{
                    "employee_name": new_name.strip(),
                    "skills_profile": new_skills.strip(),
                    "experience_years": float(new_exp),
                    "status": new_status
                }])
                st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
                try:
                    new_row.to_csv(CSV_FILE, mode='a', header=False, index=False)
                except Exception:
                    pass
                st.success(f"Added {new_name} successfully!")
                st.rerun()

    st.markdown("<hr style='border:0; border-top:1px solid #1e293b; margin: 20px 0;'>", unsafe_allow_html=True)
    
    st.markdown("<p style='color:#e2e8f0; font-weight:600; font-size:0.95rem; margin-bottom:4px;'>Algorithm Weight Calibration</p>", unsafe_allow_html=True)
    skill_weight = st.slider("Skill Match Weight", min_value=0.1, max_value=0.9, value=0.7, step=0.05)
    exp_weight = round(1.0 - skill_weight, 2)
    st.markdown(f"""
    <div style="background:#111827; padding:8px 12px; border-radius:6px; border:1px solid #1e293b; color:#38bdf8; font-size:0.85rem; font-weight:600;">
        Weight: {int(skill_weight*100)}% Skill + {int(exp_weight*100)}% Experience
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 5. MAIN HEADER & METRIC CARDS
# ==============================================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">SmartBench Resource Allocation Optimizer</div>
    <div class="hero-subtitle">
        AI-Powered bench resource allocation engine using TF-IDF tokenization and cosine similarity to match bench talents with active project scopes.
    </div>
</div>
""", unsafe_allow_html=True)

# Top KPI Metric Cards
total_count = len(df)
bench_count = len(bench)
bench_ratio = f"{(bench_count / total_count * 100):.1f}%" if total_count > 0 else "0.0%"

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">Total Headcount</div>
        <div class="kpi-number">{total_count}</div>
        <div class="kpi-footer">Total Organization Pool</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">Available on Bench</div>
        <div class="kpi-number">{bench_count}</div>
        <div class="kpi-footer">Unallocated Resources</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">Bench Ratio</div>
        <div class="kpi-number" style="color: #a855f7;">{bench_ratio}</div>
        <div class="kpi-footer">Optimization Scope</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# 6. BENCH INVENTORY ROSTER
# ==============================================================================
with st.expander("View Active Bench Engineers Roster", expanded=False):
    st.dataframe(
        bench[["employee_name", "skills_profile", "experience_years"]].rename(columns={
            "employee_name": "Resource Name",
            "skills_profile": "Skills Profile",
            "experience_years": "Experience (Years)"
        }),
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# ==============================================================================
# 7. MATCHING ENGINE
# ==============================================================================
st.markdown("<h3 style='color:#ffffff; margin-bottom: 4px;'>Semantic Resource Search</h3>", unsafe_allow_html=True)
st.caption("Provide project technical scope to calculate candidate relevance scores.")

req_col, cfg_col = st.columns([3, 1])

with req_col:
    project_description = st.text_area(
        "Project Scope & Technical Requirements",
        placeholder="e.g. Need a Python backend developer with FastAPI, PostgreSQL, Docker, and experience in cloud deployments.",
        height=100
    )

with cfg_col:
    required_exp = st.slider("Target Experience (Years)", 1, 15, 4)
    top_n = st.selectbox("Candidate Recommendations Limit", [3, 5, 10], index=1)

run_search = st.button("Find Best-Fit Resources", type="primary", use_container_width=True)

# ==============================================================================
# 8. RESULTS & CLEAN REPORT (REPLACED MESSED-UP VISUALIZATION)
# ==============================================================================
if run_search:
    if not project_description.strip():
        st.warning("Please enter a project description to compute match scores.")
    elif len(bench) == 0:
        st.error("No employees are currently marked on 'Bench'.")
    else:
        with st.spinner("Calculating semantic match vectors..."):
            # TF-IDF & Cosine Similarity
            vectorizer = TfidfVectorizer(stop_words='english')
            corpus = list(bench["skills_profile"]) + [project_description]
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
            bench["Skill_Match"] = scores
            bench["Exp_Score"] = (bench["experience_years"] / required_exp).clip(upper=1.0)
            bench["Final_Score"] = (skill_weight * bench["Skill_Match"]) + (exp_weight * bench["Exp_Score"])
            
            results = bench.sort_values(by="Final_Score", ascending=False).head(top_n).copy()
            
            # Format display
            results["Skill_Match_Pct"] = (results["Skill_Match"] * 100).round(1).astype(str) + "%"
            results["Final_Score_Pct"] = (results["Final_Score"] * 100).round(1).astype(str) + "%"
            
            clean_table = results[[
                "employee_name", "skills_profile", "experience_years", "Skill_Match_Pct", "Final_Score_Pct"
Screenshot वरून २ मुख्य चुका स्पष्ट दिसत आहेत[cite: 6]:
1. **कलर कॉन्ट्रास्टचा घोळ:** Streamlit बाय-डिफॉल्ट लाईट थीममध्ये चालू आहे, पण आपण काही घटकांना डार्क बॅकग्राउंड दिलं आहे[cite: 6]. त्यामुळे:
   - Sidebar मधील इनपुट लेबल्स ("Full Name", "Skills Inventory", "Experience") काळ्या बॅकग्राउंडवर डार्क टेक्स्ट आल्याने वाचताच येत नाहीत[cite: 6].
   - खालील बार चार्टमध्ये डावीकडचे स्किल्सचे नाव अर्धे कापले आहेत आणि फॉन्ट फिकट राखाडी असल्याने वाचता येत नाहीत[cite: 6].
2. **Scatter Plot / अवघड Visuals:** तो scatter plot समजायला किचकट वाटतो आणि जागा अडवतो[cite: 6].

त्यामुळे आपण हा visual गोंधळ पूर्णपणे काढून त्याजागी **Clean Executive UI & Skill Match Progress Bars** देऊ, जेणेकरून कोणत्याही थीममध्ये (Dark किंवा Light) प्रत्येक शब्द १००% स्पष्ट वाचता येईल.

---

### संपूर्ण फिक्स केलेला `app.py` कोड:

GitHub वर जाऊन हा संपूर्ण कोड पेस्ट करा आणि **Commit changes** करा:

```python
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

# --- Universal Clean SaaS Styling (Works cleanly in Dark & Light modes) ---
st.markdown("""
<style>
    @import url('[https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap](https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap)');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

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
    .metric-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .metric-card {
        background-color: #0f172a;
        border: 1px solid #334155;
        padding: 18px 20px;
        border-radius: 10px;
        text-align: center;
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #cbd5e1 !important; /* Crystal Clear White-Grey */
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
    section[data-testid="stSidebar"] label p {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
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
        new_name = st.text_input("Full Name", placeholder="e.g. Yash Patil")
        new_skills = st.text_area("Skills (Comma-separated)", placeholder="e.g. Python, SQL, Docker, AWS", height=80)
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

# High-Contrast Metric Cards (Zero Text Truncation)
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
        <div class="metric-number">{bench_ratio}</div>
        <div class="metric-sub">Optimization Target</div>
    </div>
    """, unsafe_allow_html=True)

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
            
            # Interactive Visual Progress Cards instead of messy plots
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
                        <span>Overall Suitability: <strong>{fit_score}%</strong> (Skill Match: {skill_score}%)</span>
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
