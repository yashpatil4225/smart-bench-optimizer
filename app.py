import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import io

# ==============================================================================
# 1. PAGE CONFIGURATION & METADATA
# ==============================================================================
st.set_page_config(
    page_title="SmartBench AI — Enterprise Talent Optimization",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. DESIGN SYSTEM & CSS (LINEAR / VERCEL SAAS THEME)
# ==============================================================================
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Root Variables */
    :root {
        --bg-primary: #0A0D14;
        --bg-secondary: #0F172A;
        --bg-card: #131B2E;
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-accent: rgba(56, 189, 248, 0.25);
        --accent-blue: #38BDF8;
        --accent-indigo: #6366F1;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
    }

    /* Base Font Override */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }

    /* Hide default Streamlit decoration header & deploy button clutter */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    #MainMenu, footer {visibility: hidden;}

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #07090E !important;
        border-right: 1px solid var(--border-subtle);
    }
    section[data-testid="stSidebar"] div.block-container {
        padding-top: 2rem;
    }

    /* SaaS Brand Identity Header */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 20px;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 24px;
    }
    .brand-logo {
        background: linear-gradient(135deg, #0284C7, #6366F1);
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1.2rem;
        color: #FFFFFF;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.35);
    }
    .brand-text {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin: 0;
    }
    .brand-badge {
        font-size: 0.65rem;
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        padding: 2px 7px;
        border-radius: 9999px;
        border: 1px solid rgba(56, 189, 248, 0.25);
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: inline-block;
        margin-top: 2px;
    }

    /* Linear-Style Hero Banner */
    .hero-panel {
        background: radial-gradient(100% 120% at 50% 0%, rgba(30, 58, 138, 0.28) 0%, rgba(10, 13, 20, 0) 100%), #0D1322;
        border: 1px solid var(--border-subtle);
        padding: 30px 36px;
        border-radius: 16px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero-panel::before {
        content: "";
        position: absolute;
        top: 0;
        left: 15%;
        right: 15%;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.6), transparent);
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #FFFFFF;
        margin: 0 0 6px 0;
    }
    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin: 0;
        max-width: 650px;
        line-height: 1.5;
    }

    /* KPI Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background-color: #0F172A;
        border: 1px solid var(--border-subtle);
        padding: 20px 24px;
        border-radius: 14px;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .kpi-card:hover {
        border-color: var(--border-accent);
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #F8FAFC;
        line-height: 1.1;
    }
    .kpi-tag {
        font-size: 0.75rem;
        color: #38BDF8;
        font-weight: 500;
        margin-top: 6px;
    }

    /* Form Card & Sidebar Components */
    .section-card {
        background: #0E1626;
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 24px;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 4px;
    }
    .section-caption {
        font-size: 0.82rem;
        color: var(--text-muted);
        margin-bottom: 18px;
    }

    /* Streamlit Input & Widget Polish */
    div[data-baseweb="input"] > div, 
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="select"] > div {
        background-color: #0B111E !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        color: #F8FAFC !important;
    }
    div[data-baseweb="input"] > div:focus-within, 
    div[data-baseweb="textarea"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 1px #38BDF8 !important;
    }

    /* Modern Buttons */
    button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7, #2563EB) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369A1, #1D4ED8) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
        transform: translateY(-1px);
    }
    button[kind="secondary"] {
        background-color: #1E293B !important;
        border: 1px solid var(--border-subtle) !important;
        color: #F8FAFC !important;
        border-radius: 8px !important;
    }

    /* Dataframe Table Elevation */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. DATA PERSISTENCE & SESSION STATE
# ==============================================================================
CSV_FILE = "employees.csv"

if "df" not in st.session_state:
    try:
        st.session_state.df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        st.session_state.df = pd.DataFrame(columns=[
            "employee_name", "skills_profile", "experience_years", "status"
        ])

df = st.session_state.df

# ==============================================================================
# 4. SIDEBAR NAVIGATION & ONBOARDING CONTROLS
# ==============================================================================
with st.sidebar:
    # SaaS Brand Header
    st.markdown("""
    <div class="brand-container">
        <div class="brand-logo">⬡</div>
        <div>
            <div class="brand-text">SmartBench AI</div>
            <div class="brand-badge">Enterprise SaaS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Resource Intake Card
    st.markdown('<div class="section-header">Resource Intake</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Register new or bench engineer into cluster</div>', unsafe_allow_html=True)
    
    with st.form(key="add_resource_form", clear_on_submit=True):
        new_name = st.text_input("Full Name", placeholder="e.g. Alex Johnson")
        new_skills = st.text_area(
            "Skills Inventory (Comma-delimited)", 
            placeholder="e.g. Python, Docker, FastAPI, PostgreSQL, AWS",
            height=85
        )
        new_exp = st.number_input("Experience (Years)", min_value=0.0, max_value=30.0, value=3.0, step=0.5)
        new_status = st.selectbox("Allocation Status", ["Bench", "Allocated"])
        
        onboard_btn = st.form_submit_button("Register Resource", use_container_width=True, type="primary")
        
        if onboard_btn:
            if not new_name.strip() or not new_skills.strip():
                st.error("Validation failed: Both Name and Skills are mandatory.")
            else:
                new_entry = pd.DataFrame([{
                    "employee_name": new_name.strip(),
                    "skills_profile": new_skills.strip(),
                    "experience_years": float(new_exp),
                    "status": new_status
                }])
                st.session_state.df = pd.concat([st.session_state.df, new_entry], ignore_index=True)
                
                # Best-effort disk sync
                try:
                    new_entry.to_csv(CSV_FILE, mode='a', header=False, index=False)
                except Exception:
                    pass
                
                st.success(f"Successfully inducted '{new_name}'")
                st.rerun()

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 24px 0;'>", unsafe_allow_html=True)

    # Hyperparameter Weight Tuning
    st.markdown('<div class="section-header">Matching Hyperparameters</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Calibrate dynamic ranking objective weights</div>', unsafe_allow_html=True)
    
    skill_weight = st.slider("Skill Match Weight", min_value=0.1, max_value=0.9, value=0.7, step=0.05)
    exp_weight = round(1.0 - skill_weight, 2)
    
    st.markdown(f"""
    <div style="background: #0B111E; border: 1px solid rgba(255,255,255,0.08); padding: 12px 14px; border-radius: 8px; margin-top: 10px;">
        <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Scoring Equation</div>
        <div style="font-size: 0.88rem; font-weight: 600; color: #38BDF8; margin-top: 2px;">
            {int(skill_weight * 100)}% Skill <span style="color: #64748B;">+</span> {int(exp_weight * 100)}% Experience
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 5. MAIN HERO & DASHBOARD METRICS
# ==============================================================================
st.markdown("""
<div class="hero-panel">
    <div class="hero-title">SmartBench Resource Allocation Optimizer</div>
    <div class="hero-subtitle">
        AI-driven resource orchestration engine utilizing TF-IDF semantic vectorization and composite similarity scoring to match project pipelines with active bench talent.
    </div>
</div>
""", unsafe_allow_html=True)

bench = df[df["status"] == "Bench"].copy()

total_headcount = len(df)
bench_headcount = len(bench)
bench_ratio = f"{(bench_headcount / total_headcount * 100):.1f}%" if total_headcount > 0 else "0.0%"

# KPI Component Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Enterprise Headcount</div>
        <div class="kpi-value">{total_headcount}</div>
        <div class="kpi-tag">Active Talent Pool</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Available on Bench</div>
        <div class="kpi-value" style="color: #38BDF8;">{bench_headcount}</div>
        <div class="kpi-tag">Ready for Allocation</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Bench Utilization Gap</div>
        <div class="kpi-value" style="color: #818CF8;">{bench_ratio}</div>
        <div class="kpi-tag">Optimization Opportunity</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# 6. TABBED DATA EXPLORATION SUITE
# ==============================================================================
tab_analytics, tab_roster = st.tabs(["📊 Bench Talent Analytics", "📋 Active Talent Roster"])

with tab_analytics:
    chart_c1, chart_c2 = st.columns(2)
    
    # 1. Available Skills Distribution Chart
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
            title="Top Skills Inventory on Bench",
            color="Count",
            color_continuous_scale=[[0, "#1E3A8A"], [1, "#38BDF8"]]
        )
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=20, t=40, b=10),
            height=290,
            yaxis=dict(autorange="reversed", showgrid=False),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)'),
            font=dict(family="Inter, sans-serif", size=12, color="#94A3B8")
        )
        chart_c1.plotly_chart(fig_bar, use_container_width=True)

    # 2. Experience Cohort Distribution
    fig_hist = px.histogram(
        bench,
        x="experience_years",
        nbins=7,
        title="Experience Density (Years)",
        color_discrete_sequence=["#6366F1"]
    )
    fig_hist.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=20, t=40, b=10),
        height=290,
        xaxis=dict(title="Experience (Years)", showgrid=False),
        yaxis=dict(title="Engineers", showgrid=True, gridcolor='rgba(255,255,255,0.06)'),
        font=dict(family="Inter, sans-serif", size=12, color="#94A3B8")
    )
    chart_c2.plotly_chart(fig_hist, use_container_width=True)

with tab_roster:
    st.dataframe(
        bench[["employee_name", "skills_profile", "experience_years"]].rename(columns={
            "employee_name": "Resource Name",
            "skills_profile": "Skills Profile",
            "experience_years": "Experience (Yrs)"
        }),
        use_container_width=True,
        hide_index=True
    )

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# 7. SEMANTIC MATCHING STUDIO
# ==============================================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Project Scope & Semantic Matching</div>', unsafe_allow_html=True)
st.markdown('<div class="section-caption">Input technical specifications to run TF-IDF Vectorization against active bench profiles.</div>', unsafe_allow_html=True)

query_col, config_col = st.columns([3, 1])

with query_col:
    project_description = st.text_area(
        "Technical Requirements / Statement of Work",
        placeholder="e.g. Architecting a scalable microservices backend using Python, FastAPI, Docker, and PostgreSQL with experience in distributed caching and AWS deployments.",
        height=115
    )

with config_col:
    required_experience = st.slider("Target Experience (Yrs)", min_value=1, max_value=15, value=4)
    top_n = st.selectbox("Recommendation Cap", [3, 5, 10], index=1)

run_matching = st.button("Execute Semantic Match", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 8. MATCHING ENGINE EXECUTION & REPORTING
# ==============================================================================
if run_matching:
    if not project_description.strip():
        st.warning("Input required: Please provide technical project requirements to compute similarity vectors.")
    elif len(bench) == 0:
        st.error("Operation halted: No resources currently available in the 'Bench' pool.")
    else:
        with st.spinner("Computing cosine similarity vectors across skill matrix..."):
            # 1. TF-IDF Semantic Tokenization
            vectorizer = TfidfVectorizer(stop_words='english')
            corpus = list(bench["skills_profile"]) + [project_description]
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # 2. Cosine Similarity Computation
            scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
            bench["Skill_Match"] = scores
            
            # 3. Normalized Experience Fit
            bench["Experience_Score"] = (bench["experience_years"] / required_experience).clip(upper=1.0)
            
            # 4. Composite Scoring Matrix
            bench["Composite_Score"] = (skill_weight * bench["Skill_Match"]) + (exp_weight * bench["Experience_Score"])
            
            results = bench.sort_values(by="Composite_Score", ascending=False).head(top_n).copy()
            
            # Format Displays
            results["Skill_Match_Pct"] = (results["Skill_Match"] * 100).round(1).astype(str) + "%"
            results["Composite_Score_Pct"] = (results["Composite_Score"] * 100).round(1).astype(str) + "%"
            
            clean_display = results[[
                "employee_name", "skills_profile", "experience_years", "Skill_Match_Pct", "Composite_Score_Pct"
            ]].rename(columns={
                "employee_name": "Resource Name",
                "skills_profile": "Skills Profile",
                "experience_years": "Experience (Yrs)",
                "Skill_Match_Pct": "Skill Similarity",
                "Composite_Score_Pct": "Composite Fit"
            })
            
            st.markdown("### Top Ranked Allocations")
            st.dataframe(clean_display, use_container_width=True, hide_index=True)
            
            # Interactive Fit Mapping (Plotly Scatter)
            st.markdown("#### Candidate Distribution Fit Matrix")
            fig_scatter = px.scatter(
                results,
                x="experience_years",
                y="Composite_Score",
                text="employee_name",
                size=[14] * len(results),
                color="Composite_Score",
                color_continuous_scale=[[0, "#3B82F6"], [1, "#38BDF8"]],
                labels={"experience_years": "Experience (Years)", "Composite_Score": "Composite Match Score"},
                title="Resource Matrix: Experience Proximity vs. Overall Compatibility"
            )
            fig_scatter.update_traces(
                textposition='top center',
                textfont=dict(family="Inter", color="#E2E8F0", size=11)
            )
            fig_scatter.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=340,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)'),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)'),
                font=dict(family="Inter, sans-serif", color="#94A3B8")
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # ==================================================================
            # 9. EXPORT & ARTIFACT HUB
            # ==================================================================
            st.markdown("#### Export Allocation Brief")
            c_csv, c_excel = st.columns(2)
            
            # Export File Definitions
            export_df = clean_display.copy()
            
            with c_csv:
                csv_bytes = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Shortlist (.CSV)",
                    data=csv_bytes,
                    file_name="SmartBench_Allocation_Brief.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            with c_excel:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    export_df.to_excel(writer, index=False, sheet_name='Shortlisted Candidates')
                
                st.download_button(
                    label="📊 Export Structured Workbook (.XLSX)",
                    data=excel_buffer.getvalue(),
                    file_name="SmartBench_Allocation_Brief.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
