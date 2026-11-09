import io
import os

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Smart Bench Optimizer",
    page_icon="🎯",
    layout="wide",
)

# ---------------------------------------------------------------
# Premium styling
# ---------------------------------------------------------------
st.markdown(
    /* सर्व सामान्य टेक्स्ट आणि लेबल्स पांढरे करण्यासाठी */
    html, body, [class*="css"], label, p, span { 
        font-family: 'Inter', sans-serif;
        color: #f8fafc !important; 
    }

    /* Text Area - टायपिंग फॉन्ट आणि प्लेसहोल्डर */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 14px !important;
        color: #ffffff !important; /* टाइप केलेला फॉन्ट पांढरा */
        font-size: 1rem !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #94a3b8 !important; /* फिकट पांढरा/राखाडी प्लेसहोल्डर */
        opacity: 1 !important;
    }

    .stTextArea textarea:focus { 
        border-color: #38bdf8 !important; 
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.4) !important; 
        color: #ffffff !important;
    }

    /* Sidebar - साइडबारमधील सर्व टेक्स्ट आणि कॅप्शन्स */
    section[data-testid="stSidebar"] {
        background: #090d1a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stCaption {
        color: #cbd5e1 !important;
    }

    /* Cards मधील लेबल्स */
    .card .label { 
        font-size: .78rem; 
        letter-spacing: .08em; 
        text-transform: uppercase; 
        color: #cbd5e1 !important; 
    }
    .card .value { 
        font-size: 1.9rem; 
        font-weight: 700; 
        color: #ffffff !important; 
        margin-top: .15rem; 
    }
    }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem; max-width: 1200px; }

    /* Hero */
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(14,165,233,0.18));
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 20px 50px rgba(0,0,0,0.35);
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        margin: 0; font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #a5b4fc, #67e8f9);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { margin: .5rem 0 0 0; color: #cbd5e1; font-size: 1.02rem; }

    /* Metric cards */
    .card {
        padding: 1.1rem 1.3rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.09);
        backdrop-filter: blur(8px);
    }
    .card .label { font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; color: #94a3b8; }
    .card .value { font-size: 1.9rem; font-weight: 700; color: #f8fafc; margin-top: .15rem; }
    .card .sub { font-size: .8rem; color: #67e8f9; }

    /* Section titles */
    .section-title { font-size: 1.15rem; font-weight: 700; margin: 1.6rem 0 .7rem 0; color: #f1f5f9; }

    /* Inputs */
    .stTextArea textarea {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        border-radius: 14px !important;
        color: #f8fafc !important;
    }
    .stTextArea textarea:focus { border-color: #818cf8 !important; box-shadow: 0 0 0 2px rgba(129,140,248,.35) !important; }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border: none; border-radius: 14px;
        padding: .8rem 1rem; font-weight: 700; font-size: 1rem;
        color: white;
        background: linear-gradient(90deg, #6366f1, #0ea5e9);
        box-shadow: 0 10px 25px rgba(99,102,241,.35);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 14px 30px rgba(14,165,233,.45); color: white; }

    .stDownloadButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.06);
        color: #e2e8f0; font-weight: 600;
    }
    .stDownloadButton > button:hover { border-color: #67e8f9; color: #67e8f9; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(10,14,30,0.95);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,0.09); }

    .footer-note { text-align:center; color:#64748b; font-size:.8rem; margin-top:2.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "employees.csv")
    return pd.read_csv(path)


df = load_data()
bench = df[df["status"] == "Bench"].copy()


# ---------------------------------------------------------------
# Matching logic (TF-IDF + cosine similarity + hybrid score)
# ---------------------------------------------------------------
def rank_employees(bench_df, project_text, required_exp, skill_w, exp_w):
    ranked = bench_df.copy()
    vectorizer = TfidfVectorizer()
    all_text = list(ranked["skills_profile"]) + [project_text]
    matrix = vectorizer.fit_transform(all_text)
    ranked["Skill_Score"] = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked["Experience_Score"] = (ranked["experience_years"] / required_exp).clip(upper=1.0)
    ranked["Final_Score"] = skill_w * ranked["Skill_Score"] + exp_w * ranked["Experience_Score"]
    ranked = ranked.sort_values("Final_Score", ascending=False).head(5).reset_index(drop=True)
    ranked.insert(0, "Rank", ranked.index + 1)
    return ranked


def to_excel_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="Top Matches")
        sheet = writer.sheets["Top Matches"]
        for col in sheet.columns:
            width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
            sheet.column_dimensions[col[0].column_letter].width = min(width + 3, 60)
    return buffer.getvalue()


# ---------------------------------------------------------------
# Sidebar (settings)
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    required_experience = st.slider("Required experience (years)", 1, 10, 4)
    skill_pct = st.slider("Skill weight (%)", 0, 100, 70, step=5)
    exp_pct = 100 - skill_pct
    st.caption(f"Skill **{skill_pct}%**  +  Experience **{exp_pct}%**")
    st.markdown("---")
    st.caption(
        "Skills are converted with TF-IDF, compared using cosine similarity, "
        "then combined with experience into a final score."
    )

# ---------------------------------------------------------------
# Hero
# ---------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎯 IT Smart-Bench &amp; Resource Allocation Optimizer</h1>
        <p>Describe your project and instantly find the best available employees from the bench.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Stat cards
# ---------------------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f'<div class="card"><div class="label">Employees on bench</div>'
        f'<div class="value">{len(bench)}</div><div class="sub">available now</div></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="card"><div class="label">Total employees</div>'
        f'<div class="value">{len(df)}</div><div class="sub">in database</div></div>',
        unsafe_allow_html=True,
    )
with c3:
    avg_exp = bench["experience_years"].mean() if len(bench) else 0
    st.markdown(
        f'<div class="card"><div class="label">Avg bench experience</div>'
        f'<div class="value">{avg_exp:.1f} yrs</div><div class="sub">across bench</div></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------
# Input
# ---------------------------------------------------------------
st.markdown('<div class="section-title">📝 Project requirement</div>', unsafe_allow_html=True)
project_description = st.text_area(
    "Project description",
    placeholder="e.g. Need a python developer with machine learning and sql knowledge",
    height=120,
    label_visibility="collapsed",
)

if st.button("🔍 Find Available Resources"):
    if not project_description.strip():
        st.warning("Please enter a project description.")
        st.session_state.pop("result", None)
    elif bench.empty:
        st.info("No employees are currently on the bench.")
        st.session_state.pop("result", None)
    else:
        st.session_state["result"] = rank_employees(
            bench,
            project_description,
            required_experience,
            skill_pct / 100,
            exp_pct / 100,
        )

# ---------------------------------------------------------------
# Results (kept in session_state so downloads don't clear them)
# ---------------------------------------------------------------
if "result" in st.session_state:
    result = st.session_state["result"]

    st.markdown('<div class="section-title">🏆 Top Matched Employees</div>', unsafe_allow_html=True)

    best = result.iloc[0]
    b1, b2, b3 = st.columns(3)
    b1.metric("Best match", best["employee_name"])
    b2.metric("Final score", f"{best['Final_Score']:.2f}")
    b3.metric("Experience", f"{int(best['experience_years'])} yrs")

    display = result[
        ["Rank", "employee_name", "skills_profile", "experience_years",
         "Skill_Score", "Experience_Score", "Final_Score"]
    ].rename(
        columns={
            "employee_name": "Employee",
            "skills_profile": "Skills",
            "experience_years": "Experience (yrs)",
            "Skill_Score": "Skill Score",
            "Experience_Score": "Experience Score",
            "Final_Score": "Final Score",
        }
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Skill Score": st.column_config.ProgressColumn("Skill Score", min_value=0.0, max_value=1.0, format="%.2f"),
            "Experience Score": st.column_config.ProgressColumn("Experience Score", min_value=0.0, max_value=1.0, format="%.2f"),
            "Final Score": st.column_config.ProgressColumn("Final Score", min_value=0.0, max_value=1.0, format="%.2f"),
        },
    )

    # Export
    st.markdown('<div class="section-title">📥 Export results</div>', unsafe_allow_html=True)
    export_frame = display.copy()
    for col in ["Skill Score", "Experience Score", "Final Score"]:
        export_frame[col] = export_frame[col].round(3)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "⬇️ Download CSV",
            data=export_frame.to_csv(index=False).encode("utf-8"),
            file_name="top_bench_matches.csv",
            mime="text/csv",
        )
    with d2:
        st.download_button(
            "⬇️ Download Excel",
            data=to_excel_bytes(export_frame),
            file_name="top_bench_matches.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

st.markdown(
    '<div class="footer-note">Built with Python · Pandas · Scikit-learn · Streamlit</div>',
    unsafe_allow_html=True,
)
