import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Smart Bench Optimizer", layout="wide")
st.title("IT Smart-Bench & Resource Allocation Optimizer")

# Data load
df = pd.read_csv("employees.csv")
bench = df[df["status"] == "Bench"].copy()
st.write(f"Employees on bench: **{len(bench)}**")

# Inputs
project_description = st.text_area(
    "Project description",
    placeholder="e.g. Need a python developer with machine learning and sql knowledge"
)
required_experience = st.slider("Required experience (years)", 1, 10, 4)

# Button
if st.button("Find Available Resources"):
    if not project_description.strip():
        st.warning("Please enter a project description.")
    else:
        # TF-IDF + cosine similarity
        vectorizer = TfidfVectorizer()
        all_text = list(bench["skills_profile"]) + [project_description]
        tfidf_matrix = vectorizer.fit_transform(all_text)
        scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        bench["Skill_Score"] = scores

        # Hybrid score (70% skill + 30% experience)
        bench["Experience_Score"] = (bench["experience_years"] / required_experience).clip(upper=1.0)
        bench["Final_Score"] = 0.7 * bench["Skill_Score"] + 0.3 * bench["Experience_Score"]

        result = bench.sort_values(by="Final_Score", ascending=False).head(5)

        st.subheader("Top Matched Employees")
        st.dataframe(
            result[["employee_name", "skills_profile", "experience_years",
                    "Skill_Score", "Final_Score"]].round(2),
            use_container_width=True
        )