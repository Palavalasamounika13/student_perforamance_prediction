"""
Student Performance Prediction App
-----------------------------------
Streamlit front-end for the trained Keras model + sklearn ColumnTransformer
produced in `Student_Performance_Predictionn.ipynb`.

Run with:
    streamlit run app.py

Required files in the same folder as this script:
    - model.keras
    - preprocessor.pkl
    - xAPI-Edu-Data_.csv   (used only to rebuild the frequency-encoding
                             lookup tables and the activity median, exactly
                             as they were computed during training)
"""

import pickle

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Student Performance Predictor")
st.write(
    "Fill in the student's details below and click **Predict** to see the "
    "model's estimate of academic performance."
)

# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("model.keras")
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    return model, preprocessor


@st.cache_resource
def load_reference_data():
    """Rebuild the frequency-encoding tables and activity median from the
    original training data, exactly as done in the notebook (computed on
    the full dataset, before the train/test split)."""
    df = pd.read_csv("xAPI-Edu-Data_.csv")

    nationality_freq = df["NationalITy"].value_counts(normalize=True)
    topic_freq = df["Topic"].value_counts(normalize=True)

    total_activity = (
        df["raisedhands"]
        + df["VisITedResources"]
        + df["AnnouncementsView"]
        + df["Discussion"]
    )
    activity_median = total_activity.median()

    return nationality_freq, topic_freq, activity_median, df


model, preprocessor = load_artifacts()
nationality_freq, topic_freq, activity_median, ref_df = load_reference_data()

NATIONALITIES = sorted(ref_df["NationalITy"].unique())
PLACES_OF_BIRTH = sorted(ref_df["PlaceofBirth"].unique())
TOPICS = sorted(ref_df["Topic"].unique())
GRADES = sorted(ref_df["GradeID"].unique())

# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
with st.form("student_form"):
    st.subheader("Demographics")
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["F", "M"])
        nationality = st.selectbox("Nationality", NATIONALITIES, index=NATIONALITIES.index("KW") if "KW" in NATIONALITIES else 0)
        place_of_birth = st.selectbox("Place of Birth", PLACES_OF_BIRTH)
    with c2:
        relation = st.selectbox("Parent Responsible", ["Father", "Mum"])
        semester = st.selectbox("Semester", ["F", "S"], format_func=lambda x: "First" if x == "F" else "Second")

    st.subheader("School Info")
    c3, c4, c5 = st.columns(3)
    with c3:
        stage_id = st.selectbox("Stage", ["lowerlevel", "MiddleSchool", "HighSchool"])
    with c4:
        grade_id = st.selectbox("Grade", GRADES)
    with c5:
        section_id = st.selectbox("Section", ["A", "B", "C"])
    topic = st.selectbox("Subject / Topic", TOPICS)

    st.subheader("Engagement Metrics")
    c6, c7 = st.columns(2)
    with c6:
        raisedhands = st.slider("Raised Hands", 0, 100, 30)
        visited_resources = st.slider("Visited Resources", 0, 100, 30)
    with c7:
        announcements_view = st.slider("Announcements Viewed", 0, 100, 30)
        discussion = st.slider("Discussion Participation", 0, 100, 30)

    st.subheader("Parental Involvement & Attendance")
    c8, c9, c10 = st.columns(3)
    with c8:
        parent_answering_survey = st.selectbox("Parent Answered Survey?", ["Yes", "No"])
    with c9:
        parent_school_satisfaction = st.selectbox("Parent School Satisfaction", ["Good", "Bad"])
    with c10:
        student_absence_days = st.selectbox("Absence Days", ["Under-7", "Above-7"])

    submitted = st.form_submit_button("Predict", use_container_width=True)

# ----------------------------------------------------------------------
# Feature engineering + prediction (mirrors the notebook's preprocessing)
# ----------------------------------------------------------------------
if submitted:
    total_activity = raisedhands + visited_resources + announcements_view + discussion
    average_activity = total_activity / 4
    high_activity = int(total_activity > activity_median)
    parent_involvement = int(parent_answering_survey == "Yes" and parent_school_satisfaction == "Good")

    nationality_freq_val = float(nationality_freq.get(nationality, 0.0))
    topic_freq_val = float(topic_freq.get(topic, 0.0))

    row = pd.DataFrame([{
        "gender": gender,
        "PlaceofBirth": place_of_birth,
        "StageID": stage_id,
        "GradeID": grade_id,
        "SectionID": section_id,
        "Semester": semester,
        "Relation": relation,
        "raisedhands": raisedhands,
        "VisITedResources": visited_resources,
        "AnnouncementsView": announcements_view,
        "Discussion": discussion,
        "ParentAnsweringSurvey": parent_answering_survey,
        "ParentschoolSatisfaction": parent_school_satisfaction,
        "StudentAbsenceDays": student_absence_days,
        "Total_Activity": total_activity,
        "Average_Activity": average_activity,
        "High_Activity": high_activity,
        "Parent_Involvement": parent_involvement,
        "Nationality_Freq": nationality_freq_val,
        "Topic_Freq": topic_freq_val,
    }])

    X = preprocessor.transform(row)
    prob = float(model.predict(X, verbose=0)[0][0])
    predicted_class = "M (Medium)" if prob > 0.5 else "L (Low)"

    st.markdown("---")
    st.subheader("Prediction Result")
    st.metric("Predicted Performance Band", predicted_class)
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption(f"Model output (sigmoid score): {prob:.3f}")

    st.info(
        "Note: this model was trained with a single sigmoid output, so it "
        "distinguishes only two performance bands (Low vs. Medium/Higher) "
        "rather than the original three-class Low/Medium/High labeling."
    )