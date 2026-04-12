import streamlit as st
import numpy as np
import joblib

# ===== LOAD MODEL FILES =====
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
cpt_encoder = joblib.load("cpt_encoder.pkl")

# ===== FUNCTION =====
def get_top_cpt_predictions(note, top_n=3):
    vec = vectorizer.transform([note])
    probs = model.predict_proba(vec)[0]

    top_indices = np.argsort(probs)[::-1][:top_n]

    return {
        "input_text": note,
        "CPT_suggestions": cpt_encoder.inverse_transform(top_indices),
        "confidence": [round(probs[i], 3) for i in top_indices]
    }

# ===== UI =====
st.set_page_config(page_title="AI Clinical Coding Assistant")

st.title("🏥 AI Clinical Coding Assistant")
st.write("Enter a clinical note and get CPT code suggestions.")

# Input
note = st.text_area("Clinical Note:", height=150)

# Button
if st.button("Generate CPT Suggestions"):

    if note.strip() == "":
        st.warning("Please enter a clinical note.")
    else:
        result = get_top_cpt_predictions(note)

        st.success("AI Model generated CPT recommendations ✅")

        st.subheader("📊 Results")

        for i in range(len(result["CPT_suggestions"])):
            cpt = result["CPT_suggestions"][i]
            conf = result["confidence"][i]
            st.write(f"**{i+1}. CPT Code:** {cpt} | **Confidence:** {conf}")

        st.subheader("🔗 JSON Output")
        st.json(result)