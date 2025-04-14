import streamlit as st
from utils import extract_text_from_image, evaluate_formulas

st.set_page_config(page_title="ML Formula Evaluator", layout="centered")

st.title("🧪 ML Formula Evaluator from Image")

st.write("Upload an image of handwritten formulas for Accuracy, Precision, Recall, and F1-score.")

uploaded_file = st.file_uploader("📷 Upload image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    with st.spinner("Processing image..."):
        extracted_text = extract_text_from_image(uploaded_file)

        st.subheader("📄 Extracted Text")
        st.code(extracted_text)

        st.subheader("🤖 Gemini Evaluation")
        evaluation = evaluate_formulas(extracted_text)

        st.markdown(evaluation)
