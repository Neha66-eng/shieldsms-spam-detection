"""Streamlit interface for SmartSMS Shield."""
import streamlit as st
from predict import load_model, predict_message

st.set_page_config(page_title="SmartSMS Shield", page_icon="🛡️", layout="centered")
st.title("🛡️ SmartSMS Shield")
st.subheader("SMS Spam Detection")
st.write("Enter a message to classify it as **Spam** or **Ham (legitimate)**.")
st.caption("Student demonstration using TF-IDF and Multinomial Naive Bayes.")

@st.cache_resource
def get_model():
    return load_model()

message = st.text_area("SMS message", placeholder="Example: Congratulations! You have won a prize...", height=130)

if st.button("Check message", type="primary"):
    if not message.strip():
        st.warning("Please enter an SMS message first.")
    else:
        try:
            result = predict_message(message, get_model())
            if result["label"] == "spam":
                st.error("Prediction: SPAM 🚨")
            else:
                st.success("Prediction: HAM / LEGITIMATE ✅")
            if result["spam_probability"] is not None:
                st.metric("Estimated spam probability", f"{result['spam_probability']:.1%}")
            st.caption("This is a model estimate, not a guarantee. Verify suspicious messages independently.")
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.info("Download the dataset and run `python train.py` before starting the app.")
