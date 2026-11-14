import joblib
import streamlit as st

st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️")

@st.cache_resource
def load_model():
    return joblib.load("spam_model.joblib")

model = load_model()

st.title("🛡️ SpamGuard AI")
st.caption("Intelligent Email Protection")

text = st.text_area("Apna email message check karo", height=180,
                    placeholder="Yahan email ya message paste karo...")

if st.button("Analyze Email"):
    if not text.strip():
        st.warning("Pehle kuch text likho.")
    else:
        spam_prob = model.predict_proba([text])[0][1]
        if spam_prob >= 0.5:
            st.error(f"🚨 Likely Spam ({spam_prob*100:.1f}% confidence)")
        else:
            st.success(f"✅ Looks Normal ({(1-spam_prob)*100:.1f}% confidence)")
        st.progress(float(spam_prob), text="Spam score")