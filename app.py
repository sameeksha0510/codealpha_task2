import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

st.set_page_config(page_title="AI FAQ Chatbot", page_icon="🤖", layout="centered")

# ---------- Styling ----------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.chat-box {
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 12px;
}
.user-msg {
    background-color: #dbeafe;
}
.bot-msg {
    background-color: #dcfce7;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sample FAQ Data ----------
faq_data = {
    "Category": [
        "Admission", "Admission", "Fees", "Hostel", "Contact",
        "Placement", "Internship", "Timing", "Certificate", "Support"
    ],
    "Question": [
        "How can I apply for admission?",
        "What documents are required for admission?",
        "What is the fee structure?",
        "Is hostel facility available?",
        "How can I contact support?",
        "Do you provide placement assistance?",
        "How can I apply for internship?",
        "What are your working hours?",
        "Will I get a certificate after completion?",
        "Where can I get help if I face issues?"
    ],
    "Answer": [
        "You can apply for admission through the official website by filling the application form.",
        "You need ID proof, academic certificates, passport-size photo, and application form.",
        "The fee structure depends on the course. Please check the official fee details or contact support.",
        "Yes, hostel facilities are available for both boys and girls based on availability.",
        "You can contact support through email or phone provided on the official website.",
        "Yes, placement assistance and career guidance are provided to eligible students.",
        "You can apply for internship by filling the internship application form and submitting required details.",
        "Our working hours are Monday to Friday, 9 AM to 6 PM.",
        "Yes, you will receive a certificate after successful completion of the program.",
        "You can contact the support team through email for any technical or general issues."
    ]
}

df = pd.DataFrame(faq_data)

# ---------- Text Preprocessing ----------
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df["Clean_Question"] = df["Question"].apply(preprocess_text)

# ---------- TF-IDF Model ----------
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(df["Clean_Question"])

def get_best_answer(user_question):
    clean_question = preprocess_text(user_question)
    user_vector = vectorizer.transform([clean_question])

    similarity = cosine_similarity(user_vector, faq_vectors)
    best_index = np.argmax(similarity)
    confidence = similarity[0][best_index]

    if confidence < 0.25:
        return (
            "Sorry, I couldn't find a proper answer for your question. Please contact support.",
            confidence,
            "Unknown"
        )

    return (
        df.iloc[best_index]["Answer"],
        confidence,
        df.iloc[best_index]["Category"]
    )

# ---------- App UI ----------
st.title("🤖 AI FAQ Chatbot")
st.write("Ask a question and the chatbot will find the best matching FAQ answer.")

st.sidebar.title("📂 FAQ Categories")
selected_category = st.sidebar.selectbox(
    "Choose category",
    ["All"] + sorted(df["Category"].unique().tolist())
)

if selected_category != "All":
    st.sidebar.write(df[df["Category"] == selected_category][["Question"]])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("💬 Ask your question:")

col1, col2 = st.columns(2)

with col1:
    ask_button = st.button("Ask")

with col2:
    clear_button = st.button("Clear Chat")

if clear_button:
    st.session_state.chat_history = []
    st.success("Chat cleared!")

if ask_button:
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        answer, confidence, category = get_best_answer(user_input)

        st.session_state.chat_history.append({
            "user": user_input,
            "bot": answer,
            "confidence": confidence,
            "category": category
        })

# ---------- Display Chat ----------
st.subheader("💬 Chat History")

for chat in st.session_state.chat_history:
    st.markdown(
        f"""
        <div class="chat-box user-msg">
        <b>👤 You:</b><br>{chat['user']}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="chat-box bot-msg">
        <b>🤖 Bot:</b><br>{chat['bot']}<br><br>
        <b>Category:</b> {chat['category']}<br>
        <b>Confidence:</b> {round(chat['confidence'] * 100, 2)}%
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(float(chat["confidence"]))

# ---------- Download Chat ----------
if st.session_state.chat_history:
    chat_text = ""

    for chat in st.session_state.chat_history:
        chat_text += f"You: {chat['user']}\n"
        chat_text += f"Bot: {chat['bot']}\n"
        chat_text += f"Category: {chat['category']}\n"
        chat_text += f"Confidence: {round(chat['confidence'] * 100, 2)}%\n\n"

    st.download_button(
        label="📥 Download Chat History",
        data=chat_text,
        file_name="chat_history.txt",
        mime="text/plain"
    )

# ---------- FAQ Table ----------
with st.expander("📋 View All FAQs"):
    st.dataframe(df[["Category", "Question", "Answer"]])