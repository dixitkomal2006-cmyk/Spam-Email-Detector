import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# -------------------- Title --------------------
st.title("📧 Spam Email Detector")

# -------------------- Dataset --------------------
emails = [
    "Win a free iPhone now",
    "Meeting at 11 am tomorrow",
    "Congratulations you won lottery",
    "Project discussion with team",
    "Claim your prize immediately",
    "Find the attached report",
    "Urgent offer expires today",
    "Schedule the meeting for Monday",
    "You have won a cash prize",
    "Monthly performance report attached",
    "Exclusive deal just for you"
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 1 = Spam, 0 = Not Spam

# -------------------- Vectorization --------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=1
)

X = vectorizer.fit_transform(emails)

# -------------------- Train/Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42
)

# -------------------- Model --------------------
model = LinearSVC(C=1.0, random_state=42)
model.fit(X_train, y_train)

# -------------------- Accuracy --------------------
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f"### Model Accuracy: **{accuracy:.2f}**")

# -------------------- Class Distribution Graph --------------------
st.subheader("Dataset Distribution")

df = pd.DataFrame({
    "Class": ["Spam" if label == 1 else "Not Spam" for label in labels]
})

st.bar_chart(df["Class"].value_counts())

# -------------------- User Input --------------------
st.subheader("Check Your Email")

user_msg = st.text_area("Enter Email Message")

if st.button("Check"):
    if user_msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        msg_vec = vectorizer.transform([user_msg])
        pred = model.predict(msg_vec)[0]

        if pred == 1:
            st.error("Result: 🚨 Spam Email")
        else:
            st.success("Result: ✅ Not Spam")
