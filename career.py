import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Career Planning Chatbot", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []

csv_path = "updated_qa_career_dataset (1).csv"
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    st.error(f"Failed to load the CSV file. Error: {e}")
    st.stop()

df = df.fillna("")
df['Question'] = df['Question'].str.lower()
df['Answer'] = df['Answer'].str.lower()

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df['Question'])

API_KEY = "AIzaSyDnB1GZeg_gMap8aFPdgqgL1eQmqI1E7PU"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def find_closest_question(user_query, vectorizer, question_vectors, df):
    query_vector = vectorizer.transform([user_query.lower()])
    similarities = cosine_similarity(query_vector, question_vectors).flatten()
    best_match_index = similarities.argmax()
    best_match_score = similarities[best_match_index]
    if best_match_score > 0.3:
        return df.iloc[best_match_index]['Answer']
    return None

def generate_detailed_response(user_query):
    """Generates an AI-driven response with a detailed explanation."""
    try:
        response = model.generate_content(user_query)
        return response.text
    except Exception as e:
        return f"Sorry, I couldn't generate a response. Error: {e}"

st.title("Career Planning Chatbot")
st.write("Welcome! Ask me anything about colleges, courses, and career planning.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    closest_answer = find_closest_question(prompt, vectorizer, question_vectors, df)
    if closest_answer:
        response = f"**Here's what I found:**\n\n{closest_answer}\n\n*Let me know if you need more details!*"
    else:
        response = generate_detailed_response(prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
