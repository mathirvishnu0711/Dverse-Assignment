import streamlit as st  # Streamlit import should come after set_page_config
st.set_page_config(page_title="FAQ Chatbot", page_icon="💬")  # This must be the first Streamlit command

import pandas as pd
import torch
import asyncio
from sentence_transformers import SentenceTransformer, util

# Ensure there is an active event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load FAQ dataset
faq_file = "Tata_comm_faq.csv"

@st.cache_data
def load_faq_data(file):
    df = pd.read_csv(file)
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("Error: The CSV file must contain 'question' and 'answer' columns.")
    return df

faq_df = load_faq_data(faq_file)

# Load BERT-based model for encoding
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Encode the questions in the FAQ dataset
faq_questions = faq_df["question"].tolist()

@st.cache_data
def encode_faq_questions(questions):
    return model.encode(questions, convert_to_tensor=True)

faq_embeddings = encode_faq_questions(faq_questions)

def get_best_match(user_query, threshold=0.5):
    """Find the best matching question using cosine similarity."""
    try:
        user_embedding = model.encode(user_query, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(user_embedding, faq_embeddings)[0]
        best_match_idx = torch.argmax(similarities).item()
        similarity_score = similarities[best_match_idx].item()

        if similarity_score < threshold:
            return "Sorry, I don't understand your question. Please try rephrasing.", similarity_score

        return faq_df.iloc[best_match_idx]["answer"], similarity_score
    except Exception as e:
        return f"Error processing query: {str(e)}", 0.0

# Streamlit UI
st.title("💬 FAQ Chatbot")
st.markdown("### Ask me anything related to Tata Communications!")

# Chat UI using Streamlit Chat Elements
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input section
if user_input := st.chat_input("Type your question here..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    
    response, score = get_best_match(user_input)

    with st.chat_message("assistant"):
        st.markdown(f"**Chatbot:** {response}")
        st.markdown(f"_(Confidence Score: {score:.2f})_")

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Expandable FAQ Data
with st.expander("📖 View FAQ Data"):
    st.dataframe(faq_df)
