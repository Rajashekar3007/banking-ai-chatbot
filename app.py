import json
import pickle
import streamlit as st

# Load model files
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Load intents
with open("intents.json") as file:
    data = json.load(file)


def get_response(user_input):
    input_data = vectorizer.transform([user_input])
    prediction = model.predict(input_data)
    tag = encoder.inverse_transform(prediction)[0]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return intent["responses"][0]


# Page config
st.set_page_config(page_title="Banking AI Chatbot")

st.title("🏦 Banking AI Chatbot")
st.write("AI-powered virtual banking assistant")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input
user_input = st.chat_input("Ask your banking question...")

if user_input:
    bot_response = get_response(user_input)

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    st.session_state.messages.append(
        {"role": "bot", "content": bot_response}
    )

# Display chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])