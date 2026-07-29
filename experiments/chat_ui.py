import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="Groq Chatbot", page_icon="🤖")
st.title("🤖 Groq Chatbot (Llama 3.3)")

# --- Initialize model once ---
@st.cache_resource
def get_model():
    return init_chat_model("groq:llama-3.3-70b-versatile")

try:
    model = get_model()
except Exception as e:
    st.error(f"Failed to initialize Groq model: {e}")
    st.info("Make sure GROQ_API_KEY is set in your .env file.")
    st.stop()

# --- Initialize chat history in session state ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful, friendly AI assistant powered by Groq.")
    ]

# --- Render existing conversation (skip the system message) ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# --- Chat input ---
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Get and show AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = model.invoke(st.session_state.messages)
                ai_text = getattr(response, "content", str(response))
                st.markdown(ai_text)
                st.session_state.messages.append(response)
            except Exception as e:
                st.error(f"Error during generation: {e}")

# --- Sidebar: clear chat ---
with st.sidebar:
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = [
            SystemMessage(content="You are a helpful, friendly AI assistant powered by Groq.")
        ]
        st.rerun()