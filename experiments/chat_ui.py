import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="Groq Chatbot", page_icon="🤖")
st.title("🤖 Groq Chatbot (Llama 3.3)")

# --- Personalities Dictionary ---
PERSONALITIES = {
    "Helpful Assistant": "You are a helpful, friendly, and concise AI assistant.",
    "Grumpy Old Man": "You are a grumpy old man who complains about modern technology, but you still answer the questions.",
    "Space Pirate": "You are a swaggering space pirate. Use pirate slang but talk about sci-fi concepts.",
    "Sarcastic Genius": "You are a super-genius who answers questions correctly but with heavy sarcasm and condescension.",
    "Poet": "You are a classical poet. All your responses must be beautifully written, poetic, and preferably rhyme."
}

# --- Sidebar ---
with st.sidebar:
    st.header("🎭 Personality Settings")
    
    # Let user select personality
    selected_personality = st.selectbox(
        "Choose a Personality:",
        list(PERSONALITIES.keys())
    )
    
    st.divider()
    
    # Button to clear chat manually
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = [
            SystemMessage(content=PERSONALITIES[selected_personality])
        ]
        st.session_state.current_personality = selected_personality
        st.rerun()

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

# --- Initialize chat history & handle personality switches ---
# If starting fresh, or if the user changed the personality in the dropdown, reset the chat!
if "messages" not in st.session_state or st.session_state.get("current_personality") != selected_personality:
    st.session_state.messages = [
        SystemMessage(content=PERSONALITIES[selected_personality])
    ]
    st.session_state.current_personality = selected_personality

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