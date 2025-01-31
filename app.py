import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

# Custom CSS styling
st.markdown("""
<style>
    /* Existing styles */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    
    /* Add these new styles for select box */
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    
    .stSelectbox svg {
        fill: white !important;
    }
    
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    /* For dropdown menu items */
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("DeepSeek code Companion")
st.caption("Your AI Pair Programer with debuging Superpowers")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:1.5b", "deepseek-r1:3b"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# Initiate chat engine
llm_engine = ChatOllama(
    model = selected_model,
    base_url = 'http://localhost:11434',
    temperature = 0.3
)

# system prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions"
    "with startegic print statements for debugging. Always respond in English."
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role":"ai", "content":"Hi! I'm DeepSeek. How I can help you code Today !"}]


# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# chat input and processing
user_query = st.chat_input("Type you coding question here")

def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    
    # Debugging: Print what the processing pipeline looks like
    print("Processing Pipeline:", processing_pipeline)

    try:
        response = processing_pipeline({"input": "Test"})  # Send a test input
        print("AI Response:", response)  # Debugging output
        return response
    except Exception as e:
        print(f"Error: {e}")
        st.error(f"Error generating AI response: {e}")
        return "Sorry, an error occurred."

try:
    test_output = llm_engine.invoke("Hello AI")
    print("Test Output from LLM:", test_output)
except Exception as e:
    print(f"Error in LLM Engine: {e}")



def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] =="user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

prompt_chain = build_prompt_chain()
print("Prompt Chain:", prompt_chain)


if user_query:
    #Add user message to log
    st.session_state.message_log.append({"role":"user","content":user_query})

    # Generate AI response
    with st.spinner("...processing"):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)

    # Add AI response to log
    st.session_state.message_log.append({"role":"ai","content":ai_response})

    # rerun the chat update
    st.rerun

    