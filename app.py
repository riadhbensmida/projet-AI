# pyrefly: ignore [missing-import]
import streamlit as st
import os
import config
from pipeline.rag_pipeline import RAGPipeline

# Page Configuration
st.set_page_config(
    page_title="CareerPath AI",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .chat-bubble {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-bubble {
        background-color: #e9ecef;
    }
    .ai-bubble {
        background-color: #d1ecf1;
        border-left: 5px solid #0c5460;
    }
    .source-box {
        font-size: 0.8em;
        color: #6c757d;
        background-color: #f1f1f1;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.title("🎓 CareerPath AI")
st.subheader("Your Intelligent Career & Internship Assistant")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/student-center.png", width=100)
    st.header("Settings")
    
    api_key = st.text_input("Groq API Key", value=config.GROQ_API_KEY, type="password")
    if api_key:
        config.GROQ_API_KEY = api_key
        os.environ["GROQ_API_KEY"] = api_key

    st.markdown("---")
    st.markdown("### 📚 Knowledge Base")
    files = os.listdir(config.CORPUS_DIR)
    for f in files:
        st.write(f"- {f}")
    
    if st.button("🔄 Rebuild Knowledge Base"):
        with st.spinner("Processing documents and building index..."):
            pipeline = RAGPipeline()
            pipeline.initialize_system(force_rebuild=True)
            st.success("Knowledge base rebuilt successfully!")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask me about internships, skills, or career paths..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check for API key
    if not config.GROQ_API_KEY or config.GROQ_API_KEY == "gsk_your_api_key_here":
        st.error("Please provide a valid Groq API Key in the sidebar.")
    else:
        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    pipeline = RAGPipeline()
                    # Ensure system is initialized
                    pipeline.initialize_system()
                    
                    response, sources = pipeline.run(prompt)
                    st.markdown(response)
                    
                    # Display Sources in an Expander
                    if sources:
                        with st.expander("🔍 View Sources"):
                            for i, doc in enumerate(sources):
                                st.markdown(f"**Source {i+1}** (from {os.path.basename(doc.metadata.get('source', 'Unknown'))}):")
                                st.info(doc.page_content)
                    
                    # Add assistant message to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>Built for AI Generative Module | Team: Riadh Ben Smida & Yossr Ben Bagou</div>", 
    unsafe_allow_html=True
)
