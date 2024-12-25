import base64
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from gtts import gTTS
import tempfile

# Define CSS for the UI
css = """
<style>
.input-container {
        display: flex;
        align-items: center;
        background-color: #f9f9f9;
        border-radius: 25px;
        padding: 10px 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .input-container input {
        border: none;
        background: transparent;
        flex-grow: 1;
        font-size: 16px;
        outline: none;
    }
    .input-container button {
        background: #000;
        color: #fff;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        padding: 0;
    }
    .input-container button img {
        width: 20px;
        height: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .main .block-container {
        padding: 2rem;
        max-width: 70%;
        margin: auto;
    }
    .stButton>button {
        background-color: #D2E3F5;
        color: black;
    }
    .chat-container {
        background-color: #eef2fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        max-height: 400px;
        overflow-y: auto;
    }
    .chat-bubble {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .bot-message {
        background-color: #d1e7fd;
        padding: 0.8rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        width: fit-content;
        max-width: 70%;
        display: inline-block;
    }
    .user-message {
        background-color: #ffe4e1;
        padding: 0.8rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        width: fit-content;
        max-width: 70%;
        margin-left: auto;
        display: inline-block;
    }
    .chat-emoji {
        width: 30px;
        height: 30px;
        margin-right: 10px;
    }
    .user-emoji {
        margin-left: 10px;
        margin-right: 0;
    }
</style>
"""

# Helper Functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_user_input(user_question):
    if not st.session_state.conversation:
        st.warning("Please upload and process your files first!")
        return

    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "bot", "content": response["answer"]})


def play_intro_voice(message):
    """
    Converts the intro message into speech and saves it as an MP3 file.
    Returns the path to the MP3 file.
    """
    tts = gTTS(text=message, lang='en')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name


import speech_recognition as sr  # Import for voice input handling

# Function to handle voice input
def record_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Please speak now.")
        try:
            audio = recognizer.listen(source, timeout=5)
            st.info("🔊 Processing your voice input...")
            query = recognizer.recognize_google(audio)
            return query
        except sr.UnknownValueError:
            st.error("Could not understand the audio. Please try again.")
        except sr.RequestError as e:
            st.error(f"Error with the speech recognition service: {e}")
    return None


def main():
    load_dotenv()
    st.set_page_config(page_title="Sakhi Bot", page_icon="📄", layout="wide")
    st.markdown(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("📄 Chat with PDFs")
    

    # Intro Message
    intro_message = (
        "Hello, I am your Sakhi Chatbot. Please upload your PDF documents and start asking questions. "
        "I will do my best to answer your questions based on the document content."
    )
    st.markdown(
        f"""
        <div style="
            background-color: #fff8e1;
            border-left: 8px solid #ff9800;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            font-family: 'Verdana', sans-serif;
            font-size: 18px;
            color: #5d4037;
        ">
            <h3 style="color: #e65100; font-weight: bold;">✨ Welcome to Your PDF Chat Assistant!</h3>
            <p style="line-height: 1.6;">{intro_message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar for Upload
    with st.sidebar:
        st.header("📂 Upload Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here:", accept_multiple_files=True
        )
        if st.button("Process Files"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Files processed successfully! You can now ask questions.")
            else:
                st.warning("⚠️ Please upload at least one PDF file.")

    # Chat Section
    user_question = st.text_input("Ask a Question:", placeholder="Type your question here...")
    if st.button("Send"):
        if user_question.strip():
            handle_user_input(user_question)
        else:
            st.warning("⚠️ Please enter a valid question.")

    # Voice Input Section
    if st.button("🎤 "):
        voice_input = record_voice()
        if voice_input:
            st.success(f"Recognized Question: {voice_input}")
            handle_user_input(voice_input)

    # Chat History
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(
                f"""
                <div class="chat-bubble">
                    <div class="user-message">{message["content"]}</div>
                    <img src="https://em-content.zobj.net/thumbs/240/apple/354/person_1f9d1.png" alt="User Emoji" class="chat-emoji user-emoji">
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="chat-bubble">
                    <img src="https://em-content.zobj.net/thumbs/240/apple/354/robot_1f916.png" alt="Bot Emoji" class="chat-emoji">
                    <div class="bot-message">{message["content"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()


