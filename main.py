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
import os

openai_api_key = st.secrets['OPENAI_API_KEY']
# Define CSS for the UI
css = """
<style>
body, html, [data-testid="stAppViewContainer"] {
    background-color: #00042C !important;
    color: #ffffff !important;
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    height: 100%;
}
header {
    background-color: #00042C !important; /* Dark background color */
    color: #ffffff !important; /* White text */
    border-bottom: 1px solid #444444; /* Optional: Add a subtle border */
}

/* Style the 'Deploy' button text */
header button {
    color: #ffffff !important; /* Change text color to white */
    font-weight: bold; /* Make the text bold */
}

/* Adjust padding and alignment for a cleaner look */
header [data-testid="stHeader"] {
    padding: 10px 20px !important;
    text-align: left; /* Align content to the left */
}

/* Remove the default shadow */
header {
    box-shadow: none !important;
}
/* Global Background */
body {
    background-color: #00042C !important;
    color: #ffffff !important;
    font-family: Arial, sans-serif;
    margin:100px;
    padding: 100px;
    height: 100%;
}

/* Main Container */
.main .block-container {
    background-color: #00042C !important;
    color: #ffffff !important;
    margin:0;
    padding: 50px;
    max-width: 100%;
    margin: 50;
    height: 100%;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #00042C !important;
    color: #ffffff !important;
    border-right: 1px solid #555555;
    height: 100vh; /* Ensure it spans the full height */
}

/* Sidebar Headers */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

# /* Welcome Message */
# .intro-message {
#     background: linear-gradient(90deg, #1a73e8, #0059c1);
#     border-radius: 8px;
#     padding: 15px;
#     color: #ffffff;
#     font-size: 18px;
#     font-weight: bold;
#     text-align: center;
#     margin-bottom: 20px;
# }

/* Buttons */
.stButton>button {
    background-color:#bad5f8;
    color: #000102;
    border-radius: 8px;
    font-size: 16px;
    font-weight: bold;
    padding: 10px 20px;
    border: none;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    transition: background-color 0.3s ease;
}

.stButton>button:hover {
    background-color: #00042C;
    cursor: pointer;
}

# /* Input Fields */
# .stTextInput, .stFileUploader {
#     background-color: #1f1f3b !important;
#     color: #ffffff !important;
#     border: 1px solid #555555;
#     border-radius: 5px;
#     padding: 10px;
#     font-size: 14px;
# }

.stTextInput::placeholder {
    color: #ffffff !important; /* Lighter placeholder text */
}
.chat-bubble {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
}
.center-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 40vh;
}


/* Chat Container */
.chat-container {
    background-color: #12122e;
    color: #ffffff !important;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
    max-height: 400px;
    overflow-y: auto;
}

.bot-message {
    background-color: #ff4466;
    padding: 0.8rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    width: fit-content;
    max-width: 70%;
    display: inline-block;
}
.user-message {
    background-color: #00bb99;
    padding: 0.8rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    width: fit-content;
    max-width: 70%;
    margin-left: auto;
    display: inline-block;
}


/* Scrollbars */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #444444;
    border-radius: 5px;
}
::-webkit-scrollbar-track {
    background: #00042C;
}

/* Full-Height Fix */
html, body, [data-testid="stAppViewContainer"] {
    height: 100%;
}

[data-testid="stSidebar"], .main .block-container {
    height: 100%;
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

    # Generate response from the conversation chain
    response = st.session_state.conversation({"question": user_question})
    bot_reply = response["answer"]

    # Append user and bot messages to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "bot", "content": bot_reply})

    # Convert the bot's response to speech
    tts = gTTS(text=bot_reply, lang="en")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)

    # Auto-play audio in the background using JavaScript
    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/mpeg;base64,{base64.b64encode(open(temp_file.name, "rb").read()).decode()}" type="audio/mpeg">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)



def play_intro_voice(message):
    """
    Converts the intro message into speech and saves it as an MP3 file.
    Returns the path to the MP3 file.
    """
    tts = gTTS(text=message, lang='en')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name
def get_pdf_text_from_path(file_paths):
    """Extract text from PDF files given their paths."""
    text = ""
    for path in file_paths:
        with open(path, "rb") as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text


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
    
    file_ = open("original-6a083ad3c3053550e566352b84cb7412.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
    f'''
    <div class="center-container">
        <img src="data:image/gif;base64,{data_url}" alt="cat gif">
    </div>
    ''',
    unsafe_allow_html=True,
)

    #st.title("📄 Sakhi Bot ")
    

    # Intro Message
    intro_message = (
        "Hello, I am your Sakhi Chatbot. Please upload your PDF documents and start asking questions. "
        "I will do my best to answer your questions based on the document content."
    )
    st.markdown(
        f"""
        <div style="
            background-color:rgb(255, 255, 255);
            border-left: 8px solidrgb(34, 0, 255);
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            font-family: 'Verdana', sans-serif;
            font-size: 18px;
            color: #5d4037;
        ">
            <h3 style="color:rgb(7, 1, 59); font-weight: bold;">✨ Welcome to Your PDF Chat Assistant!</h3>
            <p style="line-height: 1.6;">{intro_message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Predefined file paths
    pdf_file_paths = [
        "Brand Space.pdf" # Replace with the actual path
           # Add more files if needed
    ]

    # Process PDFs
    with st.spinner("Processing PDF files..."):
        raw_text = get_pdf_text_from_path(pdf_file_paths)
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.success("Files processed successfully! You can now ask questions.")

 

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
    #st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
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

