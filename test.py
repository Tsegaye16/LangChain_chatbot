import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import random
import json

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Dorner's Psi Theory Parameters
class CharacterState:
    def __init__(self, arousal=0.5, valence=0.5, dominance=0.5, sadness=0.0, anger=0.0, joy=0.0, fear=0.0,
                 selection_threshold=0.5, resolution_level=0.5, goal_directedness=0.5, securing_rate=0.5, long_term_memory={}):
        self.arousal = arousal
        self.valence = valence
        self.dominance = dominance
        self.sadness = sadness
        self.anger = anger
        self.joy = joy
        self.fear = fear
        self.selection_threshold = selection_threshold
        self.resolution_level = resolution_level
        self.goal_directedness = goal_directedness
        self.securing_rate = securing_rate
        self.long_term_memory = long_term_memory

    def update_emotions(self, emotion_data):
        self.arousal = emotion_data.get("arousal", self.arousal)
        self.valence = emotion_data.get("valence", self.valence)
        self.dominance = emotion_data.get("dominance", self.dominance)
        self.sadness = emotion_data.get("sadness", self.sadness)
        self.anger = emotion_data.get("anger", self.anger)
        self.joy = emotion_data.get("joy", self.joy)
        self.fear = emotion_data.get("fear", self.fear)
        self.selection_threshold = emotion_data.get("selection_threshold", self.selection_threshold)
        self.resolution_level = emotion_data.get("resolution_level", self.resolution_level)
        self.goal_directedness = emotion_data.get("goal_directedness", self.goal_directedness)
        self.securing_rate = emotion_data.get("securing_rate", self.securing_rate)

    def display_emotions(self):
        st.write("### Emotional State")
        emotions = {
            "Arousal": self.arousal,
            "Valence": self.valence,
            "Dominance": self.dominance,
            "Sadness": self.sadness,
            "Anger": self.anger,
            "Joy": self.joy,
            "Fear": self.fear,
            "Selection Threshold": self.selection_threshold,
            "Resolution Level": self.resolution_level,
            "Goal Directedness": self.goal_directedness,
            "Securing Rate": self.securing_rate,
        }
        for emotion, value in emotions.items():
            st.write(f"{emotion}:")
            st.progress(max(0.0, min(1.0, value)))  # Ensure value is within [0.0, 1.0]

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are a character from a book, engaging in a conversation. Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, respond in a way that acknowledges the question and suggests a lack of information.
    Be conversational and contextual in your responses.

    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def extract_characters(text):
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    prompt = f"Extract a list of characters from the following text: {text}. Return only a list of names separated by comma."
    response = model.invoke(prompt)
    characters = [char.strip() for char in response.content.split(',')]
    return characters

def simulate_emotions(user_input, character_state):
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    prompt = f"Given the user input: '{user_input}', and the character's current state (Arousal: {character_state.arousal}, Valence: {character_state.valence}, Dominance: {character_state.dominance}, Sadness: {character_state.sadness}, Anger: {character_state.anger}, Joy: {character_state.joy}, Fear: {character_state.fear}, Selection Threshold: {character_state.selection_threshold}, Resolution Level: {character_state.resolution_level}, Goal Directedness: {character_state.goal_directedness}, Securing Rate: {character_state.securing_rate}), generate new values for these parameters and describe the character's emotional state in a json format."
    response = model.invoke(prompt)

    try:
        emotion_data = eval(response.content)
        character_state.update_emotions(emotion_data)
    except (SyntaxError, NameError, TypeError):
        # Handle cases where the LLM's response is not valid JSON
        character_state.arousal = max(0.0, min(1.0, character_state.arousal + random.uniform(-0.1, 0.1)))
        character_state.valence = max(0.0, min(1.0, character_state.valence + random.uniform(-0.1, 0.1)))
        character_state.dominance = max(0.0, min(1.0, character_state.dominance + random.uniform(-0.1, 0.1)))
        character_state.sadness = max(0.0, min(1.0, character_state.sadness + random.uniform(-0.1, 0.1)))
        character_state.anger = max(0.0, min(1.0, character_state.anger + random.uniform(-0.1, 0.1)))
        character_state.joy = max(0.0, min(1.0, character_state.joy + random.uniform(-0.1, 0.1)))
        character_state.fear = max(0.0, min(1.0, character_state.fear + random.uniform(-0.1, 0.1)))
        character_state.selection_threshold = max(0.0, min(1.0, character_state.selection_threshold + random.uniform(-0.1, 0.1)))
        character_state.resolution_level = max(0.0, min(1.0, character_state.resolution_level + random.uniform(-0.1, 0.1)))
        character_state.goal_directedness = max(0.0, min(1.0, character_state.goal_directedness + random.uniform(-0.1, 0.1)))
        character_state.securing_rate = max(0.0, min(1.0, character_state.securing_rate + random.uniform(-0.1, 0.1)))

    return character_state

def user_input(user_question, character_state, character_name):
    # Check if the message is already in the chat history
    if any(message["role"] == "User" and message["message"] == user_question for message in st.session_state['chat_history'][character_name]):
        return None  # Skip processing if the message is a duplicate

    greetings = ["hi", "hello", "hey", "greetings"]
    if user_question.lower() in greetings:
        response_text = f"Hello! I'm {character_name}. How can I assist you today?"
        st.session_state['chat_history'][character_name].append({"role": "User", "message": user_question})
        st.session_state['chat_history'][character_name].append({"role": character_name, "message": response_text})
        character_state = simulate_emotions(user_question, character_state)
        return response_text

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    character_state = simulate_emotions(user_question, character_state)
    character_state.long_term_memory[user_question] = response["output_text"]

    st.session_state['chat_history'][character_name].append({"role": "User", "message": user_question})
    st.session_state['chat_history'][character_name].append({"role": character_name, "message": response["output_text"]})

    return response["output_text"]


def main():
    st.set_page_config("Chat with Characters")
    st.header("Chat with Characters from Books")

    # Inject custom CSS for styling
    st.markdown("""
        <style>
        .user { 
            text-align: right; 
            color: #2e86de;
            margin: 8px 0;
        }
        .bot { 
            text-align: left; 
            color: #10ac84;
            margin: 8px 0;
        }
        .stTextInput input {
            border-radius: 20px;
            padding: 10px 15px;
        }
        </style>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", 
                                  accept_multiple_files=True)
        if st.button("Submit & Process"):
            if 'processed_text' not in st.session_state:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    characters = extract_characters(raw_text)
                    st.session_state['characters'] = characters
                    st.session_state['processed_text'] = raw_text
                    st.success("Processing complete!")

    if 'characters' in st.session_state:
        character_name = st.selectbox("Choose a Character:", st.session_state['characters'])
        if 'character_states' not in st.session_state:
            st.session_state['character_states'] = {}
        if character_name not in st.session_state['character_states']:
            st.session_state['character_states'][character_name] = CharacterState()
        character_state = st.session_state['character_states'][character_name]

        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = {}
        if character_name not in st.session_state['chat_history']:
            st.session_state['chat_history'][character_name] = []

        # Display emotional state in sidebar
        with st.sidebar:
            st.write(f"### {character_name}'s Emotional State")
            character_state.display_emotions()

        # Display chat history with avatars
        for chat in st.session_state['chat_history'][character_name]:
            if chat["role"] == "User":
                st.markdown(f'''
                <div class="user">
                    <strong>👤 You:</strong> {chat["message"]}
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="bot">
                    <strong>🤖 {character_name}:</strong> {chat["message"]}
                </div>
                ''', unsafe_allow_html=True)

        # Chat input with clear-after-submit functionality
        def clear_text():
            st.session_state.chat_input = ""
            
        if 'chat_input' not in st.session_state:
            st.session_state.chat_input = ""

        user_question = st.text_input(
            f"Ask {character_name} a question:",
            key="chat_input",
            on_change=lambda: [
                process_input(character_state, character_name),
                clear_text()
            ],
            placeholder=f"Type your message to {character_name} here...",
            label_visibility="collapsed"
        )

def process_input(character_state, character_name):
    user_question = st.session_state.chat_input.strip()
    if user_question: 
        with st.spinner(f"{character_name} is thinking..."):
            response_text = user_input(user_question, character_state, character_name)
            if response_text:
                st.session_state.chat_input = ''
                


if __name__ == "__main__":
    main()
