import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
import os
from typing import Union, List, Optional
import re
import streamlit as st
from pdf_processor import PDFProcessor
from chat import ChatManager
from character import CharacterManager
from dotenv import load_dotenv

load_dotenv()

def main():
    st.set_page_config("Chat with Characters")
    st.header("Chat with Characters from Books")

    # Custom CSS styling
    st.markdown("""
        <style>
        .user { text-align: right; color: #2e86de; margin: 8px 0; padding: 8px 12px;
                border-radius: 18px 18px 0 18px; background-color: #f0f8ff;
                display: inline-block; max-width: 80%; float: right; clear: both; }
        .bot { text-align: left; color: #10ac84; margin: 8px 0; padding: 8px 12px;
               border-radius: 18px 18px 18px 0; background-color: #f0fff0;
               display: inline-block; max-width: 80%; float: left; clear: both; }
        .emotion-bar {
            height: 20px;
            margin-bottom: 10px;
            border-radius: 10px;
            background-color: #e0e0e0;
        }
        .emotion-fill {
            height: 100%;
            border-radius: 10px;
            background-color: #4CAF50;
            transition: width 0.5s ease-in-out;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize managers
    pdf_processor = PDFProcessor()
    character_manager = CharacterManager()

    # Initialize session state
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
        st.session_state.temp_messages = {}  # Changed to dictionary
        st.session_state.authenticated = False
        st.session_state.emotion_updates = 0
        st.session_state.book_source = None  # add book_source

    # User ID handling
    if not st.session_state.authenticated:
        with st.sidebar:
            with st.expander("🔐 Identification"):
                user_id = st.text_input("Enter your user ID (or leave blank)", key="user_id_input")
                if st.button("Start Chatting"):
                    if user_id:
                        st.session_state.user_id = user_id
                        st.session_state.authenticated = True
                        st.success("ID saved - conversations will be remembered")
                    else:
                        st.session_state.user_id = "anonymous"
                        st.session_state.authenticated = True
                        st.warning("Temporary session - chat won't be saved")
                    st.rerun()
        st.stop()

    # PDF processing
    with st.sidebar:
        st.title("Menu:")
        input_tab1, input_tab2 = st.tabs(["Upload PDF", "Paste Text"])
        with input_tab1:
            pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
            book_source = st.text_input("Enter Book Source (e.g., Book Title):")  # add book source input.
            if st.button("Submit & Process") and pdf_docs and book_source:
                with st.spinner("Processing..."):
                    raw_text = pdf_processor.get_pdf_text(pdf_docs)
                    text_chunks = pdf_processor.get_text_chunks(raw_text)
                    pdf_processor.create_vector_store(text_chunks)
                    characters = pdf_processor.extract_characters(raw_text)
                    st.session_state['characters'] = characters
                    st.session_state.book_source = book_source  # save book source
                    st.success("Processing complete!")

        with input_tab2:
            history_text = st.text_area("Paste history text here:", height=300, key="history_text")
            book_source_text = st.text_input("Enter Text Source (e.g., Book Title):", key="text_source")  # add book source for text.
            if st.button("Process Text") and history_text and book_source_text:
                with st.spinner("Processing..."):
                    characters = pdf_processor.process_input(history_text)
                    if not characters:
                        st.warning("No identifiable characters found in the text. Please provide a longer narrative content")
                    else:
                        st.session_state['characters'] = characters
                        st.session_state.book_source = book_source_text  # save book source for text.
                        st.success("Text processing complete!")

    # Chat interface
    if 'characters' in st.session_state and st.session_state.book_source:  # check if source is available.
        character_name = st.selectbox("Choose a Character:", st.session_state['characters'])

        # Initialize character-specific messages if not exists
        if character_name not in st.session_state.temp_messages:
            st.session_state.temp_messages[character_name] = []

        # Create container for emotional state
        emotion_container = st.sidebar.container()

        # Display initial emotional state
        character_state, character_id = character_manager.get_character_state(character_name, st.session_state.book_source,st.session_state.user_id)  # Get state and id.

        # Display initial emotional state
        with emotion_container:
            st.write(f"### {character_name}'s Emotional State")
            character_state.display_emotions()

        # Initialize chat_manager with book source
        chat_manager = ChatManager(st.session_state.book_source)

        # Display chat history
        if st.session_state.user_id != "anonymous":
            chat_manager.display_chat_history(character_name, st.session_state.user_id)
        else:
            for msg in st.session_state.temp_messages[character_name]:
                if msg['role'] == 'user':
                    st.markdown(f'<div class="user"><strong>👤 You:</strong> {msg["content"]}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot"><strong>🤖 {character_name}:</strong> {msg["content"]}</div>',
                                unsafe_allow_html=True)

        # Handle chat input
        if prompt := st.chat_input(f"Ask {character_name}..."):
            with st.spinner(f"{character_name} is thinking..."):
                response, updated_state = chat_manager.process_user_input(
                    prompt,
                    character_name,
                    st.session_state.user_id
                )

                if st.session_state.user_id == "anonymous":
                    st.session_state.temp_messages[character_name].append({'role': 'user', 'content': prompt})
                    st.session_state.temp_messages[character_name].append({'role': 'assistant', 'content': response})

                # Display the messages
                st.chat_message("user").write(prompt)
                st.chat_message("assistant").write(response)

                # Update emotional state display
                with emotion_container:
                    st.write(f"### {character_name}'s Emotional State")
                    latest_state, character_id = character_manager.get_character_state(character_name, st.session_state.book_source,st.session_state.user_id)
                    latest_state.display_emotions()

                # Only save character states if user is not anonymous.
                if st.session_state.user_id != "anonymous":
                    character_manager.save_character_state(character_name, latest_state, st.session_state.book_source, st.session_state.user_id) #Use session state user id

                st.rerun()

if __name__ == "__main__":
    main()