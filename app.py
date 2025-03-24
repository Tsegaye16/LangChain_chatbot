import streamlit as st
from pdf_processor import PDFProcessor
from chat import ChatManager
from character import CharacterManager
import os
from dotenv import load_dotenv

load_dotenv()

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
            padding: 8px 12px;
            border-radius: 18px 18px 0 18px;
            background-color: #f0f8ff;
            display: inline-block;
            max-width: 80%;
            float: right;
            clear: both;
        }
        .bot { 
            text-align: left; 
            color: #10ac84;
            margin: 8px 0;
            padding: 8px 12px;
            border-radius: 18px 18px 18px 0;
            background-color: #f0fff0;
            display: inline-block;
            max-width: 80%;
            float: left;
            clear: both;
        }
        .timestamp {
            font-size: 0.8em;
            color: #666;
            margin-top: 4px;
        }
        .stTextInput input {
            border-radius: 20px;
            padding: 10px 15px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Initialize managers
    pdf_processor = PDFProcessor()
    chat_manager = ChatManager()
    character_manager = CharacterManager()

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", 
            accept_multiple_files=True
        )
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_processor.get_pdf_text(pdf_docs)
                text_chunks = pdf_processor.get_text_chunks(raw_text)
                pdf_processor.create_vector_store(text_chunks)
                characters = pdf_processor.extract_characters(raw_text)
                st.session_state['characters'] = characters
                st.session_state['processed_text'] = raw_text
                st.success("Processing complete!")
        
        # User ID input
        user_id = st.text_input("Enter your user ID (for conversation history)", "default_user")
        st.session_state['user_id'] = user_id

    if 'characters' in st.session_state:
        character_name = st.selectbox("Choose a Character:", st.session_state['characters'])
        
        # Display character's emotional state in sidebar
        with st.sidebar:
            character_state = character_manager.get_character_state(character_name)
            st.write(f"### {character_name}'s Emotional State")
            character_state.display_emotions()

        # Display chat history
        chat_manager.display_chat_history(character_name, st.session_state['user_id'])

        # Chat input
        def clear_text():
            st.session_state.chat_input = ""
            
        if 'chat_input' not in st.session_state:
            st.session_state.chat_input = ""

        user_question = st.text_input(
            f"Ask {character_name} a question:",
            key="chat_input",
            on_change=lambda: [
                process_input(character_name),
                clear_text()
            ],
            placeholder=f"Type your message to {character_name} here...",
            label_visibility="collapsed"
        )

def process_input(character_name):
    user_question = st.session_state.chat_input.strip()
    if user_question:
        with st.spinner(f"{character_name} is thinking..."):
            chat_manager = ChatManager()
            response = chat_manager.process_user_input(
                user_question, 
                character_name,
                st.session_state.get('user_id', 'default_user')
            )
            if response:
                st.session_state.chat_input = ''

if __name__ == "__main__":
    main()