from character_state import CharacterState
from database import DatabaseManager
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from character import CharacterManager
import streamlit as st

class ChatManager:
    def __init__(self):
        self.db = DatabaseManager()  # Initialize DatabaseManager
        self.character_manager = CharacterManager()  # Initialize CharacterManager
        self.character_manager = CharacterManager()
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
    def get_conversational_chain(self, character_name):
        prompt_template = f"""
        You are {character_name}, a character from a book. Respond naturally to the question while staying in character.
        Be conversational and maintain your personality traits.

        Context:\n {{context}}?\n
        Question: \n{{question}}\n
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    def process_user_input(self, prompt, character_name, user_id):
        """Process input and return (response, updated_character_state)"""
        # Get current character state
        character_state = self.character_manager.get_character_state(character_name)
        
        # Generate response
        try:
            new_db = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(prompt)
            
            chain = self.get_conversational_chain(character_name)
            response = chain(
                {"input_documents": docs, "question": prompt},
                return_only_outputs=True
            )
            response_text = response["output_text"]
        except Exception as e:
            response_text = "I can't process that right now."
        
        # Update emotional state
        updated_state = self.character_manager.simulate_emotions(
            prompt, character_name, character_state
        )
        
        # Save the updated character state to database
        self.character_manager.save_character_state(character_name, updated_state)
        
        # Save conversation if authenticated
        if user_id != "anonymous":
            character_id = self.db.save_character_state(character_name, updated_state)
            conversation_id = self.db.create_conversation(character_id, user_id)
            self.db.save_message(conversation_id, "user", prompt)
            self.db.save_message(conversation_id, "assistant", response_text)
        
        return response_text, updated_state

    def _save_conversation(self, character_name, user_id, question, response):
        """Save conversation to database for authenticated users"""
        character_id = self.character_manager.db.save_character_state(character_name, CharacterState())
        conversation_id = self.character_manager.db.create_conversation(character_id, user_id)
        self.character_manager.db.save_message(conversation_id, "user", question)
        self.character_manager.db.save_message(conversation_id, "assistant", response)
    
    def display_chat_history(self, character_name, user_id):
        """Display chat history with proper message roles and timestamps"""
        if user_id == "anonymous":
            return
            
        character_id = self.character_manager.db.save_character_state(character_name, CharacterState())
        history = self.character_manager.db.get_conversation_history(character_id, user_id)
        
        for message in history:
            timestamp = message.get("timestamp", "").strftime("%H:%M") if message.get("timestamp") else ""
            if message["role"].lower() == "user":
                st.markdown(f'''
                    <div class="user">
                        <strong>👤 You:</strong> {message["content"]}
                        <div class="timestamp">{timestamp}</div>
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                    <div class="bot">
                        <strong>🤖 {character_name}:</strong> {message["content"]}
                        <div class="timestamp">{timestamp}</div>
                    </div>
                ''', unsafe_allow_html=True)  
                