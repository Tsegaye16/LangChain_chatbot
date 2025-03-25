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
        You are {character_name}, a character from a book. Respond naturally to questions while staying in character.
        
        When asked if you know someone (like "Do you know someone"), check the [Conversation History] section below.
        Only reference information that actually appears there - never make up information.
        
        If there are no mentions, say you don't know them well.
        If there are mentions, summarize what you've learned from previous conversations.
        
        Conversation History:
        {{context}}
        
        Current Question: 
        {{question}}
        
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]  # Must include "context" for QA chain
        )
        
        # Create the chain with explicit document variable name
        return load_qa_chain(
            model, 
            chain_type="stuff",
            prompt=prompt,
            document_variable_name="context"  # This tells the chain where to put the docs
        )

    def process_user_input(self, prompt, character_name, user_id):
        """Process input and return (response, updated_character_state)"""
        # Get current character state
        character_state = self.character_manager.get_character_state(character_name)
        character_id = self.db.save_character_state(character_name, character_state)
        
        # Check if asking about a person
        history_context = ""
        if "you know" in prompt.lower() or "you remember" in prompt.lower():
            # Extract name being asked about
            name_to_check = self._extract_name_from_question(prompt)
            if name_to_check:
                # Search conversation history for mentions
                mentions = self.db.search_conversations_for_mentions(character_id, name_to_check)
                
                if mentions:
                    history_context = f"Previous mentions of {name_to_check}:\n"
                    for content, role, timestamp in mentions:
                        history_context += f"- {role} said: '{content}'\n"
        
        # Generate response
        try:
            new_db = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(prompt)
            
            # Combine document context with history context
            combined_context = ""
            if docs:
                combined_context += "Book Context:\n" + "\n".join([doc.page_content for doc in docs]) + "\n\n"
            if history_context:
                print("History: ",history_context)
                combined_context += "Conversation History:\n" + history_context
            
            chain = self.get_conversational_chain(character_name)
            response = chain(
                {
                    "input_documents": docs,  # This will go into the "context" variable
                    "question": prompt,       # This goes into the "question" variable
                },
                return_only_outputs=True
            )
            response_text = response["output_text"]
        except Exception as e:
            response_text = f"I can't process that right now. Error: {str(e)}"
        
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

    def _extract_name_from_question(self, question):
        """Simple name extraction from questions like 'Do you know Tsegaye?'"""
        question_lower = question.lower()
        markers = ["you know", "you remember", "about"]
        for marker in markers:
            if marker in question_lower:
                start_idx = question_lower.index(marker) + len(marker)
                name_part = question[start_idx:].strip(" ?.,")
                # Simple cleanup - in production you'd want better NLP here
                if name_part:
                    return name_part.split()[0]  # Take first word as name
        return None
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
                