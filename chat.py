from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from database import DatabaseManager
from character import CharacterManager
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

class ChatManager:
    def __init__(self):
        self.character_manager = CharacterManager()
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
    def get_conversational_chain(self):
        """Create conversational chain for question answering"""
        prompt_template = """
        You are {character_name}, a character from a book, engaging in a conversation. 
        Answer the question as detailed as possible from the provided context.
        If the answer is not in the provided context, respond in a way that acknowledges 
        the question and suggests a lack of information.
        Be conversational and contextual in your responses.
        Maintain your character's personality and knowledge throughout the conversation.

        Context:\n {context}?\n
        Question: \n{question}\n
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "character_name"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    def process_user_input(self, user_question, character_name, user_id="default_user"):
        """Process user input and generate response"""
        # Check if this is a greeting
        greetings = ["hi", "hello", "hey", "greetings"]
        if user_question.lower() in greetings:
            response_text = f"Hello! I'm {character_name}. How can I assist you today?"
            self._save_message(character_name, user_id, "User", user_question)
            self._save_message(character_name, user_id, character_name, response_text)
            return response_text
        
        # Get character state and update emotions
        character_state = self.character_manager.get_character_state(character_name)
        character_state = self.character_manager.simulate_emotions(user_question, character_name, character_state)
        
        # Search for similar documents
        try:
            new_db = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            
            # Get conversational chain
            chain = self.get_conversational_chain()
            response = chain(
                {
                    "input_documents": docs, 
                    "question": user_question,
                    "character_name": character_name
                }, 
                return_only_outputs=True
            )
            
            response_text = response["output_text"]
            
            # Save to memory and conversation history
            self.character_manager.save_to_memory(character_name, user_question, response_text)
            self._save_message(character_name, user_id, "User", user_question)
            self._save_message(character_name, user_id, character_name, response_text)
            
            return response_text
        except Exception as e:
            st.error(f"Error processing your question: {e}")
            return "I'm sorry, I encountered an error processing your question."
    
    def _save_message(self, character_name, user_id, role, content):
        """Save message to conversation history"""
        self.character_manager.save_conversation(
            character_name, 
            user_id, 
            [{"role": role, "content": content}]
        )
    
    def get_chat_history(self, character_name, user_id=None, limit=20):
        """Get chat history for display"""
        return self.character_manager.get_conversation_history(character_name, user_id, limit)
    
    def display_chat_history(self, character_name, user_id=None):
        """Display chat history in Streamlit in correct order"""
        history = self.get_chat_history(character_name, user_id)
        
        # Create a container for the chat history
        chat_container = st.container()
        
        with chat_container:
            for message in history:
                if message["role"] == "User":
                    st.markdown(f'''
                    <div class="user">
                        <strong>👤 You:</strong> {message["content"]}
                        <div style="font-size: 0.8em; color: #666;">{message["timestamp"].strftime('%H:%M:%S')}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="bot">
                        <strong>🤖 {character_name}:</strong> {message["content"]}
                        <div style="font-size: 0.8em; color: #666;">{message["timestamp"].strftime('%H:%M:%S')}</div>
                    </div>
                    ''', unsafe_allow_html=True)
        
        # Add some spacing after the chat
        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
