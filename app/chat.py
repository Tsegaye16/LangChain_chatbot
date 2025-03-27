from character_state import CharacterState
from database import DatabaseManager
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from character import CharacterManager
import streamlit as st

class ChatManager:
    """
    Core chat management system that handles:
    - Conversational chains with book characters
    - Processing user inputs and generating responses
    - Maintaining conversation history
    - Protecting personally identifiable information (PII)
    """
    
    def __init__(self, book_source):
        """
        Initialize chat manager with required components
        
        Args:
            book_source (str): Identifier for the source material being used
        """
        self.db = DatabaseManager()  # Database operations handler
        self.character_manager = CharacterManager()  # Character state manager
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Text embeddings
        self.name_extraction_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)  # Name extraction LLM
        self.book_source = book_source  # Current book/context identifier

    def get_conversational_chain(self, character_name):
        """
        Creates a configured QA chain for character conversations with:
        - Character persona enforcement
        - Conversation history awareness
        - PII protection safeguards
        
        Args:
            character_name (str): Name of character to roleplay
            
        Returns:
            QA Chain: Configured conversation chain
        """
        prompt_template = f"""
            You are {character_name}, a character from a book. Respond naturally to questions while staying in character.

            When asked about someone (like "Tell me about someone"), summarize what you've learned about them from the [Conversation History]. 
            Include details like their general behavior or any relevant information gleaned from previous interactions, but DO NOT reveal any personally identifiable information (PII) such as specific addresses, phone numbers, email addresses, or ages. 
            If there are no mentions in the [Conversation History], state that you don't have enough information to provide a summary.
            If there are mentions in the [Book Context] and not in the [Conversation History], use the book context to provide general information, excluding PII.

            [Book Context]:
            {{context}}

            [Conversation History]:
            {{history}}

            Current Question:
            {{question}}

            Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "history"]
        )
        return load_qa_chain(
            model,
            chain_type="stuff",
            prompt=prompt,
            document_variable_name="context"
        )

    def process_user_input(self, prompt, character_name, user_id):
        """
        Processes user input through the full conversation pipeline:
        1. Retrieves character state
        2. Checks for name references in history
        3. Generates context-aware response
        4. Updates character emotions
        5. Persists data for logged-in users
        
        Args:
            prompt (str): User's input message
            character_name (str): Character being conversed with
            user_id (str): User identifier ("anonymous" for temporary sessions)
            
        Returns:
            tuple: (response_text, updated_character_state)
        """
        # Retrieve or initialize character state
        character_state, character_id = self.character_manager.get_character_state(
            character_name, 
            self.book_source, 
            user_id
        )

        # Only create new database records for authenticated users
        if character_id is None and user_id != "anonymous":
            character_id = self.db.save_character_state(
                character_name, 
                character_state, 
                self.book_source
            )

        # Check for name references in query
        history_context = ""
        name_query_phrases = ["tell me about", "you know", "describe", "who is"]
        if any(phrase in prompt.lower() for phrase in name_query_phrases):
            name_to_check = self._extract_name_from_question_using_llm(prompt)
            if name_to_check:
                print(f"Searching for mentions of: {name_to_check}")
                # Retrieve relevant conversation history
                all_conversations = self.db.search_conversations_for_mentions(
                    character_id, 
                    name_to_check
                )

                # Filter out meta-queries about knowledge
                relevant_mentions = [
                    (content, role, timestamp) 
                    for content, role, timestamp in all_conversations
                    if (name_to_check.lower() in content.lower() and 
                        "did you know" not in content.lower())
                ]

                if relevant_mentions:
                    history_context = f"Previous mentions of {name_to_check}:\n"
                    for content, role, timestamp in relevant_mentions:
                        history_context += f"- {role} said: '{content}'\n"
                    print(f"History context built: {history_context}")
                else:
                    print("No relevant mentions found.")

        try:
            # Load document embeddings and generate response
            new_db = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(prompt)

            chain = self.get_conversational_chain(character_name)
            response = chain.invoke(
                {
                    "input_documents": docs,
                    "question": prompt,
                    "history": history_context
                },
                return_only_outputs=True
            )
            response_text = response["output_text"]
        except Exception as e:
            response_text = f"I can't process that right now. Error: {str(e)}"

        # Update character's emotional state
        updated_state = self.character_manager.simulate_emotions(
            prompt, 
            character_name, 
            character_state, 
            self.book_source, 
            user_id
        )
        
        # Persist data only for authenticated users
        if user_id != "anonymous":
            self.character_manager.save_character_state(
                character_name, 
                updated_state, 
                self.book_source, 
                user_id
            )
            conversation_id = self.db.create_conversation(character_id, user_id)
            self.db.save_message(conversation_id, "user", prompt)
            self.db.save_message(conversation_id, "assistant", response_text)

        return response_text, updated_state

    def _extract_name_from_question_using_llm(self, question):
        """
        Helper method to extract names from user questions using LLM
        
        Args:
            question (str): User's input containing potential name reference
            
        Returns:
            str: Extracted name or None if not found
        """
        prompt = f"""
            Extract the full name of the person mentioned in the following question. 
            If no name is mentioned, return 'None'.

            Question: {question}

            Name:
        """
        response = self.name_extraction_model.invoke(prompt)

        name = response.content.strip().strip(" .")
        return name if name != 'None' else None

    def display_chat_history(self, character_name, user_id):
        """
        Renders conversation history in Streamlit UI
        Skips rendering for anonymous users
        
        Args:
            character_name (str): Character being conversed with
            user_id (str): User identifier
        """
        if user_id == "anonymous":
            return

        character_state, character_id = self.character_manager.db.get_character_state(
            character_name, 
            self.book_source, 
            user_id
        )
        history = self.character_manager.db.get_conversation_history(character_id, user_id)

        # Format and display each message
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