from database import DatabaseManager
from character_state import CharacterState
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import random
import json

class CharacterManager:
    """
    Manages character states, emotions, and conversations by:
    - Persisting character states to database
    - Simulating emotional responses using LLMs
    - Handling conversation history and memory
    - Serving as interface between application and character data
    """
    
    def __init__(self):
        """Initialize with database connection"""
        self.db = DatabaseManager()  # Handles all database operations
        
    def get_character_state(self, character_name, book_source, user_id):
        """
        Retrieves or creates a character's emotional state
        
        Args:
            character_name (str): Name of the character
            book_source (str): Source material identifier
            user_id (str): User identifier for personalization
            
        Returns:
            tuple: (CharacterState object, character_id)
        """
        # Try to load existing state
        state, character_id = self.db.get_character_state(character_name, book_source, user_id)
        
        # Create new state if not found
        if state is None:
            state = CharacterState()  # Default neutral state
            character_id = self.save_character_state(character_name, state, book_source, user_id)
            
        return state, character_id
    
    def save_character_state(self, character_name, state, book_source, user_id):
        """
        Persists character state to database
        
        Args:
            character_name (str): Name of character
            state (CharacterState): Current emotional state
            book_source (str): Source material identifier
            user_id (str): User identifier
            
        Returns:
            str: Database ID of saved character
        """
        return self.db.save_character_state(character_name, state, book_source, user_id)
        
    def simulate_emotions(self, user_input, character_name, character_state, book_source, user_id):
        """
        Simulates emotional response to user input using LLM analysis
        
        Process:
        1. Sends current state + user input to LLM
        2. Parses JSON response with new emotional values
        3. Updates character state
        4. Falls back to random fluctuations if LLM fails
        
        Args:
            user_input (str): User's message content
            character_name (str): Character identifier
            character_state (CharacterState): Current emotional state
            book_source (str): Source material identifier
            user_id (str): User identifier
            
        Returns:
            CharacterState: Updated emotional state
        """
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        
        # Construct detailed prompt for emotion analysis
        prompt = f"""
        Given the user input: '{user_input}', and the character's current state:
        Arousal: {character_state.arousal}, 
        Valence: {character_state.valence}, 
        Dominance: {character_state.dominance}, 
        Sadness: {character_state.sadness}, 
        Anger: {character_state.anger}, 
        Joy: {character_state.joy}, 
        Fear: {character_state.fear}, 
        Selection Threshold: {character_state.selection_threshold}, 
        Resolution Level: {character_state.resolution_level}, 
        Goal Directedness: {character_state.goal_directedness}, 
        Securing Rate: {character_state.securing_rate}

        Analyze how the user input might affect the character's emotions and cognitive parameters.
        Consider factors like the sentiment of the input, the character's personality, and the context of the conversation.

        Generate new values for these parameters in JSON format, reflecting the character's emotional response.
        Return ONLY the JSON object with these keys, and no other text.

        Example JSON output:
        {{
        "arousal": 0.6,
        "valence": 0.7,
        "dominance": 0.5,
        "sadness": 0.1,
        "anger": 0.0,
        "joy": 0.8,
        "fear": 0.2,
        "selection_threshold": 0.5,
        "resolution_level": 0.6,
        "goal_directedness": 0.7,
        "securing_rate": 0.5
        }}
        """

        try:
            # Get LLM response
            response = model.invoke(prompt)
            print(f"LLM Response: {response.content}")

            # Clean JSON string
            json_string = response.content.strip()
            json_string = json_string.removeprefix("```json").removesuffix("```").strip()

            if json_string:
                # Parse and update emotions
                emotion_data = json.loads(json_string)
                character_state.update_emotions(emotion_data)
                self.save_character_state(character_name, character_state, book_source, user_id)
            else:
                raise ValueError("Empty LLM response")

        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            print(f"Error processing LLM response: {e}")
            # Fallback: Small random fluctuations
            character_state.arousal = max(0.0, min(1.0, character_state.arousal + random.uniform(-0.1, 0.1)))
            character_state.valence = max(0.0, min(1.0, character_state.valence + random.uniform(-0.1, 0.1)))
            self.save_character_state(character_name, character_state, book_source, user_id)

        return character_state

    def get_conversation_history(self, character_name, book_source, user_id=None, limit=20):
        """
        Retrieves conversation history between user and character
        
        Args:
            character_name (str): Character identifier
            book_source (str): Source material identifier
            user_id (str): Optional user filter
            limit (int): Maximum messages to return
            
        Returns:
            list: Conversation messages (role, content, timestamp)
        """
        state, character_id = self.get_character_state(character_name, book_source, user_id)
        return self.db.get_conversation_history(character_id, user_id, limit)
    
    def save_conversation(self, character_name, book_source, user_id, messages):
        """
        Saves complete conversation to database
        
        Args:
            character_name (str): Character identifier
            book_source (str): Source material identifier
            user_id (str): User identifier
            messages (list): List of message dicts (role, content)
        """
        state, character_id = self.get_character_state(character_name, book_source, user_id)
        conversation_id = self.db.create_conversation(character_id, user_id)
        for message in messages:
            self.db.save_message(conversation_id, message["role"], message["content"])
    
    def save_to_memory(self, character_name, book_source, key, value, user_id):
        """
        Stores information in character's persistent memory
        
        Args:
            character_name (str): Character identifier
            book_source (str): Source material identifier
            key (str): Memory key/identifier
            value (str): Information to store
            user_id (str): User identifier
        """
        state, character_id = self.get_character_state(character_name, book_source, user_id)
        self.db.save_to_memory(character_id, key, value)
    
    def get_from_memory(self, character_name, book_source, key, user_id):
        """
        Retrieves information from character's memory
        
        Args:
            character_name (str): Character identifier
            book_source (str): Source material identifier
            key (str): Memory key to retrieve
            user_id (str): User identifier
            
        Returns:
            str: Retrieved memory content or None
        """
        state, character_id = self.get_character_state(character_name, book_source, user_id)
        return self.db.get_from_memory(character_id, key)