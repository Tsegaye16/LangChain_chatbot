from database import DatabaseManager
from character_state import CharacterState
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import random
import json

class CharacterManager:
    def __init__(self):
        self.db = DatabaseManager()
        
    def get_character_state(self, character_name, book_source, user_id):
        """Get character state from database or create new one"""
        state, character_id = self.db.get_character_state(character_name, book_source,user_id)
        if state is None:
            state = CharacterState()
            character_id = self.save_character_state(character_name, state, book_source,user_id)
        return state, character_id
    
    def save_character_state(self, character_name, state, book_source, user_id):
        """Save character state to database"""
        return self.db.save_character_state(character_name, state, book_source, user_id)
        
    def simulate_emotions(self, user_input, character_name, character_state, book_source, user_id):
        """Simulate emotional response to user input using LLM for emotion mapping."""
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
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
            response = model.invoke(prompt)
            print(f"LLM Response: {response.content}")

            json_string = response.content.strip()
            if json_string.startswith("```json"):
                json_string = json_string[7:]
            if json_string.endswith("```"):
                json_string = json_string[:-3]

            if json_string and json_string.strip():
                emotion_data = json.loads(json_string)
                character_state.update_emotions(emotion_data)
                self.save_character_state(character_name, character_state, book_source, user_id)
            else:
                raise ValueError("LLM returned an empty or invalid response.")

        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            print(f"Error processing LLM response: {e}")
            character_state.arousal = max(0.0, min(1.0, character_state.arousal + random.uniform(-0.1, 0.1)))
            character_state.valence = max(0.0, min(1.0, character_state.valence + random.uniform(-0.1, 0.1)))
            self.save_character_state(character_name, character_state, book_source, user_id)

        return character_state

    def get_conversation_history(self, character_name, book_source, user_id=None, limit=20):
        """Get conversation history for a character"""
        state, character_id = self.get_character_state(character_name, book_source, user_id)
        return self.db.get_conversation_history(character_id, user_id, limit)
    
    def save_conversation(self, character_name, book_source, user_id, messages):
        """Save a conversation to the database"""
        state, character_id = self.get_character_state(character_name, book_source, user_id)
        conversation_id = self.db.create_conversation(character_id, user_id)
        for message in messages:
            self.db.save_message(conversation_id, message["role"], message["content"])
    
    def save_to_memory(self, character_name, book_source, key, value, user_id):
        """Save information to character's long-term memory"""
        state, character_id = self.get_character_state(character_name, book_source, user_id)
        self.db.save_to_memory(character_id, key, value)
    
    def get_from_memory(self, character_name, book_source, key, user_id):
        """Retrieve information from character's long-term memory"""
        state, character_id = self.get_character_state(character_name, book_source, user_id)
        return self.db.get_from_memory(character_id, key)