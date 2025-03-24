from database import DatabaseManager
from character_state import CharacterState
from langchain_google_genai import ChatGoogleGenerativeAI  # Add this import
import streamlit as st
import random
import json

class CharacterManager:
    def __init__(self):
        self.db = DatabaseManager()
        
    def get_character_state(self, character_name):
        """Get character state from database or create new one"""
        state = self.db.get_character_state(character_name)
        if state is None:
            state = CharacterState()
            self.save_character_state(character_name, state)
        return state
    
    def save_character_state(self, character_name, state):
        """Save character state to database"""
        self.db.save_character_state(character_name, state)
        
    def simulate_emotions(self, user_input, character_name, character_state):
        """Simulate emotional response to user input"""
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
        
        Generate new values for these parameters in JSON format.
        Return ONLY the JSON object with these keys:
        arousal, valence, dominance, sadness, anger, joy, fear,
        selection_threshold, resolution_level, goal_directedness, securing_rate
        """
        
        try:
            response = model.invoke(prompt)
            emotion_data = json.loads(response.content)
            character_state.update_emotions(emotion_data)
            self.save_character_state(character_name, character_state)
        except (json.JSONDecodeError, AttributeError):
            # Fallback to random adjustment if LLM response is invalid
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
            self.save_character_state(character_name, character_state)
            
        return character_state
    
    def get_conversation_history(self, character_name, user_id=None, limit=20):
        """Get conversation history for a character"""
        character_id = self.db.save_character_state(character_name, CharacterState())  # Ensures character exists
        return self.db.get_conversation_history(character_id, user_id, limit)
    
    def save_conversation(self, character_name, user_id, messages):
        """Save a conversation to the database"""
        character_id = self.db.save_character_state(character_name, CharacterState())
        conversation_id = self.db.create_conversation(character_id, user_id)
        for message in messages:
            self.db.save_message(conversation_id, message["role"], message["content"])
    
    def save_to_memory(self, character_name, key, value):
        """Save information to character's long-term memory"""
        character_id = self.db.save_character_state(character_name, CharacterState())
        self.db.save_to_memory(character_id, key, value)
    
    def get_from_memory(self, character_name, key):
        """Retrieve information from character's long-term memory"""
        character_id = self.db.save_character_state(character_name, CharacterState())
        return self.db.get_from_memory(character_id, key)