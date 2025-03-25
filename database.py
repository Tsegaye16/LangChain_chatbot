import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import os
from dotenv import load_dotenv
import streamlit as st
from character_state import CharacterState

load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.connect()
        self.initialize_database()

    def connect(self):
        """Establish connection to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                dbname=os.getenv("DB_NAME", "chatbot_db"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", "postgres"),
                host=os.getenv("DB_HOST", "localhost"),
                port=os.getenv("DB_PORT", "5432")
            )
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            raise

    def initialize_database(self):
        """Create tables if they don't exist"""
        try:
            with self.conn.cursor() as cur:
                # Create characters table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS characters (
                        character_id SERIAL PRIMARY KEY,
                        name TEXT UNIQUE NOT NULL,
                        arousal FLOAT DEFAULT 0.5,
                        valence FLOAT DEFAULT 0.5,
                        dominance FLOAT DEFAULT 0.5,
                        sadness FLOAT DEFAULT 0.0,
                        anger FLOAT DEFAULT 0.0,
                        joy FLOAT DEFAULT 0.0,
                        fear FLOAT DEFAULT 0.0,
                        selection_threshold FLOAT DEFAULT 0.5,
                        resolution_level FLOAT DEFAULT 0.5,
                        goal_directedness FLOAT DEFAULT 0.5,
                        securing_rate FLOAT DEFAULT 0.5,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create conversations table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        conversation_id SERIAL PRIMARY KEY,
                        character_id INTEGER REFERENCES characters(character_id),
                        user_id TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create messages table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        message_id SERIAL PRIMARY KEY,
                        conversation_id INTEGER REFERENCES conversations(conversation_id),
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create long_term_memory table
                cur.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    memory_id SERIAL PRIMARY KEY,
                    character_id INTEGER REFERENCES characters(character_id),
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (character_id, key)  -- Add unique constraint
                        )
                    """)
                
            
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            st.error(f"Database initialization failed: {e}")
            raise

    def save_character_state(self, character_name, state):
        """Save or update character state in database"""
        try:
            with self.conn.cursor() as cur:
                # Insert or update character
                cur.execute("""
                    INSERT INTO characters (name, arousal, valence, dominance, sadness, anger, joy, fear, 
                                          selection_threshold, resolution_level, goal_directedness, securing_rate)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (name) DO UPDATE SET
                        arousal = EXCLUDED.arousal,
                        valence = EXCLUDED.valence,
                        dominance = EXCLUDED.dominance,
                        sadness = EXCLUDED.sadness,
                        anger = EXCLUDED.anger,
                        joy = EXCLUDED.joy,
                        fear = EXCLUDED.fear,
                        selection_threshold = EXCLUDED.selection_threshold,
                        resolution_level = EXCLUDED.resolution_level,
                        goal_directedness = EXCLUDED.goal_directedness,
                        securing_rate = EXCLUDED.securing_rate
                    RETURNING character_id
                """, (
                    character_name, state.arousal, state.valence, state.dominance, 
                    state.sadness, state.anger, state.joy, state.fear,
                    state.selection_threshold, state.resolution_level, 
                    state.goal_directedness, state.securing_rate
                ))
                character_id = cur.fetchone()[0]
                self.conn.commit()
                return character_id
        except Exception as e:
            self.conn.rollback()
            st.error(f"Failed to save character state: {e}")
            raise

    def get_character_state(self, character_name):
        """Retrieve character state from database"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT arousal, valence, dominance, sadness, anger, joy, fear,
                           selection_threshold, resolution_level, goal_directedness, securing_rate
                    FROM characters
                    WHERE name = %s
                """, (character_name,))
                result = cur.fetchone()
                if result:
                    return CharacterState(
                        arousal=result[0], valence=result[1], dominance=result[2],
                        sadness=result[3], anger=result[4], joy=result[5], fear=result[6],
                        selection_threshold=result[7], resolution_level=result[8],
                        goal_directedness=result[9], securing_rate=result[10]
                    )
                return None
        except Exception as e:
            st.error(f"Failed to get character state: {e}")
            raise

    def create_conversation(self, character_id, user_id):
        """Return 'anonymous' for invalid user IDs"""
        if not user_id or user_id == "anonymous":
            return "anonymous"
            
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO conversations (character_id, user_id)
                    VALUES (%s, %s)
                    RETURNING conversation_id
                """, (character_id, user_id))
                conversation_id = cur.fetchone()[0]
                self.conn.commit()
                return conversation_id
        except Exception as e:
            self.conn.rollback()
            st.error(f"Failed to create conversation: {e}")
            raise
    def save_message(self, conversation_id, role, content):
        """Save message with proper role identification"""
        if not conversation_id or conversation_id == "anonymous":
            return
            
        try:
            with self.conn.cursor() as cur:
                # Convert role to lowercase for consistency
                normalized_role = role.lower()
                cur.execute("""
                    INSERT INTO messages (conversation_id, role, content)
                    VALUES (%s, %s, %s)
                """, (conversation_id, normalized_role, content))
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            st.error(f"Failed to save message: {e}")
            raise

    def get_conversation_history(self, character_id, user_id=None, limit=20):
        """Retrieve conversation history for a character in chronological order"""
        try:
            with self.conn.cursor() as cur:
                if user_id:
                    # Get specific user's conversation with this character
                    cur.execute("""
                        SELECT m.role, m.content, m.timestamp
                        FROM messages m
                        JOIN conversations c ON m.conversation_id = c.conversation_id
                        WHERE c.character_id = %s AND c.user_id = %s
                        ORDER BY m.timestamp ASC  -- Changed to ASC for chronological order
                        LIMIT %s
                    """, (character_id, user_id, limit))
                else:
                    # Get all conversations with this character
                    cur.execute("""
                        SELECT m.role, m.content, m.timestamp
                        FROM messages m
                        JOIN conversations c ON m.conversation_id = c.conversation_id
                        WHERE c.character_id = %s
                        ORDER BY m.timestamp ASC  -- Changed to ASC for chronological order
                        LIMIT %s
                    """, (character_id, limit))
                
                return [{"role": row[0], "content": row[1], "timestamp": row[2]} for row in cur.fetchall()]
        except Exception as e:
            st.error(f"Failed to get conversation history: {e}")
            raise

    def save_to_memory(self, character_id, key, value):
        """Save information to character's long-term memory"""
        try:
            with self.conn.cursor() as cur:
                # First delete any existing entry for this key
                cur.execute("""
                    DELETE FROM long_term_memory 
                    WHERE character_id = %s AND key = %s
                """, (character_id, key))
                
                # Then insert the new value
                cur.execute("""
                    INSERT INTO long_term_memory (character_id, key, value)
                    VALUES (%s, %s, %s)
                """, (character_id, key, value))
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            st.error(f"Failed to save to memory: {e}")
            raise

    def search_conversations_for_mentions(self, character_id, search_term):
        """Search all messages for mentions of a specific term"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT m.content, m.role, m.timestamp
                    FROM messages m
                    JOIN conversations c ON m.conversation_id = c.conversation_id
                    WHERE c.character_id = %s 
                    AND m.content ILIKE %s
                    ORDER BY m.timestamp DESC
                    LIMIT 5
                """, (character_id, f'%{search_term}%'))
                return cur.fetchall()
        except Exception as e:
            st.error(f"Failed to search conversations: {e}")
            return []

    def get_from_memory(self, character_id, key):
        """Retrieve information from character's long-term memory"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT value FROM long_term_memory
                    WHERE character_id = %s AND key = %s
                """, (character_id, key))
                result = cur.fetchone()
                return result[0] if result else None
        except Exception as e:
            st.error(f"Failed to get from memory: {e}")
            raise
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()