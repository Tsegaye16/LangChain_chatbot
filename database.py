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
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS characters (
                        character_id SERIAL PRIMARY KEY,
                        name TEXT NOT NULL,
                        source TEXT NOT NULL,
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
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (name, source)
                    )
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        conversation_id SERIAL PRIMARY KEY,
                        character_id INTEGER REFERENCES characters(character_id),
                        user_id TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        message_id SERIAL PRIMARY KEY,
                        conversation_id INTEGER REFERENCES conversations(conversation_id),
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS long_term_memory (
                        memory_id SERIAL PRIMARY KEY,
                        character_id INTEGER REFERENCES characters(character_id),
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (character_id, key)
                    )
                """)

                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            st.error(f"Database initialization failed: {e}")
            raise

    def save_character_state(self, character_name, character_state, book_source, user_id):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT character_id FROM characters WHERE name = %s AND source = %s
                """, (character_name, book_source)) # removed user_id from where clause
                character_id_result = cur.fetchone()

                if character_id_result:
                    character_id = character_id_result[0]
                    cur.execute("""
                        UPDATE characters SET arousal = %s, valence = %s, dominance = %s, sadness = %s, anger = %s, joy = %s, fear = %s, selection_threshold = %s, resolution_level = %s, goal_directedness = %s, securing_rate = %s WHERE character_id = %s
                    """, (character_state.arousal, character_state.valence, character_state.dominance,
                          character_state.sadness, character_state.anger, character_state.joy, character_state.fear,
                          character_state.selection_threshold, character_state.resolution_level,
                          character_state.goal_directedness, character_state.securing_rate, character_id))
                else:
                    cur.execute("""
                        INSERT INTO characters (name, source) VALUES (%s, %s) RETURNING character_id
                    """, (character_name, book_source)) # removed user_id from insert
                    character_id = cur.fetchone()[0]
                    cur.execute("""
                        UPDATE characters SET arousal = %s, valence = %s, dominance = %s, sadness = %s, anger = %s, joy = %s, fear = %s, selection_threshold = %s, resolution_level = %s, goal_directedness = %s, securing_rate = %s WHERE character_id = %s
                    """, (character_state.arousal, character_state.valence, character_state.dominance,
                          character_state.sadness, character_state.anger, character_state.joy, character_state.fear,
                          character_state.selection_threshold, character_state.resolution_level,
                          character_state.goal_directedness, character_state.securing_rate, character_id))

                self.conn.commit()
                return character_id
        except Exception as e:
            self.conn.rollback()
            st.error(f"Failed to save character state: {e}")
            raise

    def get_character_state(self, character_name, book_source, user_id):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT character_id, arousal, valence, dominance, sadness, anger, joy, fear, selection_threshold, resolution_level, goal_directedness, securing_rate FROM characters WHERE name = %s AND source = %s
                """, (character_name, book_source)) # removed user_id from where clause
                result = cur.fetchone()

                if result:
                    character_state = CharacterState(
                        arousal=result[1], valence=result[2], dominance=result[3],
                        sadness=result[4], anger=result[5], joy=result[6], fear=result[7],
                        selection_threshold=result[8], resolution_level=result[9],
                        goal_directedness=result[10], securing_rate=result[11]
                    )
                    return character_state, result[0]
                else:
                    return None, None
        except Exception as e:
            st.error(f"Failed to get character state: {e}")
            raise

    def create_conversation(self, character_id, user_id):
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
        if not conversation_id or conversation_id == "anonymous":
            return

        try:
            with self.conn.cursor() as cur:
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
        try:
            with self.conn.cursor() as cur:
                if user_id:
                    cur.execute("""
                        SELECT m.role, m.content, m.timestamp
                        FROM messages m
                        JOIN conversations c ON m.conversation_id = c.conversation_id
                        WHERE c.character_id = %s AND c.user_id = %s
                        ORDER BY m.timestamp ASC
                        LIMIT %s
                    """, (character_id, user_id, limit))
                else:
                    cur.execute("""
                        SELECT m.role, m.content, m.timestamp
                        FROM messages m
                        JOIN conversations c ON m.conversation_id = c.conversation_id
                        WHERE c.character_id = %s
                        ORDER BY m.timestamp ASC
                        LIMIT %s
                    """, (character_id, limit))

                return [{"role": row[0], "content": row[1], "timestamp": row[2]} for row in cur.fetchall()]
        except Exception as e:
            st.error(f"Failed to get conversation history: {e}")
            raise

    def save_to_memory(self, character_id, key, value):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM long_term_memory
                    WHERE character_id = %s AND key = %s
                """, (character_id, key))

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
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT m.content, m.role, m.timestamp
                    FROM messages m
                    JOIN conversations c ON m.conversation_id = c.conversation_id
                    WHERE c.character_id = %s AND m.content ILIKE %s
                    ORDER BY m.timestamp DESC
                """, (character_id, f"%{search_term}%"))
                return cur.fetchall()
        except Exception as e:
            st.error(f"Failed to search conversations: {e}")
            return []

    def get_from_memory(self, character_id, key):
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
        if self.conn:
            self.conn.close()