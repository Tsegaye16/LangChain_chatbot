import streamlit as st

class CharacterState:
    def __init__(self, arousal=0.5, valence=0.5, dominance=0.5, sadness=0.0, anger=0.0, joy=0.0, fear=0.0,
                 selection_threshold=0.5, resolution_level=0.5, goal_directedness=0.5, securing_rate=0.5):
        self.arousal = arousal
        self.valence = valence
        self.dominance = dominance
        self.sadness = sadness
        self.anger = anger
        self.joy = joy
        self.fear = fear
        self.selection_threshold = selection_threshold
        self.resolution_level = resolution_level
        self.goal_directedness = goal_directedness
        self.securing_rate = securing_rate

    def update_emotions(self, emotion_data):
        self.arousal = emotion_data.get("arousal", self.arousal)
        self.valence = emotion_data.get("valence", self.valence)
        self.dominance = emotion_data.get("dominance", self.dominance)
        self.sadness = emotion_data.get("sadness", self.sadness)
        self.anger = emotion_data.get("anger", self.anger)
        self.joy = emotion_data.get("joy", self.joy)
        self.fear = emotion_data.get("fear", self.fear)
        self.selection_threshold = emotion_data.get("selection_threshold", self.selection_threshold)
        self.resolution_level = emotion_data.get("resolution_level", self.resolution_level)
        self.goal_directedness = emotion_data.get("goal_directedness", self.goal_directedness)
        self.securing_rate = emotion_data.get("securing_rate", self.securing_rate)

    def display_emotions(self):
        st.write("### Emotional State")
        emotions = {
            "Arousal": self.arousal,
            "Valence": self.valence,
            "Dominance": self.dominance,
            "Sadness": self.sadness,
            "Anger": self.anger,
            "Joy": self.joy,
            "Fear": self.fear,
            "Selection Threshold": self.selection_threshold,
            "Resolution Level": self.resolution_level,
            "Goal Directedness": self.goal_directedness,
            "Securing Rate": self.securing_rate,
        }
        for emotion, value in emotions.items():
            st.write(f"{emotion}:")
            st.progress(max(0.0, min(1.0, value)))