import streamlit as st

class CharacterState:
    """
    A class to represent and manage the emotional and cognitive state of a character.
    Uses a dimensional model of emotion (arousal, valence, dominance) combined with
    basic emotions (sadness, anger, joy, fear) and cognitive parameters.
    
    Attributes:
        arousal (float): Activation level (0.0=calm, 1.0=excited)
        valence (float): Positivity (0.0=negative, 1.0=positive)
        dominance (float): Control level (0.0=submissive, 1.0=dominant)
        sadness (float): Sadness intensity (0.0-1.0)
        anger (float): Anger intensity (0.0-1.0)
        joy (float): Joy intensity (0.0-1.0)
        fear (float): Fear intensity (0.0-1.0)
        selection_threshold (float): Decision-making threshold (0.0-1.0)
        resolution_level (float): Problem-solving persistence (0.0-1.0)
        goal_directedness (float): Focus on objectives (0.0-1.0)
        securing_rate (float): Resource protection tendency (0.0-1.0)
    """
    
    def __init__(self, arousal=0.5, valence=0.5, dominance=0.5, 
                 sadness=0.0, anger=0.0, joy=0.0, fear=0.0,
                 selection_threshold=0.5, resolution_level=0.5, 
                 goal_directedness=0.5, securing_rate=0.5):
        """
        Initialize character state with default neutral values (0.5) unless specified.
        
        Args:
            arousal: Initial activation level (default: 0.5)
            valence: Initial positivity (default: 0.5)
            dominance: Initial control level (default: 0.5)
            sadness: Initial sadness (default: 0.0)
            anger: Initial anger (default: 0.0)
            joy: Initial joy (default: 0.0)
            fear: Initial fear (default: 0.0)
            selection_threshold: Initial decision threshold (default: 0.5)
            resolution_level: Initial problem-solving persistence (default: 0.5)
            goal_directedness: Initial focus on objectives (default: 0.5)
            securing_rate: Initial resource protection tendency (default: 0.5)
        """
        # Core emotional dimensions
        self.arousal = arousal          # Energy level (calm vs excited)
        self.valence = valence          # Emotional positivity
        self.dominance = dominance      # Sense of control
        
        # Basic emotions
        self.sadness = sadness          # Sadness intensity
        self.anger = anger              # Anger intensity
        self.joy = joy                  # Happiness intensity
        self.fear = fear                # Fear intensity
        
        # Cognitive parameters
        self.selection_threshold = selection_threshold  # Decision-making rigor
        self.resolution_level = resolution_level        # Problem-solving persistence
        self.goal_directedness = goal_directedness      # Focus on objectives
        self.securing_rate = securing_rate              # Resource protection tendency

    def update_emotions(self, emotion_data):
        """
        Update emotional state with new values from a dictionary.
        Only updates attributes that are explicitly provided in emotion_data.
        
        Args:
            emotion_data (dict): Dictionary containing any of the emotional/cognitive 
                               attributes to update (e.g., {'joy': 0.8, 'fear': 0.2})
        """
        # Update each attribute if present in the input data
        self.arousal = emotion_data.get("arousal", self.arousal)
        self.valence = emotion_data.get("valence", self.valence)
        self.dominance = emotion_data.get("dominance", self.dominance)
        
        # Basic emotions
        self.sadness = emotion_data.get("sadness", self.sadness)
        self.anger = emotion_data.get("anger", self.anger)
        self.joy = emotion_data.get("joy", self.joy)
        self.fear = emotion_data.get("fear", self.fear)
        
        # Cognitive parameters
        self.selection_threshold = emotion_data.get("selection_threshold", self.selection_threshold)
        self.resolution_level = emotion_data.get("resolution_level", self.resolution_level)
        self.goal_directedness = emotion_data.get("goal_directedness", self.goal_directedness)
        self.securing_rate = emotion_data.get("securing_rate", self.securing_rate)

    def display_emotions(self):
        """
        Display all emotional and cognitive states as Streamlit progress bars.
        Formats the display with clear section headers and visual indicators.
        """
        st.write("### Emotional State")
        
        # Group related parameters for organized display
        emotion_groups = {
            "Core Dimensions": {
                "Arousal": self.arousal,
                "Valence": self.valence,
                "Dominance": self.dominance
            },
            "Basic Emotions": {
                "Sadness": self.sadness,
                "Anger": self.anger,
                "Joy": self.joy,
                "Fear": self.fear
            },
            "Cognitive Traits": {
                "Selection Threshold": self.selection_threshold,
                "Resolution Level": self.resolution_level,
                "Goal Directedness": self.goal_directedness,
                "Securing Rate": self.securing_rate
            }
        }
        
        # Display each group with appropriate formatting
        for group_name, emotions in emotion_groups.items():
            st.write(f"**{group_name}**")
            for emotion, value in emotions.items():
                # Ensure value is within valid range before display
                clamped_value = max(0.0, min(1.0, value))
                st.write(f"{emotion}:")
                st.progress(clamped_value)
            st.write("---")  # Visual separator between groups
