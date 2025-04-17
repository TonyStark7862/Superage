import os
import csv
import pandas as pd
from datetime import datetime
from storage.logger_config import logger

class CSVStorage:
    """
    Handles persistent storage of chat sessions and history using CSV files.
    Replaces the SQLAlchemy-based storage with simple CSV files.
    """
    def __init__(self, base_dir):
        """
        Initialize the CSV storage system.
        
        Args:
            base_dir: Base directory for storing CSV files
        """
        # Set up directory structure
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data')
        self.sessions_file = os.path.join(self.data_dir, 'sessions.csv')
        self.history_file = os.path.join(self.data_dir, 'history.csv')
        self.session_names_file = os.path.join(self.data_dir, 'session_names.csv')
        self.feedback_file = os.path.join(self.data_dir, 'feedback.csv')
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize CSV files if they don't exist
        self._initialize_files()
        
        logger.info(f"Initialized CSV storage in: {self.data_dir}")

    def _initialize_files(self):
        """Initialize CSV files with headers if they don't exist."""
        # Sessions file
        if not os.path.exists(self.sessions_file):
            with open(self.sessions_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'session_id', 'created_at'])
            logger.info(f"Created sessions file: {self.sessions_file}")
        
        # History file
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'session_id', 'role', 'content', 'timestamp'])
            logger.info(f"Created history file: {self.history_file}")
        
        # Session names file
        if not os.path.exists(self.session_names_file):
            with open(self.session_names_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'session_id', 'name'])
            logger.info(f"Created session names file: {self.session_names_file}")
        
        # Feedback file
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'session_id', 'question', 'answer', 'timestamp', 'user_email', 'feedback_rating'])
            logger.info(f"Created feedback file: {self.feedback_file}")

    def _get_next_id(self, file_path, id_col='id'):
        """
        Get the next available ID for a CSV file.
        
        Args:
            file_path: Path to the CSV file
            id_col: Name of the ID column
            
        Returns:
            Next available ID (integer)
        """
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # If the file is empty (only header), return 1
            if df.empty:
                return 1
            
            # Return the max ID + 1
            return df[id_col].max() + 1
        except Exception as e:
            logger.error(f"Error getting next ID from {file_path}: {e}")
            return 1

    def save_chat_message(self, session_id, role, content, name):
        """
        Save a chat message to the history CSV file.
        
        Args:
            session_id: ID of the session
            role: Role of the message sender (user/assistant)
            content: Content of the message
            name: Custom name for the session
        """
        try:
            # Ensure the session exists
            self._ensure_session_exists(session_id, name)
            
            # Get next ID for history
            next_id = self._get_next_id(self.history_file)
            
            # Prepare the new row
            timestamp = datetime.now().isoformat()
            new_row = [next_id, session_id, role, content, timestamp]
            
            # Append to history file
            with open(self.history_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(new_row)
            
            logger.debug(f"Saved message for session {session_id}, role: {role}")
            
            # If this is a user-assistant interaction, log it to feedback
            if role == "assistant":
                self._log_feedback(session_id, content)
                
        except Exception as e:
            logger.error(f"Error saving chat message: {e}")

    def _log_feedback(self, session_id, answer):
        """
        Log an assistant response to the feedback file.
        
        Args:
            session_id: ID of the session
            answer: The assistant's response
        """
        try:
            # Get the most recent user question for this session
            df = pd.read_csv(self.history_file)
            session_messages = df[df['session_id'] == session_id]
            
            if session_messages.empty:
                return
            
            # Find the most recent user message
            user_messages = session_messages[session_messages['role'] == 'user']
            if user_messages.empty:
                return
            
            # Get the last user message (question)
            question = user_messages.iloc[-1]['content']
            
            # Get user email from session data (if available)
            user_email = self._get_user_email_for_session(session_id)
            
            # Get next ID for feedback
            next_id = self._get_next_id(self.feedback_file)
            
            # Prepare the new row
            timestamp = datetime.now().isoformat()
            new_row = [next_id, session_id, question, answer, timestamp, user_email, ""]
            
            # Append to feedback file
            with open(self.feedback_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(new_row)
            
            logger.debug(f"Logged feedback entry for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error logging feedback: {e}")

    def _get_user_email_for_session(self, session_id):
        """
        Get the user email associated with a session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            User email or empty string if not found
        """
        try:
            # This would be implemented when we add user authentication
            # For now, return empty string
            return ""
        except Exception:
            return ""

    def _ensure_session_exists(self, session_id, name):
        """
        Ensure a session exists in the sessions CSV file.
        If it doesn't exist, create it.
        
        Args:
            session_id: ID of the session
            name: Custom name for the session
        """
        try:
            # Check if session exists
            df = pd.read_csv(self.sessions_file)
            session_exists = not df[df['session_id'] == session_id].empty
            
            if not session_exists:
                # Create new session
                next_id = self._get_next_id(self.sessions_file)
                created_at = datetime.now().isoformat()
                
                # Add to sessions file
                with open(self.sessions_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([next_id, session_id, created_at])
                
                # Add session name
                self.save_session_name(session_id, name)
                
                logger.info(f"Created new session: {session_id}")
        except Exception as e:
            logger.error(f"Error ensuring session exists: {e}")

    def save_session_name(self, session_id, name):
        """
        Save or update a session name.
        
        Args:
            session_id: ID of the session
            name: New name for the session
        """
        try:
            # Check if name already exists
            df = pd.read_csv(self.session_names_file)
            name_exists = not df[df['session_id'] == session_id].empty
            
            if name_exists:
                # Update existing name
                df.loc[df['session_id'] == session_id, 'name'] = name
                df.to_csv(self.session_names_file, index=False)
            else:
                # Add new name
                next_id = self._get_next_id(self.session_names_file)
                with open(self.session_names_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([next_id, session_id, name])
            
            logger.debug(f"Updated session name for session {session_id}: {name}")
        except Exception as e:
            logger.error(f"Error saving session name: {e}")

    def get_chat_history(self, session_id):
        """
        Get chat history for a specific session.
        
        Args:
            session_id: ID of the session to retrieve history for
            
        Returns:
            List of chat messages as dictionaries
        """
        try:
            # Read history file
            df = pd.read_csv(self.history_file)
            
            # Filter by session_id
            session_history = df[df['session_id'] == session_id]
            
            # Convert to list of dictionaries
            history = []
            for _, row in session_history.iterrows():
                history.append({
                    "role": row['role'],
                    "content": row['content']
                })
            
            return history
        except Exception as e:
            logger.error(f"Error retrieving chat history: {e}")
            return []

    def get_all_sessions(self):
        """
        Get all session IDs.
        
        Returns:
            List of session IDs
        """
        try:
            # Read sessions file
            df = pd.read_csv(self.sessions_file)
            
            # Get all session IDs
            return df['session_id'].tolist()
        except Exception as e:
            logger.error(f"Error retrieving all sessions: {e}")
            return []

    def get_all_sessions_names(self):
        """
        Get a mapping of session IDs to their custom names.
        
        Returns:
            Dictionary mapping session IDs to names
        """
        try:
            # Read session names file
            df = pd.read_csv(self.session_names_file)
            
            # Convert to dictionary
            return dict(zip(df['session_id'], df['name']))
        except Exception as e:
            logger.error(f"Error retrieving session names: {e}")
            return {}
    
    def delete_session(self, session_id):
        """
        Delete a session and all its associated data.
        
        Args:
            session_id: ID of the session to delete
        """
        try:
            # Read all files
            sessions_df = pd.read_csv(self.sessions_file)
            history_df = pd.read_csv(self.history_file)
            names_df = pd.read_csv(self.session_names_file)
            
            # Filter out the session to delete
            sessions_df = sessions_df[sessions_df['session_id'] != session_id]
            history_df = history_df[history_df['session_id'] != session_id]
            names_df = names_df[names_df['session_id'] != session_id]
            
            # Write back the filtered data
            sessions_df.to_csv(self.sessions_file, index=False)
            history_df.to_csv(self.history_file, index=False)
            names_df.to_csv(self.session_names_file, index=False)
            
            logger.info(f"Deleted session: {session_id}")
        except Exception as e:
            logger.error(f"Error deleting session: {e}")

    def save_user_feedback(self, session_id, question, answer, rating, user_email=""):
        """
        Save user feedback for a specific interaction.
        
        Args:
            session_id: ID of the session
            question: The user's question
            answer: The assistant's answer
            rating: User's rating/feedback
            user_email: User's email if available
        """
        try:
            # Read feedback file
            df = pd.read_csv(self.feedback_file)
            
            # Find the matching entry
            mask = (df['session_id'] == session_id) & (df['question'] == question) & (df['answer'] == answer)
            
            if not df[mask].empty:
                # Update existing entry
                df.loc[mask, 'feedback_rating'] = rating
                if user_email:
                    df.loc[mask, 'user_email'] = user_email
                df.to_csv(self.feedback_file, index=False)
            else:
                # Add new entry
                next_id = self._get_next_id(self.feedback_file)
                timestamp = datetime.now().isoformat()
                with open(self.feedback_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([next_id, session_id, question, answer, timestamp, user_email, rating])
            
            logger.info(f"Saved feedback for session {session_id}: rating={rating}")
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
