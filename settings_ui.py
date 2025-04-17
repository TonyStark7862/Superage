import streamlit as st
import os
from storage.logger_config import logger

def settings_page():
    """
    Render the settings page with configuration options.
    Includes agent settings, session management, and UI preferences.
    """
    st.markdown("## ⚙️ Settings")
    
    # Create tabs for different settings categories
    tab1, tab2, tab3 = st.tabs(["Agent Settings", "Session Management", "UI Preferences"])
    
    with tab1:
        agent_settings()
    
    with tab2:
        session_management()
    
    with tab3:
        ui_preferences()

def agent_settings():
    """Configure agent behavior settings."""
    st.markdown("### Agent Configuration")
    
    # Agent behavior settings
    col1, col2 = st.columns(2)
    
    with col1:
        # Custom prefix for agent prompts
        st.text_area(
            'Prefix',
            key='prefix_input',
            value=st.session_state.prefix,
            placeholder='Text to prepend to user queries',
            on_change=update_prefix
        )
    
    with col2:
        # Custom suffix for agent prompts
        st.text_area(
            'Suffix',
            key='suffix_input',
            value=st.session_state.suffix,
            placeholder='Text to append to user queries',
            on_change=update_suffix
        )
    
    # Show thinking toggle
    st.checkbox(
        "Show agent thinking process",
        value=st.session_state.show_thinking,
        key='show_thinking_toggle',
        on_change=update_show_thinking
    )
    
    # Additional agent settings
    st.markdown("### Planning Behavior")
    
    # Planning depth setting
    planning_depth = st.slider(
        "Planning Depth",
        min_value=1,
        max_value=5,
        value=3,
        help="Controls how detailed the agent's planning process will be"
    )
    
    # Execution style
    execution_style = st.radio(
        "Execution Style",
        options=["Sequential", "Adaptive"],
        index=0,
        help="Sequential: Execute steps in order. Adaptive: Adjust plan during execution"
    )
    
    # Maximum steps
    max_steps = st.number_input(
        "Maximum Steps",
        min_value=1,
        max_value=10,
        value=5,
        help="Maximum number of steps in the execution plan"
    )

def session_management():
    """Manage chat sessions and history."""
    st.markdown("### Session Management")
    
    # Display current session info
    current_session = st.session_state.session_name.get(
        st.session_state.session_id, 
        st.session_state.session_id
    )
    
    st.info(f"Current Session: {current_session}")
    
    # Session renaming
    new_name = st.text_input(
        "Rename Current Session",
        placeholder="Enter new session name",
        key="session_rename_input"
    )
    
    if st.button("Rename", key="rename_session_button"):
        if new_name:
            rename_session(new_name)
    
    # Session deletion
    st.markdown("### Clear History")
    
    if st.button("Clear Current Session History", key="clear_session_button", type="secondary"):
        if st.session_state.session_id in st.session_state.storage.get_all_sessions():
            clear_confirmation = st.checkbox(
                "Confirm deletion? This cannot be undone.",
                key="confirm_clear_session"
            )
            
            if clear_confirmation and st.button("Confirm Clear", key="confirm_clear_button", type="primary"):
                clear_session_history()

def ui_preferences():
    """Configure UI preferences."""
    st.markdown("### Interface Preferences")
    
    # Theme selection already handled by theme toggler in sidebar
    st.markdown("#### Chat Display")
    
    # Message display density
    message_density = st.select_slider(
        "Message Density",
        options=["Compact", "Balanced", "Spacious"],
        value="Balanced",
        help="Controls the spacing between chat messages"
    )
    
    # Code block theme
    code_theme = st.selectbox(
        "Code Block Theme",
        options=["Default", "Dark", "Light", "GitHub"],
        index=0,
        help="Theme for displaying code blocks in chat"
    )
    
    # Timestamp display
    show_timestamps = st.checkbox(
        "Show Message Timestamps",
        value=False,
        help="Display timestamp for each message"
    )
    
    st.markdown("#### Agent Visualization")
    
    # Thinking style
    thinking_style = st.radio(
        "Thinking Visualization Style",
        options=["Step by Step", "Mindmap", "Simple List"],
        index=0,
        help="How the agent's thinking process is displayed"
    )
    
    # Apply button
    if st.button("Apply UI Settings", type="primary"):
        st.success("UI preferences applied!")
        # These settings would be saved to session state in a full implementation

# Helper functions for settings updates
def update_prefix():
    """Update the prefix in session state."""
    st.session_state.prefix = st.session_state.prefix_input

def update_suffix():
    """Update the suffix in session state."""
    st.session_state.suffix = st.session_state.suffix_input

def update_show_thinking():
    """Update the show thinking setting in session state."""
    st.session_state.show_thinking = st.session_state.show_thinking_toggle

def rename_session(new_name):
    """Rename the current session."""
    if new_name:
        st.session_state.storage.save_session_name(st.session_state.session_id, new_name)
        st.success(f"Session renamed to: {new_name}")
        st.session_state.session_name = st.session_state.storage.get_all_sessions_names()
        st.rerun()

def clear_session_history():
    """Clear the history of the current session."""
    st.session_state.storage.delete_session(st.session_state.session_id)
    st.session_state.session_id = st.session_state.session_id  # Keep the same session ID
    st.success("Session history cleared!")
    st.rerun()
