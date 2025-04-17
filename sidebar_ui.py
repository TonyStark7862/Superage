import streamlit as st
from datetime import datetime

def sidebar():
    """
    Render the sidebar with session management and system info.
    """
    with st.sidebar:
        # App logo and title
        st.markdown(
            """
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span style="font-size: 2rem; margin-right: 0.5rem;">🤖</span>
                <h1 style="margin: 0; color: var(--primary-color);">OmniTool</h1>
            </div>
            <p style="margin-top: 0;">Next-gen agentic interface with a pluggable tool system</p>
            """, 
            unsafe_allow_html=True
        )
        
        # User info section
        if st.session_state.user_authenticated and st.session_state.user_email:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; margin-bottom: 1rem; padding: 10px; background-color: var(--hover-bg); border-radius: 5px;">
                    <span style="font-size: 1.5rem; margin-right: 0.5rem;">👤</span>
                    <div>
                        <p style="margin: 0; font-weight: bold;">Welcome,</p>
                        <p style="margin: 0; font-size: 0.9rem;">{st.session_state.user_email}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # New session button
        if st.button("➕ New Session", use_container_width=True):
            start_new_session()
            
        # Session management section
        st.markdown("### 📚 Sessions")
        
        # Get all sessions from storage
        session_id_list = st.session_state.storage.get_all_sessions()
        st.session_state.session_name = st.session_state.storage.get_all_sessions_names()
        
        # Session list
        if session_id_list:
            show_session_list(session_id_list)
        else:
            st.info("No previous sessions found.")
        
        st.markdown("---")
        
        # System info
        show_system_info()

def start_new_session():
    """Start a new chat session."""
    st.session_state.session_id = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    st.session_state.selected_page = 'Chat'
    
    # Reset agent thinking and plans for new session
    st.session_state.agent_thinking = []
    st.session_state.current_plan = None
    st.session_state.plan_steps = []
    st.session_state.current_step = 0
    
    # Force page refresh
    st.rerun()

def show_session_list(session_id_list):
    """
    Display a list of all sessions with options to select and manage them.
    
    Args:
        session_id_list: List of session IDs
    """
    # Display sessions in reverse chronological order (newest first)
    for session_id in reversed(session_id_list):
        # Get session name (or use session ID if no custom name)
        session_name = st.session_state.session_name.get(session_id, session_id)
        
        # Create a container for the session row
        with st.container():
            col1, col2 = st.columns([5, 1])
            
            # Main session button
            if col1.button(session_name, key=f"session_{session_id}", use_container_width=True):
                st.session_state.session_id = session_id
                st.rerun()
            
            # Delete button
            if col2.button("🗑️", key=f"delete_{session_id}"):
                st.session_state.delete_session_id = session_id
                st.session_state.show_delete_confirm = True
    
    # Show delete confirmation if needed
    if st.session_state.get("show_delete_confirm", False):
        delete_session_id = st.session_state.get("delete_session_id")
        delete_session_name = st.session_state.session_name.get(delete_session_id, delete_session_id)
        
        st.warning(f"Delete session: {delete_session_name}?")
        
        col1, col2 = st.columns(2)
        if col1.button("Confirm", key="confirm_delete"):
            st.session_state.storage.delete_session(delete_session_id)
            st.session_state.show_delete_confirm = False
            st.session_state.pop("delete_session_id", None)
            st.rerun()
        
        if col2.button("Cancel", key="cancel_delete"):
            st.session_state.show_delete_confirm = False
            st.session_state.pop("delete_session_id", None)
            st.rerun()

def show_system_info():
    """Display system information in the sidebar."""
    st.markdown("### 📊 System Info")
    
    # Show active tools count
    tools_count = len(st.session_state.selected_tools)
    st.markdown(f"**Active Tools:** {tools_count}")
    
    # Add a thinking toggle button
    thinking_label = "Hide Thinking" if st.session_state.show_thinking else "Show Thinking"
    if st.button(thinking_label, key="toggle_thinking"):
        st.session_state.show_thinking = not st.session_state.show_thinking
        st.rerun()
    
    # Add feedback button
    if st.button("📝 Provide Feedback", key="feedback_button"):
        st.session_state.show_feedback = True
    
    # Show feedback form if requested
    if st.session_state.get("show_feedback", False):
        with st.form("feedback_form"):
            st.markdown("### Your Feedback")
            feedback_rating = st.slider("Rate your experience", 1, 5, 3)
            feedback_text = st.text_area("Comments (optional)")
            submitted = st.form_submit_button("Submit Feedback")
            
            if submitted:
                # Save feedback (when we have a question/answer to associate it with)
                if st.session_state.session_id and hasattr(st.session_state.storage, "save_user_feedback"):
                    # We'd normally associate this with the last Q&A pair
                    # For now, just log it with session info
                    st.session_state.storage.save_user_feedback(
                        st.session_state.session_id,
                        "General feedback",
                        "System feedback",
                        str(feedback_rating),
                        st.session_state.user_email
                    )
                    st.success("Thank you for your feedback!")
                    st.session_state.show_feedback = False
    
    # Add system version
    st.markdown("**Version:** 1.0.0")
    
    # Add footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: var(--text-secondary); font-size: 0.8rem;">
            © 2025 OmniTool Project<br>
            <a href="https://github.com/user/OmniTool" target="_blank" style="color: var(--primary-color);">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )


def start_new_session():
    """Start a new chat session."""
    st.session_state.session_id = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    st.session_state.selected_page = 'Chat'
    
    # Reset agent thinking and plans for new session
    st.session_state.agent_thinking = []
    st.session_state.current_plan = None
    st.session_state.plan_steps = []
    st.session_state.current_step = 0
    
    # Force page refresh
    st.rerun()

def show_session_list(session_id_list):
    """
    Display a list of all sessions with options to select and manage them.
    
    Args:
        session_id_list: List of session IDs
    """
    # Display sessions in reverse chronological order (newest first)
    for session_id in reversed(session_id_list):
        # Get session name (or use session ID if no custom name)
        session_name = st.session_state.session_name.get(session_id, session_id)
        
        # Create a container for the session row
        with st.container():
            col1, col2 = st.columns([5, 1])
            
            # Main session button
            if col1.button(session_name, key=f"session_{session_id}", use_container_width=True):
                st.session_state.session_id = session_id
                st.rerun()
            
            # Delete button
            if col2.button("🗑️", key=f"delete_{session_id}"):
                st.session_state.delete_session_id = session_id
                st.session_state.show_delete_confirm = True
    
    # Show delete confirmation if needed
    if st.session_state.get("show_delete_confirm", False):
        delete_session_id = st.session_state.get("delete_session_id")
        delete_session_name = st.session_state.session_name.get(delete_session_id, delete_session_id)
        
        st.warning(f"Delete session: {delete_session_name}?")
        
        col1, col2 = st.columns(2)
        if col1.button("Confirm", key="confirm_delete"):
            st.session_state.storage.delete_session(delete_session_id)
            st.session_state.show_delete_confirm = False
            st.session_state.pop("delete_session_id", None)
            st.rerun()
        
        if col2.button("Cancel", key="cancel_delete"):
            st.session_state.show_delete_confirm = False
            st.session_state.pop("delete_session_id", None)
            st.rerun()

def show_system_info():
    """Display system information in the sidebar."""
    st.markdown("### 📊 System Info")
    
    # Show active tools count
    tools_count = len(st.session_state.selected_tools)
    st.markdown(f"**Active Tools:** {tools_count}")
    
    # Add a thinking toggle button
    thinking_label = "Hide Thinking" if st.session_state.show_thinking else "Show Thinking"
    if st.button(thinking_label, key="toggle_thinking"):
        st.session_state.show_thinking = not st.session_state.show_thinking
        st.rerun()
    
    # Add system version
    st.markdown("**Version:** 1.0.0")
    
    # Add footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: var(--text-secondary); font-size: 0.8rem;">
            © 2025 OmniTool Project<br>
            <a href="https://github.com/user/OmniTool" target="_blank" style="color: var(--primary-color);">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )
