import streamlit as st
from agents.agent import AgentConfig
from storage.logger_config import logger
from .callbacks_ui import Custom_chat_callback, ToolCallback

def chat_page():
    """
    Render the chat interface with agent thinking visualization.
    """
    # Display selected tools information
    st.sidebar.markdown("### 🧰 Active Tools")
    tool_names = ', '.join(st.session_state.selected_tools)
    st.sidebar.info(f"Selected tools: {tool_names}")
    
    # Main chat area
    st.markdown("## 💬 Chat")
    
    # Initialize agent if not already done
    if "agent_instance" not in st.session_state:
        configure_agent(
            st.session_state.agent,
            st.session_state.tools,
            st.session_state.memory,
            st.session_state.session_id,
            st.session_state.selected_tools
        )
    
    # Display chat history
    display_chat_history()
    
    # Display agent thinking area if enabled
    if st.session_state.show_thinking and st.session_state.agent_thinking:
        show_agent_thinking()
    
    # Display current plan if available
    if st.session_state.current_plan and st.session_state.plan_steps:
        show_current_plan()
    
    # Get user input
    user_input = st.chat_input("Type your message here...")
    
    # Process user input if provided
    if user_input:
        process_user_input(user_input)

def display_chat_history():
    """Display the chat history."""
    if "session_id" in st.session_state:
        # Get messages from storage
        messages = st.session_state.storage.get_chat_history(st.session_state.session_id)
        st.session_state.messages = messages
        
        # Display all messages
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

def show_agent_thinking():
    """Display the agent's thinking process."""
    with st.expander("🧠 Agent Thinking Process", expanded=True):
        for i, thought in enumerate(st.session_state.agent_thinking):
            st.markdown(f"**Step {i+1}**: {thought}")

def show_current_plan():
    """Display the current execution plan."""
    with st.expander("📋 Execution Plan", expanded=True):
        # Display plan overview
        st.markdown(f"### {st.session_state.current_plan}")
        
        # Create a progress indicator for the plan
        total_steps = len(st.session_state.plan_steps)
        current_step = st.session_state.current_step
        
        # Calculate progress percentage
        if total_steps > 0:
            progress = min(current_step / total_steps, 1.0)
            st.progress(progress)
        
        # Display individual steps
        for i, step in enumerate(st.session_state.plan_steps):
            # Determine step status
            if i < current_step:
                status = "✅ "  # Completed
            elif i == current_step:
                status = "⏳ "  # In progress
            else:
                status = "⏱️ "  # Pending
            
            # Format step based on type
            if step["type"] == "tool":
                st.markdown(f"{status} **Step {i+1}**: Use **{step['tool']}** - {step['description']}")
            else:
                st.markdown(f"{status} **Step {i+1}**: {step['description']}")

def process_user_input(user_input):
    """Process user input and generate agent response."""
    # Add user message to chat
    with st.chat_message("user"):
        st.write(user_input)
    
    # Store user message in history with user email if available
    session_name = st.session_state.session_name.get(st.session_state.session_id, st.session_state.session_id)
    st.session_state.storage.save_chat_message(
        st.session_state.session_id, 
        "user", 
        user_input, 
        session_name
    )
    
    # Track this question for feedback purposes
    st.session_state.last_question = user_input
    
    # Clear previous thinking and plan
    st.session_state.agent_thinking = []
    st.session_state.current_plan = None
    st.session_state.plan_steps = []
    st.session_state.current_step = 0
    
    # Process user query with the agent
    with st.chat_message("assistant"):
        # Create spinner while processing
        with st.spinner("Thinking..."):
            # Add prefix and suffix if defined
            formatted_input = st.session_state.prefix + user_input + st.session_state.suffix
            
            # Get response from agent
            response = st.session_state.agent_instance.run(
                input=formatted_input,
                callbacks=None
            )
            
            # Display the response
            st.write(response)
            
            # Add feedback buttons after response
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.messages)}"):
                    log_feedback(user_input, response, "helpful")
                    st.success("Thanks for your feedback!")
            with col2:
                if st.button("👎 Not Helpful", key=f"not_helpful_{len(st.session_state.messages)}"):
                    log_feedback(user_input, response, "not_helpful")
                    st.info("Thanks for your feedback! We'll try to improve.")
    
    # Store assistant response in history
    st.session_state.storage.save_chat_message(
        st.session_state.session_id, 
        "assistant", 
        response, 
        session_name
    )
    
    # Store this response for feedback purposes
    st.session_state.last_response = response

def log_feedback(question, answer, rating):
    """
    Log user feedback about a specific Q&A pair.
    
    Args:
        question: The user's question
        answer: The assistant's response
        rating: User rating (helpful/not_helpful)
    """
    if hasattr(st.session_state.storage, "save_user_feedback"):
        user_email = st.session_state.user_email if "user_email" in st.session_state else ""
        st.session_state.storage.save_user_feedback(
            st.session_state.session_id,
            question,
            answer,
            rating,
            user_email
        )

def configure_agent(agent_type, tools, memory, session_id, selected_tools_names):
    """Configure and initialize the agent."""
    logger.info(f'Agent config for session {session_id} with agent: {agent_type}, tools: {selected_tools_names}')
    
    # Initialize the agent config
    agent_config = AgentConfig(agent_type, tools, memory)
    
    # Store the agent instance in session state
    st.session_state.agent_instance = agent_config.initialize_agent()
