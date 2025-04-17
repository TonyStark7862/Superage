import streamlit as st
from storage.logger_config import logger

class Custom_chat_callback:
    """
    Callback handler for monitoring the agent's chat process.
    Tracks thinking steps and execution progress.
    """
    
    def __init__(self):
        """Initialize the callback handler."""
        self.thinking_steps = []
        self.current_step = 0
    
    def on_thinking_start(self, query):
        """
        Called when the agent starts the thinking process.
        
        Args:
            query: The user query that triggered the thinking
        """
        # Reset thinking steps
        self.thinking_steps = []
        self.current_step = 0
        
        # Add initial step
        self.add_thinking_step(f"Analyzing query: {query}")
    
    def add_thinking_step(self, step):
        """
        Add a step to the thinking process.
        
        Args:
            step: The thinking step description
        """
        self.thinking_steps.append(step)
        
        # Update session state
        if "agent_thinking" in st.session_state:
            st.session_state.agent_thinking = self.thinking_steps
    
    def on_planning_complete(self, plan):
        """
        Called when the agent completes the planning phase.
        
        Args:
            plan: The generated execution plan
        """
        # Add planning completion step
        self.add_thinking_step("Planning complete. Generated execution plan.")
        
        # Update session state
        if "current_plan" in st.session_state:
            st.session_state.current_plan = "Execution Plan"
        
        if "plan_steps" in st.session_state:
            st.session_state.plan_steps = plan
    
    def on_execution_start(self):
        """Called when the agent starts executing the plan."""
        self.add_thinking_step("Starting execution of the plan.")
    
    def on_step_start(self, step_number, step_description):
        """
        Called when the agent starts executing a step.
        
        Args:
            step_number: The index of the current step
            step_description: Description of the step
        """
        # Update current step
        self.current_step = step_number
        
        # Add step start notification
        self.add_thinking_step(f"Executing step {step_number + 1}: {step_description}")
        
        # Update session state
        if "current_step" in st.session_state:
            st.session_state.current_step = step_number
    
    def on_step_complete(self, step_number, result):
        """
        Called when the agent completes a step.
        
        Args:
            step_number: The index of the completed step
            result: The result of the step execution
        """
        # Add step completion notification
        self.add_thinking_step(f"Completed step {step_number + 1} with result: {result}")
    
    def on_execution_complete(self):
        """Called when the agent completes execution of the plan."""
        self.add_thinking_step("Plan execution completed successfully.")
    
    def on_error(self, error, step_number=None):
        """
        Called when an error occurs during execution.
        
        Args:
            error: The error that occurred
            step_number: The step number where the error occurred (if applicable)
        """
        # Format the error message
        if step_number is not None:
            error_msg = f"Error in step {step_number + 1}: {error}"
        else:
            error_msg = f"Error during execution: {error}"
        
        # Add error to thinking steps
        self.add_thinking_step(error_msg)
        
        # Log the error
        logger.error(error_msg)


class ToolCallback:
    """
    Callback handler for monitoring tool usage.
    Tracks when tools are invoked and their results.
    """
    
    def __init__(self):
        """Initialize the tool callback handler."""
        pass
    
    def on_tool_start(self, tool_name, input_str):
        """
        Called when a tool starts execution.
        
        Args:
            tool_name: Name of the tool being executed
            input_str: Input provided to the tool
        """
        # Log tool execution start
        logger.info(f"Tool '{tool_name}' started with input: {input_str}")
    
    def on_tool_end(self, tool_name, output):
        """
        Called when a tool completes execution.
        
        Args:
            tool_name: Name of the tool that was executed
            output: Output from the tool
        """
        # Log tool execution completion
        logger.info(f"Tool '{tool_name}' completed with output: {output}")
    
    def on_tool_error(self, tool_name, error):
        """
        Called when a tool encounters an error.
        
        Args:
            tool_name: Name of the tool where the error occurred
            error: The error that occurred
        """
        # Format error message
        error_msg = f"Error in tool '{tool_name}': {error}"
        
        # Log the error
        logger.error(error_msg)
