import streamlit as st
from tools.base_tools import Ui_Tool
from storage.logger_config import logger

class TestTool(Ui_Tool):
    """
    A simple test tool to verify the tool system is working correctly.
    """
    name = 'TestTool'
    icon = '🧪'
    title = 'Test Tool'
    description = 'A simple tool for testing the agent system. Input any text to see it processed.'
    category = 'Utilities'
    version = '1.0'
    author = 'System'
    
    def __init__(self):
        super().__init__()
        self.settings = {
            'echo_input': True,
            'add_emoji': True,
            'uppercase': False
        }
    
    def _run(self, input_str):
        """
        Process the input based on current settings.
        
        Args:
            input_str: The input string to process
            
        Returns:
            Processed string based on settings
        """
        logger.debug(f'TestTool execution with input: {input_str}')
        
        # Apply settings
        result = input_str
        
        # Apply uppercase if enabled
        if self.settings['uppercase']:
            result = result.upper()
        
        # Add emoji if enabled
        if self.settings['add_emoji']:
            result = f"✅ {result}"
        
        # Format response
        if self.settings['echo_input']:
            return f"Test successful! You said: {result}"
        else:
            return "Test successful!"
    
    def _ui(self):
        """Render custom UI controls for tool settings."""
        with st.expander("Test Tool Settings"):
            # Echo input setting
            echo_input = st.checkbox(
                "Echo input in response",
                value=self.settings['echo_input'],
                key=f"test_echo_{id(self)}",
                help="Include the input in the response"
            )
            self.settings['echo_input'] = echo_input
            
            # Add emoji setting
            add_emoji = st.checkbox(
                "Add emoji to response",
                value=self.settings['add_emoji'],
                key=f"test_emoji_{id(self)}",
                help="Include an emoji in the response"
            )
            self.settings['add_emoji'] = add_emoji
            
            # Uppercase setting
            uppercase = st.checkbox(
                "Convert to uppercase",
                value=self.settings['uppercase'],
                key=f"test_uppercase_{id(self)}",
                help="Convert the response to uppercase"
            )
            self.settings['uppercase'] = uppercase
            
            # Test button
            if st.button("Test Now", key=f"test_button_{id(self)}"):
                test_result = self._run("This is a test message")
                st.success(test_result)
