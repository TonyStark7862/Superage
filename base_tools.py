import streamlit as st
from storage.logger_config import logger

class Ui_Tool:
    """
    Base class for all tools in the system.
    Tools should inherit from this class and implement _run method.
    They can optionally implement _ui method to provide custom UI controls.
    """
    # Default attributes that should be overridden by subclasses
    name = 'Base_tool'
    link = r'https://github.com/user/OmniTool/tree/master'
    icon = '🔧'
    title = 'Base Tool'
    description = 'Base tool description'
    
    # Optional attributes
    category = 'General'  # Tool category for organization
    version = '1.0'       # Tool version
    author = 'System'     # Tool author
    requires = []         # List of required tools or dependencies
    
    def __init__(self):
        """Initialize the tool with default settings."""
        # Tool-specific settings can be initialized here
        self.settings = {}
        
    def _run(self, input_str):
        """
        Execute the tool with the given input.
        This method should be overridden by subclasses.
        
        Args:
            input_str: The input string to process
            
        Returns:
            The result of the tool execution as a string
        """
        logger.debug(f'Base tool execution with input: {input_str}')
        return 'This is a base tool. Override _run method in your subclass.'
    
    def _ui(self):
        """
        Render custom UI controls for the tool.
        This method can be overridden by subclasses to provide
        custom configuration options in the UI.
        """
        # Default implementation doesn't add any UI controls
        pass
    
    def run(self, input_str):
        """
        Public method to execute the tool.
        Handles logging and error management.
        
        Args:
            input_str: The input string to process
            
        Returns:
            The result of the tool execution
        """
        try:
            logger.info(f'Executing tool: {self.name} with input: {input_str}')
            result = self._run(input_str)
            logger.info(f'Tool {self.name} execution complete')
            return result
        except Exception as e:
            error_msg = f"Error executing tool {self.name}: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def get_metadata(self):
        """
        Get metadata about the tool.
        Useful for tool discovery and documentation.
        
        Returns:
            Dictionary with tool metadata
        """
        return {
            'name': self.name,
            'title': self.title,
            'description': self.description,
            'icon': self.icon,
            'link': self.link,
            'category': self.category,
            'version': self.version,
            'author': self.author,
            'requires': self.requires
        }
