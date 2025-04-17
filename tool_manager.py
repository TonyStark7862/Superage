import os
import importlib
import inspect
from .base_tools import Ui_Tool
from storage.logger_config import logger

class ToolManager:
    """
    Responsible for discovering, registering, and managing tools.
    Makes tools pluggable by dynamically loading them from the tools_list directory.
    """
    def __init__(self):
        self.structured_tools = self._make_tools_list()
        self.tools_ui = {}
        self.tools_description = self._make_tools_description()

    def _make_tools_description(self):
        """Create a mapping of tool names to their descriptions."""
        tools_description = {}
        for t in self.structured_tools:
            tools_description.update({t.name: t.description})
        return tools_description

    def get_tools(self):
        """Get all available tools."""
        return self.structured_tools
    
    def get_tool_names(self):
        """Get names of all available tools."""
        return [tool.name for tool in self.structured_tools]

    def get_selected_tools(self, selected_tool_names):
        """Get tools based on their names."""
        return [tool for tool in self.structured_tools if tool.name in selected_tool_names]
    
    def register_tool(self, tool):
        """Register a new tool at runtime."""
        # Check if the tool is already registered
        if tool.name not in self.get_tool_names():
            self.structured_tools.append(tool)
            self.tools_description.update({tool.name: tool.description})
            return True
        return False
    
    def unregister_tool(self, tool_name):
        """Unregister a tool by name."""
        self.structured_tools = [tool for tool in self.structured_tools if tool.name != tool_name]
        if tool_name in self.tools_description:
            del self.tools_description[tool_name]
        return True

    def _make_tools_list(self):
        """
        Discover and load all tools from the tools_list directory.
        This makes tools pluggable - just add a new tool file and it's automatically loaded.
        """
        base_tool_list = []
        
        # Path to the tools_list directory
        tools_list_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools_list')
        
        # Ensure the directory exists
        if not os.path.exists(tools_list_dir):
            os.makedirs(tools_list_dir)
            logger.info(f"Created tools_list directory at {tools_list_dir}")
        
        # Add tools_list directory to path for importing
        if tools_list_dir not in os.path.sys.path:
            os.path.sys.path.append(tools_list_dir)
        
        # List all .py files in the tools directory (excluding __init__.py)
        python_files = [f[:-3] for f in os.listdir(tools_list_dir) 
                        if f.endswith('.py') and f != "__init__.py"]
        
        # Import each tool module
        for file_name in python_files:
            try:
                # Import the module
                module = importlib.import_module(f"tools_list.{file_name}")
                
                # Get all classes in the module
                for name, obj in inspect.getmembers(module):
                    # Check if it's a class that inherits from Ui_Tool
                    if (inspect.isclass(obj) and 
                        obj != Ui_Tool and 
                        issubclass(obj, Ui_Tool)):
                        try:
                            # Instantiate the tool and add it to the list
                            tool_instance = obj()
                            base_tool_list.append(tool_instance)
                            logger.info(f"Loaded tool: {tool_instance.name}")
                        except Exception as e:
                            logger.error(f"Error instantiating tool {name}: {e}")
                
                # Get all functions in the module
                for name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj) and hasattr(obj, "__doc__") and obj.__doc__:
                        try:
                            # Create a tool from the function
                            tool_instance = self._create_tool_from_function(name, obj)
                            if tool_instance:
                                base_tool_list.append(tool_instance)
                                logger.info(f"Loaded function tool: {name}")
                        except Exception as e:
                            logger.error(f"Error creating tool from function {name}: {e}")
                            
            except Exception as e:
                logger.error(f"Error importing module {file_name}: {e}")
        
        return base_tool_list
    
    def _create_tool_from_function(self, name, func):
        """Create a tool from a function with a docstring."""
        if not func.__doc__:
            return None
            
        # Create a new tool class dynamically
        class FunctionTool(Ui_Tool):
            name = name
            description = func.__doc__.strip()
            icon = "🧰"
            title = name.replace("_", " ").title()
            
            def _run(self, input_str):
                try:
                    return func(input_str)
                except Exception as e:
                    return f"Error executing {name}: {str(e)}"
                    
        # Instantiate and return the tool
        return FunctionTool()
