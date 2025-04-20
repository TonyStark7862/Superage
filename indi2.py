# -*- coding: utf-8 -*-
# Use standard encoding declaration for safety, although content uses ASCII
import streamlit as st
import os
import inspect
import ast
import csv
from datetime import datetime
import pandas as pd # Keep pandas for now, might be used by custom tools
from typing import List, Dict, Union, Callable, Any
import importlib.util
import sys
from PIL import Image

# --- Try to import streamlit_option_menu ---
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False
    # Don't display warning automatically, handle it in menusetup
# --- End import attempt ---

# Config Constants
# Use __file__ to get the directory of the current script
# Fallback to current working directory if __file__ is not defined (e.g., in some execution contexts)
try:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

MODELS = ['abc_model'] # Your custom model (Placeholder)
MAX_ITERATIONS = 20 # Keep for potential future use
APP_NAME = "Omega Agent" # Renamed App

# =============================================
# STORAGE MANAGEMENT (Enhanced for Star Feature)
# =============================================

class CSVStorage:
    def __init__(self, csv_path=None, sessions_csv_path=None):
        self.csv_path = csv_path or os.path.join(BASE_DIR, "omega_chat_history.csv")
        self.sessions_path = sessions_csv_path or os.path.join(BASE_DIR, "omega_sessions.csv")
        self._initialize_csv_files()

    def _initialize_csv_files(self):
        # Create chat history CSV if doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Added session_name to history for potential future use, though primarily managed in sessions.csv
                writer.writerow(["session_id", "timestamp", "role", "content", "session_name"])

        # Create sessions CSV if doesn't exist - now includes 'starred'
        if not os.path.exists(self.sessions_path):
            with open(self.sessions_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["session_id", "session_name", "starred"]) # Added 'starred' column

    def save_chat_message(self, session_id, role, content, session_name=""):
        # Get the latest session name from sessions file if not provided
        if not session_name:
             all_sessions_data = self.get_all_sessions_data()
             session_name = all_sessions_data.get(session_id, {}).get("name", session_id) # Fallback to id

        try:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                writer.writerow([session_id, timestamp, role, content, session_name])
        except IOError as e:
            st.error(f"Error saving chat message: {e}")


    def get_chat_history(self, session_id):
        if not os.path.exists(self.csv_path):
            return []

        messages = []
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Check if session_id matches, handle potential read errors
                    if row.get("session_id") == session_id:
                        role = row.get("role", "unknown")
                        content = row.get("content", "")
                        messages.append({"role": role, "content": content})
        except FileNotFoundError:
             st.error(f"Chat history file not found: {self.csv_path}")
             return []
        except Exception as e:
            st.error(f"Error reading chat history: {e}")
            return [] # Return empty list on error
        return messages

    def get_all_sessions_data(self):
        """Returns a dict mapping session_id to {'name': session_name, 'starred': bool}."""
        if not os.path.exists(self.sessions_path):
            return {}

        session_data = {}
        try:
            with open(self.sessions_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    session_id = row.get("session_id")
                    session_name = row.get("session_name", session_id) # Default name to ID
                    # Read starred status, default to False if missing or invalid
                    starred_str = row.get("starred", "0")
                    starred = starred_str == "1" # Treat "1" as True, anything else as False
                    if session_id:
                        session_data[session_id] = {"name": session_name, "starred": starred}
        except FileNotFoundError:
             st.error(f"Sessions file not found: {self.sessions_path}")
             return {}
        except Exception as e:
            st.error(f"Error reading sessions data: {e}")
            return {} # Return empty dict on error
        return session_data

    def update_session_metadata(self, session_id, session_name=None, starred=None):
        """Updates name and/or starred status for a session_id."""
        all_data = self.get_all_sessions_data()

        if session_id not in all_data:
             # Create new entry if it doesn't exist
             all_data[session_id] = {"name": session_name or session_id, "starred": starred or False}
        else:
            # Update existing entry
            if session_name is not None:
                all_data[session_id]["name"] = session_name
            if starred is not None:
                all_data[session_id]["starred"] = starred

        # Write back all sessions
        try:
            with open(self.sessions_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["session_id", "session_name", "starred"])
                for s_id, data in all_data.items():
                    starred_val = "1" if data.get("starred", False) else "0"
                    writer.writerow([s_id, data.get("name", s_id), starred_val])
        except IOError as e:
            st.error(f"Error updating session metadata: {e}")


    def delete_session(self, session_id):
        # === Delete from Chat History ===
        all_messages = []
        try:
            if os.path.exists(self.csv_path):
                with open(self.csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    # Keep messages that are NOT from the session_id being deleted
                    all_messages = [row for row in reader if row.get("session_id") != session_id]

                # Write back the filtered messages
                with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                    # Ensure fieldnames match the original header, handle potential missing keys gracefully
                    fieldnames = ["session_id", "timestamp", "role", "content", "session_name"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    writer.writeheader()
                    writer.writerows(all_messages)
        except IOError as e:
            st.error(f"Error modifying chat history during delete: {e}")
        except Exception as e:
            st.error(f"Unexpected error during chat history delete: {e}")


        # === Delete from Sessions Metadata ===
        session_data = self.get_all_sessions_data()
        if session_id in session_data:
            del session_data[session_id]

            try:
                # Write back remaining sessions
                with open(self.sessions_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["session_id", "session_name", "starred"])
                    for s_id, data in session_data.items():
                        starred_val = "1" if data.get("starred", False) else "0"
                        writer.writerow([s_id, data.get("name", s_id), starred_val])
            except IOError as e:
                 st.error(f"Error updating sessions file during delete: {e}")

# =============================================
# DOCUMENT MANAGEMENT (Kept class, UI removed)
# =============================================

class DocumentManager:
    def __init__(self):
        self.documents = {}
        self.database = None # Placeholder for future vector db integration

    def add_document(self, name, content):
        self.documents[name] = content

    def list_documents(self):
        return list(self.documents.keys())

    def remove_document(self, name):
        if name in self.documents:
            del self.documents[name]

    def similarity_search(self, query, k=1):
        """Placeholder similarity search - returns first doc if any"""
        # In a real implementation, this would use embeddings
        if not self.documents:
            return []
        # Just return the first document found as a simple placeholder
        doc_name = list(self.documents.keys())[0]
        content = self.documents[doc_name]
        # Limit results to k (though currently only returns 0 or 1)
        return [SimpleDocument(f"Document: {doc_name}", content)][:k]


class SimpleDocument:
    def __init__(self, name, page_content):
        self.name = name
        self.page_content = page_content


# =============================================
# TOOLS SYSTEM (Minor text changes)
# =============================================

class BaseTool:
    name = 'Base_tool'
    link = 'https://github.com/example/omega-agent' # Updated link placeholder
    icon = '[T]' # Using text as icon fallback
    title = 'Base Tool'
    description = 'Core tool functionality blueprint.' # Slightly enhanced description

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self, input_data):
        """Execute the tool with the given input"""
        return self._run(input_data)

    def _run(self, input_data):
        """This function should be overwritten when creating a tool"""
        print(f'Running base tool with input: {input_data}')
        return f'Base tool executed with: {input_data}' # Slightly modified return

    def _ui(self):
        """Overwrite this function to add options to the tool UI"""
        pass # No UI by default


class CalculatorTool(BaseTool):
    name = 'calculator'
    icon = '[Calc]' # Text fallback
    title = 'Computational Engine' # Enhanced title
    description = 'Performs advanced arithmetic and mathematical computations. Input format: standard mathematical expression (e.g., "2 + 2", "sin(pi/2)", "sqrt(16)").' # Enhanced description

    def _run(self, expression):
        try:
            # Use eval with a restricted environment for safety
            # Consider safer alternatives like 'asteval' or 'numexpr' in production
            safe_dict = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'len': len, 'pow': pow, 'int': int, 'float': float,
                'str': str, 'sorted': sorted, 'list': list, 'dict': dict,
                'set': set, 'tuple': tuple, 'range': range,
                # Common math constants
                'pi': 3.141592653589793, 'e': 2.718281828459045
            }

            import math
            # Import safe math functions explicitly
            safe_math_functions = ['acos', 'asin', 'atan', 'atan2', 'ceil', 'cos',
                                 'cosh', 'degrees', 'exp', 'fabs', 'floor', 'fmod',
                                 'frexp', 'hypot', 'ldexp', 'log', 'log10', 'modf',
                                 'pow', 'radians', 'sin', 'sinh', 'sqrt', 'tan', 'tanh']
            for name in safe_math_functions:
                 if hasattr(math, name):
                    safe_dict[name] = getattr(math, name)

            # Evaluate using the restricted environment
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            # Use f-string formatting for clarity
            return f"Calculation Result: {result}"
        except Exception as e:
            # Provide a more informative error message
            return f"Error during calculation: {str(e)}. Please check expression syntax and allowed functions."

# --- Helper functions for tool loading --- (Mostly unchanged, checked for UTF-8 safety)
def has_required_attributes(cls):
    """Check that class possesses a name and description attribute"""
    required_attributes = ['name', 'description']
    try:
        for attr in required_attributes:
            if not hasattr(cls, attr):
                return False
        return True
    except Exception:
        return False


def has_docstring(function_node):
    """Check if the provided function node from AST has a docstring."""
    # Ensure body is not empty and first element is an expression with a string value
    return (len(function_node.body) > 0 and
            isinstance(function_node.body[0], ast.Expr) and
            isinstance(function_node.body[0].value, ast.Str))


def evaluate_function_string(func_str):
    """
    Evaluates the provided function string to check:
    1. If it parses and compiles without errors
    2. If the function defined in the string has a docstring
    3. Extracts the function name

    Returns a tuple (runs_without_error: bool_or_exception, has_doc: bool, toolname: str or None)
    """
    try:
        # Use ast.parse for safe parsing
        parsed_ast = ast.parse(func_str)

        # Find the first FunctionDef node
        function_node = next((node for node in parsed_ast.body if isinstance(node, ast.FunctionDef)), None)

        if function_node is None:
            # No function definition found in the string
            return "No function definition found.", False, None

        # Extract tool name
        tool_name = function_node.name

        # Check for docstring using the AST node
        doc_exist = has_docstring(function_node)

        # Try compiling to check for syntax errors not caught by parse
        # Use compile built-in for safer check than exec
        compile(func_str, '<string>', 'exec')

        # If compile succeeds, assume it "runs" syntactically
        return True, doc_exist, tool_name

    except SyntaxError as e:
        return f"Syntax Error: {e}", False, None
    except Exception as e:
        # Catch other potential errors during parsing/compiling
        return f"Error evaluating function: {e}", False, None

def get_class_func_from_module(module):
    """Filter the classes and functions found inside a given module"""
    members = inspect.getmembers(module)
    functions = []
    classes = []
    for name, member in members:
         # Check if it's a function defined in this module
        if inspect.isfunction(member) and member.__module__ == module.__name__:
            # Ensure it has a docstring to be considered a tool
            if member.__doc__:
                functions.append((name, member))
        # Check if it's a class defined in this module
        elif inspect.isclass(member) and member.__module__ == module.__name__:
            classes.append((name, member))

    return classes, functions

def import_from_file(file_path, module_name=None):
    """Import a module from a file path safely."""
    if module_name is None:
        # Create a safe module name from the filename
        module_name = os.path.basename(file_path).replace(".py", "")
        module_name = f"custom_tool_{module_name}" # Add prefix for safety

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            print(f"Warning: Could not create module spec for {file_path}")
            return None

        module = importlib.util.module_from_spec(spec)
        # Important: Add module to sys.modules *before* execution
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except FileNotFoundError:
        print(f"Error: File not found during import: {file_path}")
        return None
    except ImportError as e:
        print(f"Error importing module {module_name} from {file_path}: {e}")
        return None
    except Exception as e:
        # Catch other potential errors during module execution
        print(f"Unexpected error loading module {module_name} from {file_path}: {e}")
        # Attempt to remove potentially broken module from sys.modules
        if module_name in sys.modules:
            del sys.modules[module_name]
        return None


def monitor_folder(folder_path):
    """Monitor a folder path and return the list of successfully imported modules from .py files inside."""
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
        except OSError as e:
             st.error(f"Error creating tools directory {folder_path}: {e}")
             return [] # Cannot proceed if directory creation fails

    # Make sure folder is in path for relative imports within tools (if any)
    if folder_path not in sys.path:
        sys.path.append(folder_path)

    monitored_modules = []
    try:
        # List all .py files in the directory (excluding __init__.py)
        python_files = [f for f in os.listdir(folder_path)
                        if f.endswith('.py') and f != "__init__.py"]

        # Dynamically import all modules
        for py_file in python_files:
            file_path = os.path.join(folder_path, py_file)
            module = import_from_file(file_path)
            if module:
                monitored_modules.append(module)

    except FileNotFoundError:
         st.error(f"Tools directory not found during monitoring: {folder_path}")
         return []
    except Exception as e:
        st.error(f"Error listing or processing files in {folder_path}: {e}")
        return []

    return monitored_modules


class ToolManager:
    def __init__(self):
        self.structured_tools = self._discover_tools() # Renamed internal method
        self.tools_description = self._build_tools_description() # Renamed internal method

    def _build_tools_description(self):
        """Build the description dictionary from loaded tools."""
        return {tool.name: tool.description for tool in self.structured_tools}

    def get_tools(self) -> List[BaseTool]:
        """Returns the list of available tool instances."""
        return self.structured_tools

    def get_tool_names(self) -> List[str]:
        """Returns a list of names of the available tools."""
        return [tool.name for tool in self.structured_tools]

    def get_selected_tools(self, selected_tool_names: List[str]) -> List[BaseTool]:
        """Filters the available tools based on a list of selected names."""
        return [tool for tool in self.structured_tools if tool.name in selected_tool_names]

    def _discover_tools(self) -> List[BaseTool]:
        """Build the list of available tools from built-in and custom sources."""
        # Built-in tools
        discovered_tools = [CalculatorTool()] # Add other built-in tools here if needed

        # Define tools directory path
        tools_dir = os.path.join(BASE_DIR, "omega_tools") # Renamed folder
        custom_tools_dir = os.path.join(tools_dir, "custom_tools")

        # Ensure directories exist
        os.makedirs(tools_dir, exist_ok=True)
        os.makedirs(custom_tools_dir, exist_ok=True)

        # Monitor custom tools directory
        monitored_modules = monitor_folder(custom_tools_dir)

        # Process discovered tool modules
        for module in monitored_modules:
            try:
                classes, functions = get_class_func_from_module(module)

                # Add class-based tools (must inherit BaseTool and have required attributes)
                for _, cls in classes:
                    if issubclass(cls, BaseTool) and cls is not BaseTool and has_required_attributes(cls):
                        try:
                           # Instantiate the tool class
                           discovered_tools.append(cls())
                        except Exception as e:
                           print(f"Error instantiating tool class {cls.__name__} from {module.__name__}: {e}")


                # Add function-based tools (must be callable and have a docstring)
                for name, func in functions:
                     # Check again for callable and docstring for safety
                     if callable(func) and func.__doc__:
                        # Dynamically create a Tool class inheriting from BaseTool
                        # Ensure unique class name using module path if possible
                        dynamic_class_name = f"{module.__name__.replace('.', '_')}_{name}_Tool"
                        try:
                            func_tool_cls = type(
                                dynamic_class_name,
                                (BaseTool,),
                                {
                                    "name": name,
                                    "description": func.__doc__.strip(), # Use stripped docstring
                                    "icon": "[F]", # Default icon for function tools
                                    "title": name.replace("_", " ").title(), # Auto-generate title
                                    # Lambda wraps the function call; use a static method context if needed
                                    "_run": staticmethod(lambda input_data, fn=func: str(fn(input_data)))
                                }
                            )
                            discovered_tools.append(func_tool_cls())
                        except Exception as e:
                             print(f"Error creating dynamic tool class for function {name} from {module.__name__}: {e}")


            except Exception as e:
                print(f"Error processing module {module.__name__}: {e}")

        # Ensure unique tool names (preferring built-in or earlier loaded ones)
        final_tools = {}
        for tool in discovered_tools:
             if tool.name not in final_tools:
                 final_tools[tool.name] = tool
             else:
                 print(f"Warning: Duplicate tool name '{tool.name}' found. Keeping the first one loaded.")

        return list(final_tools.values())


# =============================================
# AGENT SYSTEM
# =============================================

class AgentMemory:
    def __init__(self):
        # Store history as list of dicts: {'role': 'user'/'assistant', 'content': message}
        self.chat_memory: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
         """Adds a message to the memory."""
         # Basic validation
         if role not in ["user", "assistant"]:
             print(f"Warning: Invalid role '{role}' provided to AgentMemory.")
             role = "unknown" # Or handle as an error
         self.chat_memory.append({"role": role, "content": content})

    def get_history(self) -> List[Dict[str, str]]:
        """Returns the current chat history."""
        return self.chat_memory

    def clear(self):
        """Clears the chat memory."""
        self.chat_memory = []


class SimpleAgent:
    def __init__(self, model: str, tools: List[BaseTool], memory: AgentMemory):
        self.model = model # Name of the model (e.g., 'abc_model')
        self.tools = {tool.name: tool for tool in tools} # Store tools in a dict for quick lookup
        self.memory = memory
        self.max_iterations = MAX_ITERATIONS # Max tool use iterations per turn (if needed)

    def run(self, input_text: str, callbacks=None) -> str:
        """Processes user input, potentially using tools, and returns the agent's response."""
        # Add user message to memory *before* processing
        self.memory.add_message("user", input_text)

        # Build the initial prompt using current memory
        prompt = self._build_prompt()

        # === Placeholder LLM Call ===
        # Replace this section with your actual call to the 'abc_model' API or library
        # The model needs to understand the prompt format, especially the tool usage instructions.
        llm_response = self._call_abc_model_placeholder(prompt)
        # === End Placeholder LLM Call ===


        # --- Tool Use Logic (Simple Parsing Example) ---
        # This part assumes the LLM response indicates tool use in a specific format.
        # You might need a more robust parsing method depending on your LLM's capabilities.
        tool_call_prefix = "Action: Use Tool: "
        if llm_response.strip().startswith(tool_call_prefix):
             # Extract tool name and input (example assumes format: "Action: Use Tool: [tool_name]([input])")
             action_part = llm_response.strip()[len(tool_call_prefix):]
             tool_name, tool_input = self._parse_tool_call(action_part) # Implement parsing logic

             if tool_name and tool_name in self.tools:
                 st.info(f"Omega Agent: Using tool '{tool_name}' with input '{tool_input}'...")
                 try:
                     tool_result = self.tools[tool_name].run(tool_input)
                     st.success(f"Tool '{tool_name}' result: {tool_result}")

                     # Add the tool result as context and get a final response
                     # We might want to add the *result* to memory, not just the initial LLM thought.
                     # This example adds a system message about the tool use.
                     tool_context_message = f"Observation: Tool '{tool_name}' executed with input '{tool_input}'. Result: {tool_result}"
                     # Option 1: Add to memory (if you want the LLM to remember the tool use)
                     # self.memory.add_message("system", tool_context_message) # Role 'system' might need handling

                     # Option 2: Build a new prompt with the result for a final generation step
                     final_prompt = self._build_prompt(additional_context=tool_context_message)

                     # === Placeholder LLM Call (Final Response) ===
                     final_response = self._call_abc_model_placeholder(final_prompt)
                     # === End Placeholder LLM Call ===
                     llm_response = final_response # Use the response generated *after* the tool call

                 except Exception as e:
                     st.error(f"Error executing tool '{tool_name}': {e}")
                     # Inform the LLM about the error
                     error_message = f"Observation: Error executing tool '{tool_name}': {str(e)}"
                     error_prompt = self._build_prompt(additional_context=error_message)
                     # === Placeholder LLM Call (Error Response) ===
                     llm_response = self._call_abc_model_placeholder(error_prompt)
                     # === End Placeholder LLM Call ===
             else:
                  # Handle case where tool name is invalid or not found
                  llm_response = f"Omega Agent Error: Could not find or parse tool '{tool_name}'."

        # Add final AI response to memory
        self.memory.add_message("assistant", llm_response)
        return llm_response

    def _parse_tool_call(self, action_string: str) -> (Union[str, None], Union[str, None]):
        """ Parses 'tool_name(input_string)' format. Basic example. """
        try:
             # Find first parenthesis
             open_paren = action_string.find('(')
             # Find last parenthesis
             close_paren = action_string.rfind(')')
             if 0 < open_paren < close_paren:
                 tool_name = action_string[:open_paren].strip()
                 tool_input = action_string[open_paren + 1:close_paren].strip()
                 # Basic input cleaning (remove quotes if they surround the whole input)
                 if len(tool_input) >= 2 and tool_input.startswith('"') and tool_input.endswith('"'):
                     tool_input = tool_input[1:-1]
                 elif len(tool_input) >= 2 and tool_input.startswith("'") and tool_input.endswith("'"):
                      tool_input = tool_input[1:-1]
                 return tool_name, tool_input
        except Exception as e:
             print(f"Error parsing tool call '{action_string}': {e}")
        return None, None # Return None if parsing fails

    def _build_prompt(self, additional_context: str = None) -> str:
        """Builds the full prompt for the LLM including history, tools, and instructions."""
        # System Message / Persona
        prompt = f"""You are {APP_NAME}, a sophisticated, multi-network AI agent.
You have access to a suite of modular cognitive enhancers (tools) to augment your capabilities.
Your goal is to provide accurate, efficient, and insightful responses.
Analyze the user's request and the conversation history.
If a task requires external computation, data retrieval, or specific actions, leverage the appropriate tool.
To use a tool, respond *only* with the following format on a single line:
Action: Use Tool: [tool_name]([input_string])

Available Tools:
"""
        # Format available tools
        if self.tools:
             for tool_name, tool_instance in self.tools.items():
                 prompt += f"- {tool_name}: {tool_instance.description}\n"
        else:
             prompt += "- No tools currently available.\n"

        prompt += "\nConversation History:\n"
        # Format chat history
        for msg in self.memory.get_history():
             # Simple formatting, adjust as needed for your model
             prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"

        # Add any immediate additional context (e.g., tool results or errors)
        if additional_context:
             prompt += f"System: {additional_context}\n" # Use a 'System' role or similar

        # Add the prompt for the Assistant's turn
        prompt += "Assistant:" # Ready for the AI's response

        return prompt

    def _call_abc_model_placeholder(self, prompt: str) -> str:
        """
        *** Placeholder Function ***
        Replace this with the actual call to your 'abc_model'.
        This function should send the 'prompt' string to your model
        and return the model's generated text response.
        """
        print("-" * 20 + " Prompt Sent to LLM (Placeholder) " + "-" * 20)
        print(prompt)
        print("-" * 60)

        # --- Simulated LLM Responses for Testing ---
        if "Action: Use Tool:" in prompt: # If responding to a tool result
             return "Omega Agent: Based on the tool's output, the final answer is..."
        elif "calculator" in self.tools and ("calculate" in prompt.lower() or any(c in prompt for c in "+-*/^")):
             # Simulate LLM deciding to use the calculator
             # Find a simple expression to simulate calculation
             parts = prompt.split("User:")[-1].split() # Get last user message words
             expr = ""
             for part in parts:
                 if part.isdigit() or part in ['+','-','*','/','(',')']:
                      expr += part
             if not expr: expr = "1+1" # Default fallback if no expression found
             return f"Action: Use Tool: calculator({expr})" # Simulate tool call format
        else:
            # Default response if no tool use is simulated
            return f"Omega Agent: This is a placeholder response from the {self.model}. Processing request based on available information."
        # --- End Simulated Responses ---


# =============================================
# UI COMPONENTS
# =============================================

def sidebar():
    with st.sidebar:
        st.markdown(f"### {APP_NAME} Sessions")
        st.markdown("---") # Separator

        if st.button(" [+] Start New Session", use_container_width=True):
            # Generate a unique session ID based on timestamp
            new_session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            st.session_state.session_id = new_session_id
            # Initialize session metadata (name defaults to ID, not starred)
            st.session_state.storage.update_session_metadata(new_session_id, session_name=f"Session {new_session_id[:10]}", starred=False)
            # Clear agent memory for the new session
            if "agent_instance" in st.session_state:
                 st.session_state.agent_instance.memory.clear()
            st.session_state.messages = [] # Clear displayed messages
            # Set flag to trigger rerun after state updates are processed
            st.session_state.trigger_rerun = True


        st.markdown("---")
        st.markdown("**Active Sessions**")

        # Get all session data {session_id: {"name": name, "starred": bool}}
        try:
             st.session_state.all_sessions_data = st.session_state.storage.get_all_sessions_data()
        except Exception as e:
             st.error(f"Failed to load session data: {e}")
             st.session_state.all_sessions_data = {}

        session_ids = list(st.session_state.all_sessions_data.keys())

        # Sort sessions: Starred first, then by timestamp (implicit in ID format)
        session_ids.sort(key=lambda s_id: (
            not st.session_state.all_sessions_data.get(s_id, {}).get("starred", False), # Starred first (False comes before True)
            s_id # Then sort by ID (timestamp) descending implicitly if needed
        ), reverse=True) # Show newest first within starred/unstarred


        if not session_ids:
            st.caption("No active sessions yet.")
        else:
             # Initialize editing state if it doesn't exist
             if 'editing_session_id' not in st.session_state:
                 st.session_state.editing_session_id = None

             for s_id in session_ids:
                 session_info = st.session_state.all_sessions_data.get(s_id, {})
                 session_name = session_info.get("name", s_id) # Default to ID if name missing
                 is_starred = session_info.get("starred", False)
                 is_current_session = (st.session_state.session_id == s_id)

                 # Use columns for layout: Name | Rename Trigger | Star | Delete
                 col1, col2, col3, col4 = st.columns([6, 2, 1, 1]) # Adjust ratios as needed

                 with col1:
                     # Session Name / Load Button / Edit Input
                     if st.session_state.editing_session_id == s_id:
                         # --- Edit Mode ---
                         new_name = st.text_input(
                             f"Rename_{s_id}",
                             value=session_name,
                             label_visibility="collapsed",
                             key=f"edit_input_{s_id}"
                         )
                         # Check if name changed (on_change doesn't trigger reliably for Enter key)
                         if new_name != session_name:
                             st.session_state.storage.update_session_metadata(s_id, session_name=new_name)
                             st.session_state.editing_session_id = None # Exit edit mode
                             st.session_state.trigger_rerun = True # Rerun to update display
                         # Add a small button to save/confirm, helps with triggering update
                         if st.button("Save", key=f"save_rename_{s_id}", use_container_width=True):
                              st.session_state.storage.update_session_metadata(s_id, session_name=new_name)
                              st.session_state.editing_session_id = None # Exit edit mode
                              st.session_state.trigger_rerun = True # Rerun to update display

                     else:
                         # --- Display Mode ---
                         button_type = "primary" if is_current_session else "secondary"
                         # Display name, make it a button to load the session
                         if st.button(f"{session_name}", key=f"load_{s_id}", use_container_width=True, type=button_type):
                             st.session_state.session_id = s_id
                             st.session_state.editing_session_id = None # Ensure exit edit mode on load
                             # Load history into memory when switching session
                             st.session_state.messages = st.session_state.storage.get_chat_history(s_id)
                             st.session_state.memory.clear()
                             for msg in st.session_state.messages:
                                 st.session_state.memory.add_message(msg["role"], msg["content"])
                             st.session_state.trigger_rerun = True # Rerun to update main page

                 with col2:
                      # Rename Trigger (only show if not editing this session)
                      if st.session_state.editing_session_id != s_id:
                          if st.button("(rename)", key=f"rename_trigger_{s_id}", help="Edit session name"):
                              st.session_state.editing_session_id = s_id # Enter edit mode for this session
                              st.session_state.trigger_rerun = True # Rerun to show the input field

                 with col3:
                     # Star/Unstar Button
                     star_text = "[Unstar]" if is_starred else "[Star]"
                     star_help = "Unmark as favorite" if is_starred else "Mark as favorite"
                     if st.button(star_text, key=f"star_{s_id}", help=star_help):
                         st.session_state.storage.update_session_metadata(s_id, starred=(not is_starred))
                         st.session_state.editing_session_id = None # Ensure exit edit mode
                         st.session_state.trigger_rerun = True # Rerun to update sort order/display

                 with col4:
                     # Delete Button
                     if st.button("[Del]", key=f"delete_{s_id}", help="Delete session"):
                         st.session_state.storage.delete_session(s_id)
                         st.session_state.editing_session_id = None # Ensure exit edit mode
                         # If deleting the current session, switch to a default or new one
                         if st.session_state.session_id == s_id:
                             # Try to find another session, otherwise create a new one
                             remaining_sessions = list(st.session_state.storage.get_all_sessions_data().keys())
                             if remaining_sessions:
                                 st.session_state.session_id = remaining_sessions[0]
                             else:
                                  # Create a fresh session if none are left
                                  new_session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                                  st.session_state.session_id = new_session_id
                                  st.session_state.storage.update_session_metadata(new_session_id, session_name=f"Session {new_session_id[:10]}", starred=False)
                         st.session_state.trigger_rerun = True # Rerun to update list

def chat_page():
    # Header
    st.markdown("### 💭 Conversation Interface")
    st.markdown("---")

    # Show selected tools dynamically
    selected_tool_names = [tool.name for tool in st.session_state.tools]
    if selected_tool_names:
        s = ', '.join(selected_tool_names)
        st.info(f'Active Cognitive Enhancers (Tools): {s}')
    else:
        st.info("No Cognitive Enhancers (Tools) currently selected.")


    # Get current session name for saving messages
    current_session_data = st.session_state.storage.get_all_sessions_data()
    current_session_name = current_session_data.get(st.session_state.session_id, {}).get("name", st.session_state.session_id)


    # Initialize agent if not already done for the current session/toolset
    # We might need to re-initialize if tools change, but keep memory separate
    if "agent_instance" not in st.session_state or \
       st.session_state.agent_instance.tools != {t.name: t for t in st.session_state.tools} or \
       st.session_state.agent_instance.model != st.session_state.model: # Re-init if model changes too
        st.session_state.agent_instance = SimpleAgent(
            st.session_state.model,
            st.session_state.tools,
            st.session_state.memory # Crucially, reuse the memory object
        )
        print(f"Agent re-initialized for session {st.session_state.session_id}") # Debug print

    # Load and display messages from memory (already loaded on session switch)
    st.session_state.messages = st.session_state.memory.get_history()
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
             st.write(msg["content"]) # Use write for better markdown/code rendering

    # Input area
    if prompt := st.chat_input(f"Interact with {APP_NAME}..."):
        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)

        # Save user message (memory is handled by agent.run)
        st.session_state.storage.save_chat_message(st.session_state.session_id, "user", prompt, current_session_name)

        # Get response from agent
        with st.chat_message("assistant"):
            with st.spinner(f"{APP_NAME} is processing..."):
                try:
                    # --- Agent Execution ---
                    # Add prefix/suffix if they exist
                    full_prompt = f"{st.session_state.prefix}{prompt}{st.session_state.suffix}"
                    response = st.session_state.agent_instance.run(full_prompt) # Agent adds user message to memory
                    # --- End Agent Execution ---

                    st.write(response) # Display response

                    # Save assistant response (memory is handled by agent.run)
                    st.session_state.storage.save_chat_message(st.session_state.session_id, "assistant", response, current_session_name)

                except Exception as e:
                     st.error(f"An error occurred during agent execution: {e}")
                     # Optionally save error state?
                     # st.session_state.storage.save_chat_message(st.session_state.session_id, "system", f"Error: {e}", current_session_name)

        # No explicit rerun needed here, Streamlit handles chat_input update

def tools_page():
    st.markdown("### 🛠️ Cognitive Enhancers (Tools)")
    st.markdown("---")

    # Tool selection state management (initialize if needed)
    if "selected_tool_names" not in st.session_state:
        st.session_state.selected_tool_names = [tool.name for tool in st.session_state.tools]

    # Tool search/filter
    tool_search = st.text_input('Filter Enhancers', placeholder='Search by name or description', key='tool_search')

    available_tools = st.session_state.tool_manager.get_tools() # Get all tool instances
    filtered_tools = available_tools

    # Filter tools based on search term (case-insensitive)
    if tool_search:
        search_term = tool_search.lower()
        filtered_tools = [
            tool for tool in available_tools
            if search_term in tool.name.lower() or search_term in tool.description.lower()
        ]

    # Display tools as cards
    if not filtered_tools:
        st.warning(f"No tools found matching '{tool_search}'.")
    else:
         cols = st.columns(3) # Adjust number of columns if needed
         current_selection = set(st.session_state.selected_tool_names)

         for i, tool in enumerate(filtered_tools):
             col = cols[i % 3]
             with col:
                 with st.container(border=True):
                     # Checkbox for selection - use tool name as identifier
                     is_selected = st.checkbox(
                         f"{tool.icon} {tool.title}",
                         value=(tool.name in current_selection), # Check against current selection set
                         key=f"tool_select_{tool.name}",
                         help=f"{tool.description}\n(ID: {tool.name})" # Add description to tooltip
                     )
                     # Update selection state based on checkbox interaction
                     if is_selected and tool.name not in current_selection:
                          st.session_state.selected_tool_names.append(tool.name)
                          st.session_state.trigger_rerun = True # Rerun needed to update agent
                     elif not is_selected and tool.name in current_selection:
                          st.session_state.selected_tool_names.remove(tool.name)
                          st.session_state.trigger_rerun = True # Rerun needed to update agent

                     # Display tool description (optional, already in tooltip)
                     # st.caption(f"{tool.description[:60]}...") # Show truncated description

                     # Call the tool's UI method if it exists (for tool-specific settings)
                     if hasattr(tool, '_ui') and callable(tool._ui):
                         tool._ui()


    # Update the main tools list in session state if changes occurred
    st.session_state.tools = st.session_state.tool_manager.get_selected_tools(st.session_state.selected_tool_names)


    # --- Create New Tool Section ---
    st.markdown("---")
    st.markdown("### Integrate New Enhancer Module")

    with st.expander("Define a Custom Tool Function"):
        new_tool_filename = st.text_input("Tool Filename (.py)", placeholder="e.g., weather_lookup")
        new_tool_function = st.text_area(
            "Tool Function Code (Python)",
            height=300,
            placeholder="""# Example:
def my_tool(input_data: str) -> str:
    \"\"\"Provide a clear description of what this tool does.
    Explain the expected input format and the output it returns.
    This description is crucial for the Omega Agent to understand how to use the tool.
    \"\"\"
    # Your tool logic here
    processed_result = input_data.upper() # Replace with actual logic
    return f"Processed: {processed_result}"
""",
            help="Define a Python function with a clear docstring. The function name becomes the tool ID."
        )

        if st.button("Integrate Tool Module"):
            if new_tool_filename and new_tool_function:
                # Basic filename validation
                if not new_tool_filename.endswith('.py'):
                    safe_filename = f"{new_tool_filename.split('.')[0]}.py" # Ensure .py extension
                else:
                    safe_filename = new_tool_filename

                if not safe_filename.replace(".py", "").isidentifier():
                     st.error("Filename must be a valid Python identifier (letters, numbers, underscores, cannot start with a number).")
                else:
                    # Validate function string using the helper
                    validation_result, has_doc, func_name = evaluate_function_string(new_tool_function)

                    if validation_result is True and has_doc is True:
                        # Create the tools directory if it doesn't exist
                        tools_dir = os.path.join(BASE_DIR, "omega_tools", "custom_tools")
                        os.makedirs(tools_dir, exist_ok=True)

                        # Save the tool code to the file
                        tool_path = os.path.join(tools_dir, safe_filename)
                        try:
                            with open(tool_path, "w", encoding='utf-8') as f:
                                f.write(new_tool_function)

                            st.success(f"Tool '{func_name}' integrated successfully as '{safe_filename}'!")
                            # Reinitialize tool manager to pick up the new tool
                            st.session_state.tool_manager = ToolManager()
                            # Add new tool to selection? Optional, maybe user selects it manually.
                            st.session_state.trigger_rerun = True # Rerun to refresh the tools list display

                        except IOError as e:
                            st.error(f"Error saving tool file: {e}")
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {e}")

                    else:
                        # Display validation errors
                        if validation_result is not True:
                            st.error(f"Tool code validation failed: {validation_result}")
                        if not has_doc:
                            st.error("Tool function must have a docstring explaining its usage.")
            else:
                st.error("Please provide both a filename and function code for the tool.")


def info_page():
    st.markdown(f"### ℹ️ About {APP_NAME}")
    st.markdown("---")

    st.markdown(f"""
    ## {APP_NAME}: Synergistic Multi-Network AI Agent Framework

    **Omega Agent** represents a paradigm shift in human-AI collaboration. This advanced framework facilitates seamless interaction with a core language model, dynamically augmented by a configurable suite of **Cognitive Enhancers** (Tools).

    Leveraging a modular architecture, {APP_NAME} enables sophisticated task execution by intelligently delegating specific computational or data-retrieval operations to specialized tool modules.

    ### Core Capabilities:

    * **Adaptive LLM Integration:** Designed for flexible integration with diverse large language models (Current Placeholder: `{st.session_state.model}`).
    * **Dynamic Tool Ecosystem:** Select, manage, and even integrate new Cognitive Enhancers on-the-fly via the intuitive interface. Tools range from computational engines to bespoke function modules.
    * **Persistent Session Context:** Robust session management ensures continuity across interactions, preserving conversational state and agent memory. Sessions can be favorited for quick access.
    * **Contextual Awareness (Future):** Foundational support for incorporating external knowledge bases (Document Management - UI currently streamlined).
    * **Developer Extensibility:** Create and deploy custom Python functions as new Cognitive Enhancers directly through the framework.

    ### Operational Synergy Flow:

    1.  **Enhancer Configuration (Tools Page):** Curate the active set of Cognitive Enhancers required for your objectives. Define and integrate novel capabilities as needed.
    2.  **Interaction  (Chat Page):** Engage with the {APP_NAME}. Initiate new dialogues or resume prior contextual sessions. The agent intelligently invokes configured enhancers based on conversational requirements.
    3.  **Framework Genesis (Info Page):** Review architectural principles and operational guidelines.

    ### Crafting Custom Cognitive Enhancers:

    Extend the agent's capabilities by authoring standard Python functions:

    1.  Utilize the **"Integrate New Enhancer Module"** section within the Tools page.
    2.  Ensure each function includes a comprehensive **docstring**. This metadata is critical for the agent's understanding of the tool's purpose, input parameters, and expected output schema.

    **Example Enhancer Definition:**
    ```python
    # File: stock_price_lookup.py
    import random # Placeholder for actual API call

    def get_stock_price(symbol: str) -> str:
        \"\"\"Retrieves the current stock price for a given ticker symbol.
        Input: A valid stock ticker symbol (e.g., "GOOGL", "MSFT").
        Output: A string indicating the current price or an error message.
        \"\"\"
        try:
            # In a real scenario, call a financial API here
            price = round(random.uniform(100, 500), 2) # Simulated price
            return f"The current price for {symbol} is ${price}."
        except Exception as e:
            return f"Error retrieving price for {symbol}: {e}"
    ```

    **{APP_NAME}**: Augmenting Intelligence, Amplifying Potential.
    """)

# =============================================
# MAIN APPLICATION SETUP
# =============================================

def menusetup():
    """Set up the main navigation menu."""
    # Updated list without Settings
    list_menu = ["Chat", "Tools", "Info"]
    # Corresponding page functions
    list_pages = [chat_page, tools_page, info_page]
    st.session_state.dictpages = dict(zip(list_menu, list_pages))
    # Icons (using streamlit-option-menu syntax if available)
    list_icons = ['chat-left-text-fill', 'tools', 'info-circle-fill'] # Adjusted icons

    # Determine current page index
    try:
        default_index = list_menu.index(st.session_state.selected_page)
    except ValueError:
        default_index = 0 # Default to first page if current selection is invalid

    # Use streamlit-option-menu if available
    if HAS_OPTION_MENU:
        # Use on_change callback for immediate navigation
        def menu_callback():
            # Update selected_page based on the key from option_menu
            st.session_state.selected_page = st.session_state.app_menu_key
            # Reset editing state when changing tabs
            st.session_state.editing_session_id = None


        selected_title = option_menu(
            None, # No title for the menu itself
            list_menu,
            icons=list_icons,
            menu_icon="cast", # Main menu icon (optional)
            default_index=default_index,
            orientation="horizontal",
            key='app_menu_key', # Use a key to access the selection
            on_change=menu_callback # Use callback for navigation
        )
        # The selection is now handled by the callback, no further action needed here

    else:
        # Fallback to buttons if streamlit-option-menu is not installed
        st.warning("Using fallback buttons for navigation. Install 'streamlit-option-menu' for a better experience.", icon="⚠️")
        cols = st.columns(len(list_menu))
        for i, page_name in enumerate(list_menu):
            with cols[i]:
                # Use primary button style if it's the selected page
                button_type = "primary" if page_name == st.session_state.selected_page else "secondary"
                if st.button(page_name, use_container_width=True, type=button_type):
                    st.session_state.selected_page = page_name
                    # Reset editing state when changing tabs
                    st.session_state.editing_session_id = None
                    st.rerun() # Rerun required for button navigation


def pageselection():
    """Call the appropriate page function based on selection."""
    page_func = st.session_state.dictpages.get(st.session_state.selected_page)
    if page_func:
        page_func()
    else:
        st.error(f"Page '{st.session_state.selected_page}' not found. Defaulting to Chat.")
        st.session_state.selected_page = "Chat"
        chat_page()


def ensure_session_state():
    """Initialize session state variables if they don't exist."""

    # Trigger for explicit reruns after state changes
    if "trigger_rerun" not in st.session_state:
        st.session_state.trigger_rerun = False

    # Central Storage Manager
    if "storage" not in st.session_state:
        st.session_state.storage = CSVStorage()

    # Session ID and Data
    if "session_id" not in st.session_state:
        # Try to load existing sessions, pick the latest starred or just latest
        all_sessions = st.session_state.storage.get_all_sessions_data()
        if all_sessions:
             session_ids = list(all_sessions.keys())
             session_ids.sort(key=lambda s_id: (
                 not all_sessions.get(s_id, {}).get("starred", False), s_id
             ), reverse=True)
             st.session_state.session_id = session_ids[0]
        else:
            # Create a brand new session if none exist
             new_session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
             st.session_state.session_id = new_session_id
             st.session_state.storage.update_session_metadata(new_session_id, session_name=f"Session {new_session_id[:10]}", starred=False)

    # Load all session data into state for sidebar
    if "all_sessions_data" not in st.session_state:
         st.session_state.all_sessions_data = st.session_state.storage.get_all_sessions_data()

    # Editing state for session rename
    if 'editing_session_id' not in st.session_state:
         st.session_state.editing_session_id = None

    # Model Selection (using placeholder)
    if "model" not in st.session_state:
        st.session_state.model = MODELS[0] # Default to the first model

    # Tool Management
    if "tool_manager" not in st.session_state:
        st.session_state.tool_manager = ToolManager()

    if "initial_tools" not in st.session_state:
        # Default tools to enable initially (e.g., calculator)
        st.session_state.initial_tools = ['calculator'] # Adjust as needed

    if "selected_tool_names" not in st.session_state:
        # Ensure initial tools are valid before setting them
        valid_initial_tools = [
             t for t in st.session_state.initial_tools
             if t in st.session_state.tool_manager.get_tool_names()
        ]
        st.session_state.selected_tool_names = valid_initial_tools

    if "tools" not in st.session_state:
         st.session_state.tools = st.session_state.tool_manager.get_selected_tools(st.session_state.selected_tool_names)

    # Agent Memory (persistent object across reruns for a session)
    if "memory" not in st.session_state:
        st.session_state.memory = AgentMemory()
        # Load history for the initial session_id
        initial_history = st.session_state.storage.get_chat_history(st.session_state.session_id)
        for msg in initial_history:
             st.session_state.memory.add_message(msg["role"], msg["content"])

    # Chat Messages for display (synced with memory)
    if "messages" not in st.session_state:
        st.session_state.messages = st.session_state.memory.get_history()

    # Document Management (keep class instance, UI removed)
    if "doc_manager" not in st.session_state:
        st.session_state.doc_manager = DocumentManager()

    # UI State
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Chat" # Default to Chat page now

    # Prompt Engineering Affixes
    if "prefix" not in st.session_state:
        st.session_state.prefix = ''
    if "suffix" not in st.session_state:
        st.session_state.suffix = ''

    # Agent Instance (initialized/updated in chat_page)
    # if "agent_instance" not in st.session_state: # Initialization moved to chat_page


def main():
    """Main application entry point"""
    # Set page config first
    icon_path = os.path.join(BASE_DIR, "appicon.ico") # Assuming icon is in the same directory
    page_icon = None
    try:
        if os.path.exists(icon_path):
             page_icon = Image.open(icon_path)
    except Exception as e:
         print(f"Warning: Could not load page icon '{icon_path}': {e}")

    st.set_page_config(
        page_title=APP_NAME,
        page_icon=page_icon,
        layout="wide", # Use wide layout
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/example/omega-agent', # Update link
            'Report a bug': "https://github.com/example/omega-agent/issues", # Update link
            'About': f"# {APP_NAME}\nSynergistic Multi-Network AI Agent Framework."
        }
    )

    # Initialize session state variables
    ensure_session_state()

    # Handle explicit reruns requested by callbacks
    if st.session_state.get("trigger_rerun", False):
        st.session_state.trigger_rerun = False # Reset the flag
        st.rerun()


    # Create tools directory structure if it doesn't exist
    tools_dir = os.path.join(BASE_DIR, "omega_tools", "custom_tools")
    os.makedirs(tools_dir, exist_ok=True)

    # --- UI Rendering ---
    st.title(APP_NAME)
    menusetup() # Display the main navigation menu
    sidebar() # Display the sidebar content
    pageselection() # Display the content of the selected page

if __name__ == "__main__":
    main()
