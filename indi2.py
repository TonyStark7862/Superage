# -*- coding: utf-8 -*-
import streamlit as st
import os
import inspect
import ast
import json
import csv
from datetime import datetime
import pandas as pd
from typing import List, Dict, Union, Callable, Any
import importlib.util
import sys
from PIL import Image
import tempfile
import traceback # Added for better error reporting if needed

# Config Constants
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
# --- Removed MODELS as settings page is removed ---
# MODELS = ['abc_model'] # Our custom model (Assuming agent still uses this)
DEFAULT_MODEL = 'abc_model' # Define the default/only model
MAX_ITERATIONS = 20

# =============================================
# STORAGE MANAGEMENT (Added 'starred' status)
# =============================================

class CSVStorage:
    def __init__(self, csv_path=None):
        if csv_path is None:
            # Store history in a subdirectory for better organization
            data_dir = os.path.join(BASE_DIR, "omega_data")
            os.makedirs(data_dir, exist_ok=True)
            self.csv_path = os.path.join(data_dir, "chat_history.csv")
            self.sessions_path = os.path.join(data_dir, "sessions.csv")
        else:
            # Ensure directory exists if custom path is given
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            self.csv_path = csv_path
            # Assume sessions file is in the same custom directory
            self.sessions_path = os.path.join(os.path.dirname(csv_path), "sessions.csv")

        # Initialize CSV files if they don't exist
        self._initialize_csv_files()

    def _initialize_csv_files(self):
        # Create chat history CSV if doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Added session_name to chat history for easier lookup if needed
                writer.writerow(["session_id", "timestamp", "role", "content", "session_name"])

        # Create sessions CSV if doesn't exist, now includes 'starred'
        if not os.path.exists(self.sessions_path):
            with open(self.sessions_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["session_id", "session_name", "starred"])
        else:
            # Check if existing sessions file has the 'starred' column
            with open(self.sessions_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and "starred" not in header:
                    # Need to add the starred column
                    sessions_data = list(reader)
            if header and "starred" not in header:
                 with open(self.sessions_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(header + ["starred"])
                    for row in sessions_data:
                        writer.writerow(row + [0]) # Default starred to 0 (False)


    def save_chat_message(self, session_id, role, content, session_name=None):
        if session_name is None:
            session_name = self.get_session_details(session_id).get('name', session_id)

        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([session_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), role, content, session_name])

    def get_chat_history(self, session_id):
        if not os.path.exists(self.csv_path):
            return []

        messages = []
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("session_id") == session_id:
                        messages.append({"role": row["role"], "content": row["content"]})
        except FileNotFoundError:
            return []
        except Exception as e:
            st.error(f"Error reading chat history: {e}")
            return []
        return messages

    def get_all_sessions(self):
        """Returns a list of session_ids"""
        if not os.path.exists(self.sessions_path):
            return []

        session_ids = set()
        try:
             with open(self.sessions_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "session_id" in row:
                        session_ids.add(row["session_id"])
        except FileNotFoundError:
            return []
        except Exception as e:
            st.error(f"Error reading sessions list: {e}")
            return []
        return list(session_ids)

    def get_all_session_details(self):
        """Returns a dictionary {session_id: {'name': str, 'starred': bool}}"""
        if not os.path.exists(self.sessions_path):
            return {}

        session_details = {}
        try:
            with open(self.sessions_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "session_id" in row and "session_name" in row:
                        # Handle potential missing 'starred' column gracefully
                        starred_val = row.get("starred", "0")
                        try:
                            is_starred = bool(int(starred_val))
                        except (ValueError, TypeError):
                            is_starred = False # Default to false if conversion fails
                        session_details[row["session_id"]] = {
                            "name": row["session_name"],
                            "starred": is_starred
                        }
        except FileNotFoundError:
             return {}
        except Exception as e:
            st.error(f"Error reading session details: {e}")
            return {}
        return session_details

    def get_session_details(self, session_id):
        """Gets details for a specific session."""
        all_details = self.get_all_session_details()
        return all_details.get(session_id, {"name": session_id, "starred": False})


    def save_session_details(self, session_id, session_name, starred):
        """Saves or updates a session's details."""
        all_details = self.get_all_session_details()
        all_details[session_id] = {"name": session_name, "starred": starred}
        self._write_sessions_file(all_details)

    def toggle_session_star(self, session_id):
        """Toggles the starred status of a session."""
        all_details = self.get_all_session_details()
        if session_id in all_details:
            all_details[session_id]['starred'] = not all_details[session_id]['starred']
            self._write_sessions_file(all_details)
        else:
            # If session exists in chat history but not sessions file, add it
             if session_id in self.get_all_sessions():
                 self.save_session_details(session_id, session_id, True) # Star it by default if adding

    def delete_session(self, session_id):
        # --- Delete from chat history ---
        all_messages = []
        if os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames if reader.fieldnames else ["session_id", "timestamp", "role", "content", "session_name"]
                    all_messages = [row for row in reader if row.get("session_id") != session_id]

                with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_messages)
            except Exception as e:
                 st.error(f"Error updating chat history after delete: {e}")

        # --- Delete from sessions file ---
        all_details = self.get_all_session_details()
        if session_id in all_details:
            del all_details[session_id]
            self._write_sessions_file(all_details)

    def _write_sessions_file(self, session_details_dict):
        """Helper function to write the entire sessions CSV."""
        try:
            with open(self.sessions_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["session_id", "session_name", "starred"])
                for s_id, details in session_details_dict.items():
                    writer.writerow([s_id, details.get("name", s_id), 1 if details.get("starred", False) else 0])
        except Exception as e:
             st.error(f"Error writing sessions file: {e}")


# --- Document Manager Removed as Settings page is removed ---
# class DocumentManager: ...
# class SimpleDocument: ...


# =============================================
# TOOLS SYSTEM (Unchanged, except link/icon in BaseTool)
# =============================================

class BaseTool:
    name = 'Base_tool'
    link = 'https://github.com/your-repo/omega' # Example link update
    icon = '🔧' # Default icon
    description = 'Base tool description'

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self, input_data):
        """Execute the tool with the given input"""
        return self._run(input_data)

    def _run(self, input_data):
        """This function should be overwritten when creating a tool"""
        print(f'Running base tool with input: {input_data}')
        return f'Success: {input_data}'

    def _ui(self):
        """Overwrite this function to add options to the tool UI"""
        pass

class CalculatorTool(BaseTool):
    name = 'calculator'
    icon = '🧮'
    title = 'Calculator'
    description = 'Perform arithmetic calculations. Input format: mathematical expression (e.g., "2 + 2", "sin(30)", "sqrt(16)")'

    def _run(self, expression):
        try:
            # Use eval with a restricted environment for safety
            # In a production app, you would want to use a safer evaluation method
            safe_dict = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'len': len, 'pow': pow, 'int': int, 'float': float,
                'str': str, 'sorted': sorted, 'list': list, 'dict': dict,
                'set': set, 'tuple': tuple, 'range': range
            }

            import math
            for name in dir(math):
                 if not name.startswith("_"): # Avoid private math attributes
                    safe_dict[name] = getattr(math, name)

            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

# --- Tool Helper Functions (Unchanged) ---
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
    return (len(function_node.body) > 0 and
            isinstance(function_node.body[0], ast.Expr) and
            isinstance(function_node.body[0].value, ast.Str))


def evaluate_function_string(func_str):
    """
    Evaluates the provided function string to check:
    1. If it runs without errors
    2. If the function in the string has a docstring

    Returns a tuple (runs_without_error: bool, has_doc: bool, toolname: str)
    """
    try:
        parsed_ast = ast.parse(func_str)

        # Check if the parsed AST contains a FunctionDef (function definition)
        function_node = next((node for node in parsed_ast.body if isinstance(node, ast.FunctionDef)), None)
        if not function_node:
             return False, False, None # Not a valid function definition structure

        # Extract tool name
        tool_name = function_node.name

        # Compiling the function string - use a temp dict to avoid polluting globals
        temp_globals = {}
        compiled_func = compile(func_str, '<string>', 'exec')
        exec(compiled_func, temp_globals)

        # Check for docstring
        doc_exist = has_docstring(function_node)

        # Optional: Basic check if the function is callable after exec
        if tool_name not in temp_globals or not callable(temp_globals[tool_name]):
             raise ValueError(f"Function '{tool_name}' not found or not callable after execution.")

        return True, doc_exist, tool_name

    except Exception as e:
        # Return the exception itself for better debugging in UI
        return e, False, None


def get_class_func_from_module(module):
    """Filter the classes and functions found inside a given module"""
    members = inspect.getmembers(module)
    # Filter and store only functions and classes defined inside module
    functions = []
    classes = []
    for name, member in members:
        try: # Add try-except for safety
            if inspect.isfunction(member) and member.__module__ == module.__name__:
                if member.__doc__: # Only include functions with docstrings
                    functions.append((name, member))

            if inspect.isclass(member) and member.__module__ == module.__name__:
                classes.append((name, member))
        except Exception:
             pass # Ignore members that cause errors during inspection
    return classes, functions


def import_from_file(file_path, module_name=None):
    """Import a module from a file path"""
    if module_name is None:
        module_name = os.path.basename(file_path).replace(".py", "")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None: # Check loader too
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module # Register module before execution
    try:
        spec.loader.exec_module(module)
    except Exception as e:
         print(f"Error executing module {module_name} from {file_path}: {e}")
         # Remove partially loaded module on error
         if module_name in sys.modules:
             del sys.modules[module_name]
         return None
    return module


def monitor_folder(folder_path):
    """Monitor a folder path and return the list of modules from .py files inside"""
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
        except OSError as e:
             print(f"Error creating tools directory {folder_path}: {e}")
             return [] # Return empty list if directory creation fails

    # Make sure folder is in path (important for relative imports within tools)
    if folder_path not in sys.path:
        sys.path.insert(0, folder_path) # Insert at beginning

    # List all .py files in the directory (excluding __init__.py)
    try:
        python_files = [f for f in os.listdir(folder_path)
                        if f.endswith('.py') and f != "__init__.py"]
    except FileNotFoundError:
         print(f"Tools directory not found during scan: {folder_path}")
         return []
    except Exception as e:
         print(f"Error listing files in tools directory {folder_path}: {e}")
         return []

    # Dynamically import all modules
    monitored_modules = []
    for py_file in python_files:
        file_path = os.path.join(folder_path, py_file)
        module_name = py_file[:-3] # Module name is filename without .py
        try:
            # If module already imported, reload it to pick up changes
            if module_name in sys.modules:
                 # Use importlib.reload for safer reloading
                 module = importlib.reload(sys.modules[module_name])
            else:
                 module = import_from_file(file_path, module_name)

            if module:
                monitored_modules.append(module)
        except Exception as e:
            st.warning(f"Error importing/reloading tool '{py_file}': {e}")
            # Clean up sys.modules if import failed
            if module_name in sys.modules:
                 del sys.modules[module_name]

    return monitored_modules


class ToolManager:
    def __init__(self):
        self.structured_tools = self._make_tools_list()
        self.tools_description = self._make_tools_description()

    def _make_tools_description(self):
        tools_description = {}
        for tool in self.structured_tools:
            tools_description[tool.name] = tool.description
        return tools_description

    def get_tools(self):
        return self.structured_tools

    def get_tool_names(self):
        return [tool.name for tool in self.structured_tools]

    def get_selected_tools(self, selected_tool_names):
        return [tool for tool in self.structured_tools if tool.name in selected_tool_names]

    def _make_tools_list(self):
        """Build the list of available tools"""
        base_tools = [CalculatorTool()] # Start with built-in tools

        # Define tools directory structure
        tools_root_dir = os.path.join(BASE_DIR, "omega_tools") # Renamed root folder
        custom_tools_dir = os.path.join(tools_root_dir, "custom_tools")

        # Ensure directories exist
        os.makedirs(tools_root_dir, exist_ok=True)
        os.makedirs(custom_tools_dir, exist_ok=True)

        # Monitor custom tools directory
        monitored_modules = monitor_folder(custom_tools_dir)

        # Process tool modules
        processed_tool_names = {tool.name for tool in base_tools} # Keep track of names to avoid duplicates

        for module in monitored_modules:
            try:
                # Get classes and functions from the module
                classes, functions = get_class_func_from_module(module)

                # Add class-based tools
                for _, cls in classes:
                    # Check inheritance and required attributes more carefully
                    if inspect.isclass(cls) and issubclass(cls, BaseTool) and cls is not BaseTool:
                        if has_required_attributes(cls):
                            try:
                                instance = cls()
                                if instance.name not in processed_tool_names:
                                     base_tools.append(instance)
                                     processed_tool_names.add(instance.name)
                                else:
                                     print(f"Warning: Duplicate tool name '{instance.name}' found in {module.__name__}. Skipping.")
                            except Exception as e:
                                st.warning(f"Error instantiating tool class {cls.__name__}: {e}")
                        # else: # Optional: Warn about classes missing attributes
                        #     print(f"Warning: Class {cls.__name__} in {module.__name__} looks like a tool but lacks required attributes (name, description).")


                # Add function-based tools
                for name, func in functions:
                    if callable(func) and func.__doc__:
                         if name not in processed_tool_names:
                            # Create a tool from the function
                            func_tool = type(
                                f"{name}_Tool",
                                (BaseTool,),
                                {
                                    "name": name,
                                    "description": func.__doc__.strip(), # Strip whitespace from docstring
                                    "icon": "🛠️", # Default icon for function tools
                                    "title": name.replace("_", " ").title(),
                                    # Use a lambda that captures the function correctly
                                    "_run": lambda self, input_data, captured_func=func: str(captured_func(input_data))
                                }
                            )
                            try:
                                base_tools.append(func_tool())
                                processed_tool_names.add(name)
                            except Exception as e:
                                 st.warning(f"Error instantiating tool from function {name}: {e}")
                         else:
                             print(f"Warning: Duplicate tool name '{name}' found (function vs class/built-in). Skipping function.")

            except Exception as e:
                st.warning(f"Error processing tool module {module.__name__}: {e}\n{traceback.format_exc()}") # Add traceback

        return base_tools


# =============================================
# AGENT SYSTEM (Unchanged core logic, removed prefix/suffix)
# =============================================

class AgentMemory:
    def __init__(self):
        self.chat_memory = []

    def add_user_message(self, message):
        self.chat_memory.append({"role": "user", "content": message})

    def add_ai_message(self, message):
        self.chat_memory.append({"role": "assistant", "content": message})

    def get_memory(self):
        return self.chat_memory

    def clear(self):
        self.chat_memory = []


class SimpleAgent:
    def __init__(self, model, tools, memory):
        self.model = model # Currently fixed to DEFAULT_MODEL
        self.tools = tools
        self.memory = memory
        self.max_iterations = MAX_ITERATIONS

    def run(self, input_text): # Removed callbacks for simplicity now
        """Process the input and return a response"""
        # --- Document search removed ---
        # if "database" in st.session_state and st.session_state.database: ...

        # Add user message to memory *before* building prompt
        self.memory.add_user_message(input_text)

        # Build full prompt with current memory
        full_prompt = self._build_prompt() # No longer needs input_text passed

        # --- Simple response generation (no tool loop for now) ---
        # In a real scenario, this would involve parsing the response,
        # checking for tool calls, executing them, and potentially looping.
        response = self._call_omega_model(full_prompt)

        # Add AI response to memory
        self.memory.add_ai_message(response)

        return response

    def _build_prompt(self, tool_info=None): # Removed input_text
        """Build a full prompt including memory, tools info"""
        # Format chat history
        history = ""
        for msg in self.memory.get_memory(): # Get memory directly
            history += f"{msg['role'].title()}: {msg['content']}\n\n"

        # Format tools info
        tools_prompt_part = ""
        if self.tools: # Only add tool info if tools are selected
            tools_prompt_part = "You have access to the following tools:\n"
            for tool in self.tools:
                tools_prompt_part += f"- {tool.name}: {tool.description}\n"
            tools_prompt_part += "\nWhen you need to use a tool, structure your thought process clearly.\n" # Simplified instruction

        # Build the full prompt
        # NOTE: Prompt structure is crucial and model-dependent. This is a generic example.
        prompt = f"""
You are Omega, a multi-network advanced AI agent. Engage in helpful and informative conversation.
{tools_prompt_part}
Conversation History:
{history}Assistant:""" # Let the model generate the response following "Assistant:"

        # --- Tool info part removed, needs integration into the agent loop if re-enabled ---
        # if tool_info:
        #     prompt += f"Tool usage information:\n{tool_info}\n\n"

        # --- User input is now implicitly the last message in history ---
        # prompt += f"User: {input_text}\nAssistant:"

        # Clean up extra whitespace
        return prompt.strip()


    def _call_omega_model(self, prompt):
        """Placeholder for calling the actual AI model (e.g., 'abc_model')"""
        print("-" * 50)
        print(f"--- Sending Prompt to {self.model} ---")
        print(prompt)
        print("-" * 50)
        # This is where you would integrate with your actual model API
        # For example: response = your_model_library.generate(prompt=prompt, model=self.model)
        # Placeholder response:
        response = f"Omega Placeholder Response: Received prompt for model '{self.model}'. Processing..."
        # Simulate tool usage detection for demonstration
        if "calculate 2+2" in prompt.lower() and any(t.name == 'calculator' for t in self.tools):
             response += "\nThought: The user asked for a calculation. I have a calculator tool. I should use it.\nAction: I'll use the calculator tool with input: 2 + 2"
             # In a real agent, you'd parse this, run the tool, and form a final response.
             tool_result = "[Simulated Calculator Result: 4]"
             response += f"\nObservation: {tool_result}\nFinal Answer: The result of 2 + 2 is 4."
        elif "calculate" in prompt.lower() and not any(t.name == 'calculator' for t in self.tools):
             response += "\nThought: The user asked for a calculation, but the calculator tool is not available. I cannot perform the calculation directly."
             response += "\nFinal Answer: I can't perform the calculation because the calculator tool isn't enabled."


        return response


# =============================================
# UI COMPONENTS (Modified for new session handling, removed settings)
# =============================================

def sidebar():
    with st.sidebar:
        st.image("appicon.ico" if os.path.exists("appicon.ico") else "https://img.icons8.com/fluency/48/000000/omega.png", width=60) # Placeholder icon
        st.markdown("## Omega Sessions") # Renamed
        st.markdown("---")

        if st.button("➕ Start New Session", use_container_width=True):
            # Generate a more unique session ID
            new_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            st.session_state.session_id = new_session_id
            st.session_state.selected_page = 'Chat' # Go to chat page on new session
            st.session_state.editing_session_id = None # Clear any editing state
            # Initialize session details (name=ID, starred=False)
            st.session_state.storage.save_session_details(new_session_id, new_session_id, False)
            # Clear agent memory for the new session
            if "agent_instance" in st.session_state:
                 st.session_state.agent_instance.memory.clear()
            st.rerun()

        st.markdown("---")

        # Fetch all session details once
        all_sessions_details = st.session_state.storage.get_all_session_details()
        session_ids = list(all_sessions_details.keys())

        # Sort sessions: Starred first, then by reverse chronological (most recent first)
        # Assumes session IDs are sortable chronologically (like the new format)
        session_ids.sort(key=lambda s_id: (
            not all_sessions_details.get(s_id, {}).get('starred', False), # Starred first (False comes before True when sorting)
            s_id # Then sort by ID (descending for time)
        ), reverse=True)


        st.markdown("**Session History**")
        session_expander = st.expander(label='View Sessions', expanded=True)

        with session_expander:
            if not session_ids:
                st.caption("No sessions yet. Start a new one!")

            for session_id in session_ids:
                session_info = all_sessions_details.get(session_id, {"name": session_id, "starred": False})
                session_name = session_info.get("name", session_id)
                is_starred = session_info.get("starred", False)
                is_editing = (st.session_state.get("editing_session_id") == session_id)

                # Use columns for layout
                col1, col2, col3, col4 = st.columns([0.6, 0.1, 0.1, 0.1]) # Adjust ratios as needed

                with col1:
                    # If editing, show text input
                    if is_editing:
                        new_name = st.text_input(
                            "New Name",
                            value=session_name,
                            key=f"rename_{session_id}",
                            label_visibility="collapsed",
                            placeholder="Enter new name and press Enter",
                            # Use on_change to save and exit edit mode
                            on_change=handle_rename_session,
                            args=(session_id, is_starred),
                        )
                    # Otherwise, show the session button
                    else:
                        # Highlight the active session
                        button_type = "primary" if session_id == st.session_state.session_id else "secondary"
                        display_name = session_name if len(session_name) < 30 else session_name[:27] + "..." # Truncate long names
                        if st.button(display_name, key=f"session_{session_id}", use_container_width=True, type=button_type):
                             if st.session_state.session_id != session_id:
                                st.session_state.session_id = session_id
                                st.session_state.editing_session_id = None # Cancel editing if switching
                                # Reload history and potentially agent state
                                st.session_state.messages = st.session_state.storage.get_chat_history(session_id)
                                # Re-create agent instance to load history potentially? Or clear/load memory.
                                st.session_state.memory.clear()
                                for msg in st.session_state.messages:
                                     if msg['role'] == 'user':
                                         st.session_state.memory.add_user_message(msg['content'])
                                     elif msg['role'] == 'assistant':
                                          st.session_state.memory.add_ai_message(msg['content'])
                                st.rerun()


                # Edit button (only show if not editing)
                with col2:
                    if not is_editing:
                        if st.button("✏️", key=f"edit_{session_id}", help="Rename Session", use_container_width=True):
                             st.session_state.editing_session_id = session_id
                             st.rerun() # Rerun to show the text input

                # Star button
                with col3:
                     star_icon = "⭐" if is_starred else "☆"
                     if st.button(star_icon, key=f"star_{session_id}", help="Toggle Favorite", use_container_width=True):
                         st.session_state.storage.toggle_session_star(session_id)
                         st.session_state.editing_session_id = None # Cancel editing
                         st.rerun()

                # Delete button
                with col4:
                     if st.button("🗑️", key=f"delete_{session_id}", help="Delete Session", use_container_width=True):
                         st.session_state.storage.delete_session(session_id)
                         st.session_state.editing_session_id = None # Cancel editing
                         # If deleting the current session, switch to a default or new one
                         if st.session_state.session_id == session_id:
                             remaining_sessions = st.session_state.storage.get_all_sessions()
                             st.session_state.session_id = remaining_sessions[0] if remaining_sessions else None
                             if st.session_state.session_id is None: # If no sessions left, start new
                                 # Trigger the new session logic indirectly
                                 st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                                 st.session_state.storage.save_session_details(st.session_state.session_id, st.session_state.session_id, False)

                         st.rerun()

def handle_rename_session(session_id, current_starred_status):
    """Callback function for session rename text input"""
    new_name_key = f"rename_{session_id}"
    new_name = st.session_state.get(new_name_key)
    if new_name: # Check if name is not empty
        st.session_state.storage.save_session_details(session_id, new_name, current_starred_status)
        st.session_state.editing_session_id = None # Exit editing mode
        # No rerun here, the rerun happens naturally after the callback finishes


def chat_page():
    # Header
    st.markdown("### 💭 Chat")
    st.markdown("---")

    # Show selected tools (if any)
    if st.session_state.selected_tools:
        s = ', '.join(st.session_state.selected_tools)
        st.info(f'Active tools: {s}')
    else:
         st.info('No tools selected. Go to the Tools page to enable them.')

    # Ensure agent instance exists and has the correct tools/memory
    # Check if tools have changed since last agent creation
    current_tool_names = {tool.name for tool in st.session_state.tools}
    agent_tool_names = set()
    if "agent_instance" in st.session_state:
        agent_tool_names = {tool.name for tool in st.session_state.agent_instance.tools}

    if "agent_instance" not in st.session_state or current_tool_names != agent_tool_names:
        st.session_state.agent_instance = SimpleAgent(
            model=DEFAULT_MODEL, # Use the default model
            tools=st.session_state.tools, # Use currently selected tools
            memory=st.session_state.memory # Use shared memory
        )
        print("Agent instance created or updated with new tools.") # Debugging

    # Load chat history for the current session IF memory is empty (e.g., after page reload)
    if not st.session_state.memory.get_memory():
         st.session_state.messages = st.session_state.storage.get_chat_history(st.session_state.session_id)
         for msg in st.session_state.messages:
             if msg['role'] == 'user':
                 st.session_state.memory.add_user_message(msg['content'])
             elif msg['role'] == 'assistant':
                  st.session_state.memory.add_ai_message(msg['content'])
         print(f"Loaded {len(st.session_state.messages)} messages into memory for session {st.session_state.session_id}") # Debugging


    # Display messages from memory
    current_messages = st.session_state.memory.get_memory()
    for msg in current_messages:
        with st.chat_message(msg["role"]):
             st.write(msg["content"]) # Use write for better markdown/code rendering

    # Input
    if prompt := st.chat_input("Enter your message to Omega..."):
        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)

        # No prefix/suffix needed anymore
        original_prompt = prompt

        # Get response (Agent adds messages to its memory internally now)
        with st.chat_message("assistant"):
            with st.spinner("Omega is thinking..."):
                response = st.session_state.agent_instance.run(original_prompt)
            st.write(response) # Use write for better markdown/code rendering

        # Save both messages to storage
        session_details = st.session_state.storage.get_session_details(st.session_state.session_id)
        session_name = session_details.get('name', st.session_state.session_id)
        st.session_state.storage.save_chat_message(st.session_state.session_id, "user", original_prompt, session_name)
        st.session_state.storage.save_chat_message(st.session_state.session_id, "assistant", response, session_name)
        # No need to explicitly save session name here unless it changed

        # Rerun to clear the input box? Streamlit usually handles this with chat_input
        # st.rerun() # Usually not needed


def tools_page():
    st.markdown("### 🛠️ Tools")
    st.markdown("---")

    # Refresh tools list in case new files were added
    st.session_state.tool_manager = ToolManager()
    all_tool_names = st.session_state.tool_manager.get_tool_names()
    all_tools_dict = {tool.name: tool for tool in st.session_state.tool_manager.get_tools()}


    # Tool selection state needs to be managed carefully
    # Initialize selected_tools in session state if not present
    if 'selected_tools' not in st.session_state:
         st.session_state.selected_tools = st.session_state.initial_tools # Use initial defaults
         # Sync clicked_cards based on selected_tools
         st.session_state.clicked_cards = {tool_name: (tool_name in st.session_state.selected_tools) for tool_name in all_tool_names}

    # Keep track of tools selected *in this run*
    currently_selected_in_ui = []

    # Tool search
    tool_search = st.text_input('Filter Tools', placeholder='Filter by name or description', key='tool_search')

    # Filter tools based on search
    filtered_tool_names = all_tool_names
    if tool_search:
        search_lower = tool_search.lower()
        filtered_tool_names = [
            name for name in all_tool_names
            if search_lower in name.lower() or
               search_lower in all_tools_dict[name].description.lower()
        ]

    st.markdown(f"**Available Tools ({len(filtered_tool_names)}/{len(all_tool_names)})**")

    # Display tools as cards
    cols = st.columns(3) # Adjust number of columns if needed
    for i, tool_name in enumerate(filtered_tool_names):
        tool = all_tools_dict[tool_name]
        col = cols[i % 3]

        with col:
            card = st.container(border=True)
            with card:
                # Use the tool_name for the key, ensure it's unique
                checkbox_key = f"tool_checkbox_{tool_name}"
                # Default value comes from st.session_state.selected_tools
                default_value = tool_name in st.session_state.selected_tools
                is_selected = st.checkbox(
                    f"{tool.icon} {tool.title}",
                    value=default_value,
                    key=checkbox_key,
                    help=tool.description # Add description to help text
                )

                st.caption(f"({tool.name})") # Show the actual tool name

                # Add to the list if selected in this UI pass
                if is_selected:
                    currently_selected_in_ui.append(tool_name)
                    # Call the tool's UI method if it exists
                    if hasattr(tool, '_ui') and callable(tool._ui):
                        tool._ui()

    # Update the main selected_tools list based on the UI interaction *after* rendering all checkboxes
    # This avoids issues where checking one box affects the default state of others in the same run
    if set(st.session_state.selected_tools) != set(currently_selected_in_ui):
         st.session_state.selected_tools = currently_selected_in_ui
         # Update the actual tools used by the agent
         st.session_state.tools = st.session_state.tool_manager.get_selected_tools(st.session_state.selected_tools)
         # Optionally rerun if immediate effect is needed, but often better to let Streamlit handle it
         st.experimental_rerun()


    # Add new tool section
    st.markdown("---")
    st.markdown("### Create New Tool")

    with st.expander("Add a Custom Tool (Python Function)"):
        new_tool_name = st.text_input("Tool File Name (e.g., `my_weather_tool`)", key="new_tool_filename")
        new_tool_function = st.text_area(
            "Tool Code (Python Function)",
            height=300,
            key="new_tool_code",
            placeholder="""# Required: Define a function with a docstring.
# The function name will be the tool name.
# The docstring is the tool description.
# It must accept one argument (input_data) and return a string.

import requests # Example: Add necessary imports

def get_weather(location: str) -> str:
    \"\"\"Fetches the current weather for a given location.
    Input should be the city name (e.g., 'London').\"\"\"
    try:
        # Replace with a real API call
        api_key = "YOUR_API_KEY" # Consider how to handle secrets
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()
        description = data['weather'][0]['description']
        temp = data['main']['temp']
        return f"Current weather in {location}: {description}, {temp}°C"
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather: {e}"
    except Exception as e:
        return f"Error processing weather data: {str(e)}"

# You can add helper functions if needed, but only the main function
# with the docstring will become the tool.
"""
        )

        if st.button("Create Tool", key="create_tool_button"):
            if new_tool_name and new_tool_function:
                 # Basic validation on name
                 if not new_tool_name.isidentifier():
                     st.error("Tool file name must be a valid Python identifier (letters, numbers, underscores, cannot start with a number).")
                 else:
                    # Validate function code string
                    validation_result, has_doc, func_name = evaluate_function_string(new_tool_function)

                    if validation_result is True and has_doc is True:
                        # Create the tools directory if it doesn't exist
                        tools_dir = os.path.join(BASE_DIR, "omega_tools", "custom_tools")
                        os.makedirs(tools_dir, exist_ok=True)

                        # Save the tool (ensure .py extension)
                        tool_filename = f"{new_tool_name}.py" if not new_tool_name.endswith(".py") else new_tool_name
                        tool_path = os.path.join(tools_dir, tool_filename)

                        try:
                             with open(tool_path, "w", encoding='utf-8') as f:
                                 f.write(new_tool_function)

                             st.success(f"Tool '{func_name}' saved successfully as `{tool_filename}`!")
                             # Clear inputs after success
                             st.session_state.new_tool_filename = ""
                             st.session_state.new_tool_code = ""
                             # No need to re-initialize tool manager here, it happens on page load/rerun
                             st.rerun() # Rerun to refresh the tool list display

                        except OSError as e:
                             st.error(f"Error saving tool file: {e}")
                        except Exception as e:
                             st.error(f"An unexpected error occurred: {e}")

                    else:
                        if validation_result is not True:
                            st.error(f"Error in function code: {validation_result}")
                        if not has_doc:
                            st.error("Function definition must include a docstring for the tool description.")
            else:
                st.error("Please provide both a valid file name and the Python function code for the tool.")


# --- Settings Page Removed ---
# def settings_page(): ...

def info_page():
    st.markdown("### ℹ️ About Omega") # Renamed
    st.markdown("---")

    # Enhanced Description
    st.markdown("""
    # Ω Omega: Advanced Multi-Network AI Agent

    **Omega represents a paradigm shift in intelligent systems, functioning as a sophisticated, multi-network AI agent designed for complex task execution and synergistic human-AI collaboration.**

    Leveraging a distributed cognitive architecture, Omega integrates seamlessly with a dynamic array of specialized tools and knowledge sources. Its core capabilities enable fluid interaction, contextual understanding, and adaptive problem-solving across diverse domains.

    ## Core Architecture & Capabilities:

    * **Advanced AI Integration:** Built upon a foundation allowing integration with state-of-the-art language models (like the configured `abc_model`) for nuanced understanding and generation.
    * **Dynamic Tool Orchestration:** Omega intelligently selects and utilizes a configurable suite of tools – from computational engines to data retrieval systems – to augment its reasoning and execution capabilities. View and manage available tools on the **Tools** page.
    * **Persistent Session Management:** Robust session handling allows for conversation continuity and state preservation. Manage your sessions via the sidebar, including starring favorites and renaming for clarity.
    * **Extensible Tool Framework:** Developers can rapidly prototype and integrate custom tools using Python functions with descriptive docstrings via the **Tools** page, fostering a constantly evolving ecosystem.
    * **Streamlined User Interface:** A focused interface prioritizes efficient interaction and clear visualization of the AI's process.

    ## Operational Workflow:

    1.  **Initialization:** Omega loads its core configuration and available toolset.
    2.  **Interaction (Chat Page):** Engage with Omega through the chat interface. Initiate new conversations or resume existing ones selected from the sidebar.
    3.  **Tool Selection (Tools Page):** Activate or deactivate tools to tailor Omega's capabilities for specific tasks. Create new, custom tools as needed.
    4.  **Task Execution:** Based on the conversation and selected tools, Omega processes requests, potentially invoking tools to gather information or perform actions, and provides informative responses.

    ## Vision:

    Omega aims to be a cornerstone platform for applications requiring advanced AI reasoning, flexible tool usage, and persistent, context-aware interactions. Its architecture is designed for scalability and adaptation to future advancements in AI and networked systems.

    ---
    *Omega utilizes UTF-8 character encoding for broad compatibility.*
    *Session data is stored locally in the `omega_data` directory.*
    *Custom tools are located in the `omega_tools/custom_tools` directory.*
    """)
    # --- Removed the example code block ---


# =============================================
# MAIN APPLICATION SETUP (Adjusted for removed settings)
# =============================================

def option_menu_cb(): # This might not be needed if using buttons or native components
    """Callback for menu selection (if using streamlit-option-menu)"""
    # This assumes 'menu_opt' is the key used in option_menu
    if 'menu_opt' in st.session_state:
        st.session_state.selected_page = st.session_state.menu_opt


def menusetup():
    """Set up the navigation menu"""
    # --- Updated menu items and icons ---
    list_menu = ["Chat", "Tools", "Info"]
    list_pages = [chat_page, tools_page, info_page]
    st.session_state.dictpages = dict(zip(list_menu, list_pages))
    list_icons = ['chat-left-text-fill', 'tools', 'info-circle-fill'] # Use filled icons for active look

    # --- Sticky Tabs CSS (Optional) ---
    # Adjust 'top' value based on your Streamlit header height (usually 0 or around 40-60px)
    st.markdown("""
    <style>
        /* Target the Streamlit Blocks container that holds the option_menu */
        div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stHorizontalBlock"] {
            position: sticky;
            top: 0; /* Adjust this value if header overlaps */
            z-index: 999;
            background-color: white; /* Match Streamlit's background */
            padding-top: 10px; /* Add some padding */
            padding-bottom: 10px; /* Add some padding */
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); /* Optional shadow */
            margin-bottom: 20px; /* Space below the sticky menu */
        }
        /* Fallback for older Streamlit versions or custom HTML */
        .stApp > header { /* Hide default Streamlit header if menu is sticky */
             display: none;
        }
        .nav-container { /* Your custom HTML nav container */
            position: sticky;
            top: 0;
            z-index: 999;
            background-color: white;
            /* Add other styles as needed */
        }
    </style>
    """, unsafe_allow_html=True)


    # --- Attempt to use streamlit-option-menu first ---
    try:
        from streamlit_option_menu import option_menu

        # Ensure selected_page exists and is valid before passing to default_index
        if 'selected_page' not in st.session_state or st.session_state.selected_page not in list_menu:
             st.session_state.selected_page = "Chat" # Default to Chat

        selected = option_menu(
            None, # No title for the menu
            list_menu,
            icons=list_icons,
            # menu_icon="cast", # Optional: add an icon to the menu bar itself
            default_index=list_menu.index(st.session_state.selected_page),
            orientation="horizontal",
            key='menu_opt', # Use a key to access selection state
            styles={ # Optional styling
                 "container": {"padding": "5px!important", "background-color": "#fafafa"},
                 "icon": {"color": "orange", "font-size": "18px"},
                 "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
                 "nav-link-selected": {"background-color": "#02ab21", "font-weight": "bold"}, # Green background for selected
            }
        )
        # Update selected page based on menu interaction
        # Check if the selected value is different from the current state to avoid unnecessary reruns
        if selected and selected != st.session_state.selected_page:
            st.session_state.selected_page = selected
            st.session_state.editing_session_id = None # Cancel editing on tab switch
            st.rerun() # Rerun is needed to display the new page content

    except ImportError:
        st.warning("`streamlit-option-menu` not found. Using basic buttons for navigation.")
        # Fallback to simple buttons if option_menu is not installed
        cols = st.columns(len(list_menu))
        for i, page_name in enumerate(list_menu):
             with cols[i]:
                 # Highlight the active button
                 button_type = "primary" if st.session_state.selected_page == page_name else "secondary"
                 if st.button(page_name, use_container_width=True, type=button_type, key=f"nav_{page_name}"):
                     if st.session_state.selected_page != page_name:
                         st.session_state.selected_page = page_name
                         st.session_state.editing_session_id = None # Cancel editing on tab switch
                         st.rerun() # Rerun to show the new page


def pageselection():
    """Call the appropriate page function based on selection"""
    page_func = st.session_state.dictpages.get(st.session_state.selected_page)
    if page_func:
         page_func()
    else:
         # Fallback if selected_page is somehow invalid
         st.error("Selected page not found!")
         st.session_state.selected_page = "Chat" # Reset to default
         st.rerun()


def ensure_session_state():
    """Initialize session state variables if they don't exist"""
    if "session_id" not in st.session_state or st.session_state.session_id is None:
        # Try to load the most recent session, or create a new one if none exist
        all_sessions = CSVStorage().get_all_session_details()
        if all_sessions:
            # Sort by ID (assuming chronological format like implemented)
             sorted_ids = sorted(all_sessions.keys(), reverse=True)
             st.session_state.session_id = sorted_ids[0]
        else:
             # Create a brand new session ID if none exist
             st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
             # Ensure the initial session is saved in the sessions file
             CSVStorage().save_session_details(st.session_state.session_id, st.session_state.session_id, False)


    # --- Model is now fixed ---
    # if "model" not in st.session_state:
    #     st.session_state.model = DEFAULT_MODEL

    if "tool_manager" not in st.session_state:
        st.session_state.tool_manager = ToolManager()
        # st.session_state.tool_list = st.session_state.tool_manager.structured_tools # Not directly used anymore

    if "initial_tools" not in st.session_state:
        # Default tools to enable initially (e.g., calculator)
        st.session_state.initial_tools = ['calculator']

    # Initialize selected_tools based on initial_tools if not set
    if "selected_tools" not in st.session_state:
         st.session_state.selected_tools = st.session_state.initial_tools

    # Initialize the actual tool instances based on selected_tools
    if "tools" not in st.session_state:
         st.session_state.tools = st.session_state.tool_manager.get_selected_tools(st.session_state.selected_tools)

    # Clicked cards state for tool checkboxes (might be redundant now)
    if "clicked_cards" not in st.session_state:
         st.session_state.clicked_cards = {tool_name: (tool_name in st.session_state.selected_tools)
                                          for tool_name in st.session_state.tool_manager.get_tool_names()}

    if "memory" not in st.session_state:
        st.session_state.memory = AgentMemory()
        # Note: Memory is now loaded per-session in chat_page if empty

    # --- Document related states removed ---
    # if "doc_manager" not in st.session_state: ...
    # if "documents" not in st.session_state: ...
    # if "database" not in st.session_state: ...

    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Chat" # Default to Chat page

    # --- Prefix/Suffix removed ---
    # if "prefix" not in st.session_state: ...
    # if "suffix" not in st.session_state: ...

    if "storage" not in st.session_state:
        st.session_state.storage = CSVStorage()

    # State for session renaming UI
    if "editing_session_id" not in st.session_state:
         st.session_state.editing_session_id = None



def main():
    """Main application entry point"""
    # Set page config - Use Omega Branding
    icon_path = "appicon.ico"
    try:
        im = Image.open(icon_path) if os.path.exists(icon_path) else None
    except Exception as e:
        print(f"Warning: Could not load icon {icon_path}: {e}")
        im = None # Use Streamlit default icon

    st.set_page_config(
        page_title="Omega AI Agent", # Renamed
        page_icon=im,
        layout="wide", # Use wide layout for potentially more space
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo/omega', # Update link
            'Report a bug': "https://github.com/your-repo/omega/issues", # Update link
            'About': "# Omega: Advanced Multi-Network AI Agent" # Updated About
        }
    )

    # Initialize session state ensuring all keys are present
    ensure_session_state()

    # Ensure tools directory exists (ToolManager also does this, but good practice)
    tools_dir = os.path.join(BASE_DIR, "omega_tools", "custom_tools")
    os.makedirs(tools_dir, exist_ok=True)

    # --- Main UI Structure ---
    # Title is now part of the Info page, remove main title if desired for cleaner look
    # st.title("Ω Omega AI Agent") # Optional main title

    # Render Sidebar (Session Management)
    sidebar()

    # Render Main Content Area
    # Column layout: Fixed width for main content, potentially empty space on sides
    # Or keep it simple and let Streamlit manage width
    main_col, _ = st.columns([3, 1]) # Example: Give main content 3/4 width

    with main_col:
        # Render Navigation Menu (Tabs)
        menusetup()

        # Render the selected page content
        pageselection()


if __name__ == "__main__":
    main()
