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
import traceback
import locale # For checking default encoding (optional)

# --- Config Constants ---
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
# MODELS removed as settings page is removed
DEFAULT_MODEL = 'abc_model' # Agent will use this
MAX_ITERATIONS = 20
# --- Define data directory ---
DATA_DIR = os.path.join(BASE_DIR, "omega_data") # Store data in subfolder
TOOLS_DIR = os.path.join(BASE_DIR, "omega_tools") # Tools folder

# =============================================
# STORAGE MANAGEMENT (Using system default encoding)
# =============================================
class CSVStorage:
    def __init__(self, csv_path=None, sessions_path=None):
        # --- Determine System Default Encoding (for information/debugging) ---
        try:
            self.default_encoding = locale.getpreferredencoding(False)
            print(f"--- CSVStorage: Using system default file encoding: {self.default_encoding} ---")
        except Exception:
            self.default_encoding = "unknown (using Python default)"
            print("--- CSVStorage: Could not determine system default encoding. Relying on Python's default. ---")
        # --- End Encoding Info ---

        if csv_path is None:
            os.makedirs(DATA_DIR, exist_ok=True) # Ensure data dir exists
            self.csv_path = os.path.join(DATA_DIR, "chat_history.csv")
        else:
            self.csv_path = csv_path

        if sessions_path is None:
            os.makedirs(DATA_DIR, exist_ok=True) # Ensure data dir exists
            self.sessions_path = os.path.join(DATA_DIR, "sessions.csv")
        else:
            self.sessions_path = sessions_path

        # Initialize CSV files if they don't exist
        self._initialize_csv_files()

    def _initialize_csv_files(self):
        # Create chat history CSV if doesn't exist
        if not os.path.exists(self.csv_path):
            try:
                # WRITE: Using system default encoding (no encoding specified)
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["session_id", "timestamp", "role", "content", "session_name"])
            except Exception as e:
                 st.error(f"Error creating {self.csv_path}: {e}")

        # Create/Update sessions CSV
        if not os.path.exists(self.sessions_path):
            try:
                # WRITE: Using system default encoding
                with open(self.sessions_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Add 'starred' column from the start now
                    writer.writerow(["session_id", "session_name", "starred"])
            except Exception as e:
                 st.error(f"Error creating {self.sessions_path}: {e}")
        else:
            # Check if existing sessions file has the 'starred' column
            header = None
            sessions_data = []
            needs_update = False
            try:
                # READ: Using system default encoding
                with open(self.sessions_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header and "starred" not in header:
                        needs_update = True
                        sessions_data = list(reader) # Read data only if update needed
            except Exception as e:
                st.error(f"Error reading {self.sessions_path} for header check: {e}")
                return # Avoid rewrite if read failed

            # Rewrite if update needed and read successful
            if needs_update and header:
                try:
                    # WRITE: Using system default encoding
                    with open(self.sessions_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(header + ["starred"])
                        for row in sessions_data:
                            writer.writerow(row + [0]) # Add default starred=0
                    print(f"Added 'starred' column to {self.sessions_path}")
                except Exception as e:
                    st.error(f"Error rewriting {self.sessions_path} to add 'starred' column: {e}")


    def save_chat_message(self, session_id, role, content, session_name=None):
        # Get session name using potentially modified get_session_details
        if session_name is None:
             session_name = self.get_session_details(session_id).get('name', session_id)
        try:
            # APPEND: Using system default encoding
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([session_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), role, content, session_name])
        except Exception as e:
             st.error(f"Error appending to {self.csv_path}: {e}")

    def get_chat_history(self, session_id):
        if not os.path.exists(self.csv_path): return []
        messages = []
        try:
            # READ: Using system default encoding
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Basic check for required keys
                    if row and row.get("session_id") == session_id and "role" in row and "content" in row:
                        messages.append({"role": row["role"], "content": row["content"]})
        except FileNotFoundError: return []
        except Exception as e:
            st.error(f"Error reading chat history from {self.csv_path}: {e}")
            return []
        return messages

    def get_all_sessions(self):
        """Returns a list of session_ids from the sessions file."""
        if not os.path.exists(self.sessions_path): return []
        session_ids = set()
        try:
             # READ: Using system default encoding
             with open(self.sessions_path, 'r') as f:
                 reader = csv.DictReader(f) # Use DictReader to easily access "session_id"
                 for row in reader:
                     if row and "session_id" in row:
                         session_ids.add(row["session_id"])
        except FileNotFoundError: return []
        except Exception as e:
            st.error(f"Error reading session IDs from {self.sessions_path}: {e}")
            return []
        return list(session_ids)

    def get_all_session_details(self):
        """Returns dict {session_id: {'name': str, 'starred': bool}}"""
        if not os.path.exists(self.sessions_path): return {}
        session_details = {}
        try:
            # READ: Using system default encoding
            with open(self.sessions_path, 'r') as f:
                reader = csv.DictReader(f)
                if "starred" not in reader.fieldnames:
                     st.warning(f"'starred' column missing in {self.sessions_path}. Favorites may not work until file is updated.")
                for row in reader:
                    if row and "session_id" in row and "session_name" in row:
                        starred_val = row.get("starred", "0") # Default if missing
                        try: is_starred = bool(int(starred_val))
                        except (ValueError, TypeError): is_starred = False
                        session_details[row["session_id"]] = {"name": row["session_name"], "starred": is_starred}
        except FileNotFoundError: return {}
        except Exception as e:
            st.error(f"Error reading session details from {self.sessions_path}: {e}")
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
        elif session_id in self.get_all_sessions(): # Add details if missing but session exists
             self.save_session_details(session_id, session_id, True) # Star new entry

    def delete_session(self, session_id):
        # Delete from chat history
        all_messages = []
        fieldnames = ["session_id", "timestamp", "role", "content", "session_name"] # Default header
        if os.path.exists(self.csv_path):
            read_success = False
            try:
                # READ: Using system default encoding
                with open(self.csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames: fieldnames = reader.fieldnames
                    all_messages = [row for row in reader if row.get("session_id") != session_id]
                read_success = True
            except Exception as e:
                 st.error(f"Error reading chat history ({self.csv_path}) for deletion: {e}")

            if read_success: # Only rewrite if read was successful
                 try:
                     # WRITE: Using system default encoding
                     with open(self.csv_path, 'w', newline='') as f:
                         writer = csv.DictWriter(f, fieldnames=fieldnames)
                         writer.writeheader()
                         writer.writerows(all_messages)
                 except Exception as e:
                     st.error(f"Error rewriting chat history ({self.csv_path}) after delete: {e}")

        # Delete from sessions file
        all_details = self.get_all_session_details()
        if session_id in all_details:
            del all_details[session_id]
            self._write_sessions_file(all_details) # Uses default encoding write

    def _write_sessions_file(self, session_details_dict):
        """Helper function to write the entire sessions CSV using system default encoding."""
        try:
            # WRITE: Using system default encoding
            with open(self.sessions_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["session_id", "session_name", "starred"])
                for s_id, details in session_details_dict.items():
                    writer.writerow([s_id, details.get("name", s_id), 1 if details.get("starred", False) else 0])
        except Exception as e:
             st.error(f"Error writing sessions file {self.sessions_path}: {e}")

# --- DocumentManager Removed ---
# class DocumentManager: ...
# class SimpleDocument: ...

# =============================================
# TOOLS SYSTEM (No icons)
# =============================================

class BaseTool:
    name = 'Base_tool'
    link = '[https://github.com/your-repo/omega](https://github.com/your-repo/omega)' # Updated link
    # icon removed
    description = 'Base tool description'
    title = 'Base Tool' # Added title

    def __init__(self, **kwargs):
        for key, value in kwargs.items(): setattr(self, key, value)
        if not hasattr(self, 'title'): self.title = self.name.replace("_", " ").title() # Default title

    def run(self, input_data): return self._run(input_data)
    def _run(self, input_data): print(f'Running base tool: {input_data}'); return f'Success: {input_data}'
    def _ui(self): pass

class CalculatorTool(BaseTool):
    name = 'calculator'
    # icon removed
    title = 'Calculator'
    description = 'Perform arithmetic calculations. Input format: mathematical expression (e.g., "2 + 2", "sin(30)", "sqrt(16)")'
    def _run(self, expression):
        try:
            safe_dict = {'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum, 'len': len, 'pow': pow, 'int': int, 'float': float, 'str': str, 'sorted': sorted, 'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'range': range}
            import math
            for name in dir(math):
                 if not name.startswith("_"): safe_dict[name] = getattr(math, name)
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return f"Result: {result}"
        except Exception as e: return f"Error: {str(e)}"

# --- Tool Helper Functions ---
def has_required_attributes(cls):
    required_attributes = ['name', 'description']
    try:
        for attr in required_attributes:
            if not hasattr(cls, attr): return False
        return True
    except Exception: return False

def has_docstring(function_node):
    return (len(function_node.body) > 0 and isinstance(function_node.body[0], ast.Expr) and isinstance(function_node.body[0].value, ast.Str))

def evaluate_function_string(func_str):
    try:
        parsed_ast = ast.parse(func_str)
        function_node = next((node for node in parsed_ast.body if isinstance(node, ast.FunctionDef)), None)
        if not function_node: return False, False, None
        tool_name = function_node.name; temp_globals = {}
        compiled_func = compile(func_str, '<string>', 'exec')
        exec(compiled_func, temp_globals)
        doc_exist = has_docstring(function_node)
        if tool_name not in temp_globals or not callable(temp_globals[tool_name]): raise ValueError(f"Function '{tool_name}' not found/callable")
        return True, doc_exist, tool_name
    except Exception as e: return e, False, None

def get_class_func_from_module(module):
    members = inspect.getmembers(module); functions = []; classes = []
    for name, member in members:
        try:
            if inspect.isfunction(member) and member.__module__ == module.__name__ and member.__doc__: functions.append((name, member))
            # Make sure it's actually a class defined in the module, not imported
            if inspect.isclass(member) and member.__module__ == module.__name__: classes.append((name, member))
        except Exception: pass # Ignore errors during inspection
    return classes, functions

def import_from_file(file_path, module_name=None):
     if module_name is None: module_name = os.path.basename(file_path).replace(".py", "")
     spec = importlib.util.spec_from_file_location(module_name, file_path)
     if spec is None or spec.loader is None: return None
     module = importlib.util.module_from_spec(spec); sys.modules[module_name] = module
     try: spec.loader.exec_module(module)
     except Exception as e:
          print(f"Error executing module {module_name} from {file_path}: {e}")
          if module_name in sys.modules: del sys.modules[module_name] # Clean up partial import
          return None
     return module

def monitor_folder(folder_path):
     # Use TOOLS_DIR consistently
     if not os.path.exists(folder_path):
         try: os.makedirs(folder_path)
         except OSError as e: print(f"Error creating dir {folder_path}: {e}"); return []
     if folder_path not in sys.path: sys.path.insert(0, folder_path) # Insert at start
     try: python_files = [f for f in os.listdir(folder_path) if f.endswith('.py') and f != "__init__.py"]
     except Exception as e: print(f"Error listing dir {folder_path}: {e}"); return []
     monitored_modules = []
     for py_file in python_files:
         file_path = os.path.join(folder_path, py_file); module_name = py_file[:-3]
         try: # Reload if already imported
             if module_name in sys.modules: module = importlib.reload(sys.modules[module_name])
             else: module = import_from_file(file_path, module_name)
             if module: monitored_modules.append(module)
         except Exception as e:
             st.warning(f"Error importing/reloading tool '{py_file}': {e}")
             if module_name in sys.modules: del sys.modules[module_name] # Clean up partial import/reload
     return monitored_modules

class ToolManager:
    def __init__(self):
        self.structured_tools = self._make_tools_list()
        self.tools_description = self._make_tools_description()
    def _make_tools_description(self): return {tool.name: tool.description for tool in self.structured_tools if hasattr(tool, 'name') and hasattr(tool, 'description')}
    def get_tools(self): return self.structured_tools
    def get_tool_names(self): return [tool.name for tool in self.structured_tools if hasattr(tool, 'name')]
    def get_selected_tools(self, selected_tool_names): return [t for t in self.structured_tools if hasattr(t, 'name') and t.name in selected_tool_names]
    def _make_tools_list(self):
        base_tools = [CalculatorTool()] # Start with built-ins
        # Define tool directories relative to main TOOLS_DIR
        custom_tools_dir = os.path.join(TOOLS_DIR, "custom_tools")
        # Ensure directories exist
        os.makedirs(TOOLS_DIR, exist_ok=True)
        os.makedirs(custom_tools_dir, exist_ok=True)

        monitored_modules = monitor_folder(custom_tools_dir)
        processed_tool_names = {t.name for t in base_tools}
        for module in monitored_modules:
            try:
                classes, functions = get_class_func_from_module(module)
                # Add class-based tools (check inheritance carefully)
                for _, cls in classes:
                    if inspect.isclass(cls) and issubclass(cls, BaseTool) and cls is not BaseTool and has_required_attributes(cls):
                        try:
                            instance = cls()
                            if instance.name not in processed_tool_names:
                                base_tools.append(instance); processed_tool_names.add(instance.name)
                            # else: print(f"Warning: Duplicate tool name '{instance.name}' (class)")
                        except Exception as e: st.warning(f"Error instantiating tool class {cls.__name__}: {e}")
                # Add function-based tools (check name collision)
                for name, func in functions:
                    if callable(func) and func.__doc__ and name not in processed_tool_names:
                        func_tool = type(f"{name}_Tool", (BaseTool,), {
                            "name": name, "description": func.__doc__.strip(),
                            "title": name.replace("_", " ").title(), # Default title
                            # No icon defined here
                            "_run": lambda self, input_data, cf=func: str(cf(input_data))})
                        try:
                            base_tools.append(func_tool()); processed_tool_names.add(name)
                        except Exception as e: st.warning(f"Error instantiating tool function {name}: {e}")
                    # elif name in processed_tool_names: print(f"Warning: Duplicate tool name '{name}' (function)")
            except Exception as e: st.warning(f"Error processing module {module.__name__}: {e}\n{traceback.format_exc()}")
        return base_tools

# =============================================
# AGENT SYSTEM
# =============================================
class AgentMemory:
    def __init__(self): self.chat_memory = []
    def add_user_message(self, message): self.chat_memory.append({"role": "user", "content": message})
    def add_ai_message(self, message): self.chat_memory.append({"role": "assistant", "content": message})
    def get_memory(self): return self.chat_memory
    def clear(self): self.chat_memory = []

class SimpleAgent:
    def __init__(self, model, tools, memory):
        self.model = model; self.tools = tools; self.memory = memory; self.max_iterations = MAX_ITERATIONS
    def run(self, input_text): # Removed callbacks=None for simplicity
        self.memory.add_user_message(input_text) # Add user message first
        full_prompt = self._build_prompt() # Build prompt based on current memory
        response = self._call_omega_model(full_prompt) # Call LLM
        self.memory.add_ai_message(response) # Add AI response
        return response

    def _build_prompt(self): # No longer needs input_text passed directly
        history = "".join(f"{msg['role'].title()}: {msg['content']}\n\n" for msg in self.memory.get_memory())
        tools_prompt_part = ""
        if self.tools:
            tools_prompt_part = "You have access to the following tools:\n" + "".join(f"- {tool.name}: {tool.description}\n" for tool in self.tools) + "\nWhen you need to use a tool, think step-by-step and state your intention clearly.\n" # Modified instruction
        # Updated prompt structure for Omega
        prompt = f"""
You are Omega, a multi-network advanced AI agent. Engage in helpful and informative conversation.
{tools_prompt_part}
Conversation History:
{history}Assistant:""" # Let model generate from here
        return prompt.strip()

    def _call_omega_model(self, prompt):
        # Placeholder - Replace with your actual model call
        print("-" * 50 + f"\n--- Sending Prompt to {self.model} ---\n{prompt}\n" + "-" * 50)
        # Simulate response and potential tool use thought process
        response = f"Omega Placeholder Response: Processing request using model '{self.model}'..."
        if "calculate 2+2" in prompt.lower() and any(t.name == 'calculator' for t in self.tools):
             # Simulate tool call thought process (adapt based on actual model needs)
             response += "\nThought: User wants a calculation. The 'calculator' tool is available. I will prepare the input and state my action.\nAction: Use calculator tool with input '2 + 2'."
             # In a real agent, you'd parse 'Action', run the tool, get the result, and formulate final answer
             tool_result = "[Simulated Tool Result: 4]"
             response += f"\nObservation: {tool_result}\nFinal Answer: The result of 2 + 2 is 4."
        elif "calculate" in prompt.lower() and not any(t.name == 'calculator' for t in self.tools):
             response += "\nThought: User wants a calculation, but 'calculator' tool is not enabled.\nFinal Answer: I cannot perform calculations as the required tool is not active."
        # Add more simulation logic here if needed
        return response

# =============================================
# UI COMPONENTS (No icons, default encoding, no Settings)
# =============================================

def sidebar():
    with st.sidebar:
        # Optional: Add Omega logo if you have one
        # st.image("omega_logo.png", width=60)
        st.markdown("## Omega Sessions")
        st.markdown("---")

        if st.button("Start New Session", use_container_width=True):
            # Use more robust session ID format
            new_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            st.session_state.session_id = new_session_id
            st.session_state.selected_page = 'Chat'
            st.session_state.editing_session_id = None # Clear editing state
             # Ensure new session details are saved (using default encoding)
            st.session_state.storage.save_session_details(new_session_id, new_session_id, False)
            if "agent_instance" in st.session_state: st.session_state.agent_instance.memory.clear() # Clear memory for new session
            st.rerun()

        st.markdown("---")
        all_sessions_details = st.session_state.storage.get_all_session_details()
        session_ids = list(all_sessions_details.keys())
        # Sort: Starred first, then by reverse ID (most recent)
        session_ids.sort(key=lambda s_id: (not all_sessions_details.get(s_id, {}).get('starred', False), s_id), reverse=True)

        st.markdown("**Session History**")
        session_expander = st.expander(label='View Sessions', expanded=True)
        with session_expander:
            if not session_ids: st.caption("No sessions yet.")
            for session_id in session_ids:
                session_info = all_sessions_details.get(session_id, {"name": session_id, "starred": False})
                session_name = session_info.get("name", session_id); is_starred = session_info.get("starred", False)
                is_editing = (st.session_state.get("editing_session_id") == session_id)
                # Adjusted columns for text buttons
                col1, col2, col3, col4 = st.columns([0.5, 0.17, 0.18, 0.15]) # Name | Edit | Star/Unstar | Delete

                with col1: # Session Name / Rename Input
                    if is_editing:
                        st.text_input(f"rename_{session_id}", value=session_name, key=f"rename_input_{session_id}", label_visibility="collapsed", placeholder="Enter new name", on_change=handle_rename_session, args=(session_id, is_starred))
                    else:
                        button_type = "primary" if session_id == st.session_state.session_id else "secondary"
                        display_name = session_name if len(session_name) < 35 else session_name[:32] + "..." # Truncate long names
                        if st.button(display_name, key=f"session_{session_id}", use_container_width=True, type=button_type, help=f"Switch to: {session_name}"):
                             if st.session_state.session_id != session_id:
                                st.session_state.session_id = session_id; st.session_state.editing_session_id = None
                                # Reload history into memory
                                st.session_state.messages = st.session_state.storage.get_chat_history(session_id)
                                st.session_state.memory.clear()
                                for msg in st.session_state.messages:
                                     role, content = msg.get('role'), msg.get('content')
                                     if role == 'user' and content: st.session_state.memory.add_user_message(content)
                                     elif role == 'assistant' and content: st.session_state.memory.add_ai_message(content)
                                st.rerun()
                with col2: # Edit Button
                    if not is_editing:
                        if st.button("Edit", key=f"edit_{session_id}", help="Rename Session", use_container_width=True):
                             st.session_state.editing_session_id = session_id; st.rerun()
                with col3: # Star/Unstar Button
                     star_text = "Unstar" if is_starred else "Star"
                     if st.button(star_text, key=f"star_{session_id}", help="Toggle Favorite", use_container_width=True):
                         st.session_state.storage.toggle_session_star(session_id); st.session_state.editing_session_id = None; st.rerun()
                with col4: # Delete Button
                     if st.button("Delete", key=f"delete_{session_id}", help="Delete Session", use_container_width=True):
                         st.session_state.storage.delete_session(session_id); st.session_state.editing_session_id = None
                         # Switch session if current one was deleted
                         if st.session_state.session_id == session_id:
                             remaining = st.session_state.storage.get_all_sessions()
                             st.session_state.session_id = remaining[0] if remaining else f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}" # Go to newest or create new
                             if not remaining: # Ensure the newly created ID is saved if needed
                                st.session_state.storage.save_session_details(st.session_state.session_id, st.session_state.session_id, False)
                         st.rerun()

def handle_rename_session(session_id, current_starred_status):
    """Callback function for session rename text input"""
    new_name_key = f"rename_input_{session_id}" # Match the text_input key
    new_name = st.session_state.get(new_name_key)
    if new_name and new_name.strip(): # Check if name is not empty or just whitespace
        st.session_state.storage.save_session_details(session_id, new_name.strip(), current_starred_status)
        st.session_state.editing_session_id = None # Exit editing mode
        # Rerun should happen naturally after callback finishes and state changes
    elif new_name is not None: # User cleared the input or entered only whitespace
         st.warning("Session name cannot be empty.")
         # Don't exit editing mode, let user fix it. Force a rerun to clear the warning if needed.
         # st.experimental_rerun() # Usually not needed, state change might trigger it

def chat_page():
    st.markdown("### Chat") # Removed icon
    st.markdown("---")

    # Ensure storage is initialized in session state
    if "storage" not in st.session_state:
        st.session_state.storage = CSVStorage()
    if "memory" not in st.session_state:
        st.session_state.memory = AgentMemory()
    if "tools" not in st.session_state: # Ensure tools state exists
         if 'tool_manager' not in st.session_state: st.session_state.tool_manager = ToolManager()
         if 'selected_tools' not in st.session_state: st.session_state.selected_tools = ['calculator'] # Default
         st.session_state.tools = st.session_state.tool_manager.get_selected_tools(st.session_state.selected_tools)


    if st.session_state.selected_tools:
        st.info(f'Active tools: {", ".join(st.session_state.selected_tools)}')
    else:
         st.info('No tools selected. Go to the Tools page to enable them.')

    # --- Agent Initialization ---
    current_tool_names = {tool.name for tool in st.session_state.tools}
    agent_tool_names = set()
    if "agent_instance" in st.session_state: agent_tool_names = {tool.name for tool in st.session_state.agent_instance.tools}

    # Recreate agent if instance doesn't exist or tools changed
    if "agent_instance" not in st.session_state or current_tool_names != agent_tool_names:
        st.session_state.agent_instance = SimpleAgent(model=DEFAULT_MODEL, tools=st.session_state.tools, memory=st.session_state.memory)
        print("Agent instance created or updated.")

    # --- Load History / Display ---
    # Load history if memory is empty (e.g., page reload/session switch)
    if not st.session_state.memory.get_memory() and "session_id" in st.session_state:
         st.session_state.messages = st.session_state.storage.get_chat_history(st.session_state.session_id)
         for msg in st.session_state.messages:
             role, content = msg.get('role'), msg.get('content')
             if role == 'user' and content: st.session_state.memory.add_user_message(content)
             elif role == 'assistant' and content: st.session_state.memory.add_ai_message(content)
         print(f"Loaded {len(st.session_state.messages)} messages into memory for session {st.session_state.session_id}")

    # Display messages from memory
    for msg in st.session_state.memory.get_memory():
        with st.chat_message(msg["role"]):
             st.write(msg["content"]) # Use write for markdown rendering

    # --- Input and Response ---
    if prompt := st.chat_input("Enter your message to Omega..."):
        with st.chat_message("user"):
            st.write(prompt)
        # Agent adds user message to memory internally now
        with st.chat_message("assistant"):
            with st.spinner("Omega is thinking..."):
                response = st.session_state.agent_instance.run(prompt) # Agent run handles memory
            st.write(response)
        # Save interaction (Agent already updated memory)
        session_details = st.session_state.storage.get_session_details(st.session_state.session_id)
        session_name = session_details.get('name', st.session_state.session_id)
        st.session_state.storage.save_chat_message(st.session_state.session_id, "user", prompt, session_name)
        st.session_state.storage.save_chat_message(st.session_state.session_id, "assistant", response, session_name)
        # Rerun usually not needed for chat input


def tools_page():
    st.markdown("### Tools") # Removed icon
    st.markdown("---")

    # Ensure tool manager is initialized
    if 'tool_manager' not in st.session_state: st.session_state.tool_manager = ToolManager()
    all_tool_names = st.session_state.tool_manager.get_tool_names()
    all_tools_dict = {tool.name: tool for tool in st.session_state.tool_manager.get_tools()}

    if 'selected_tools' not in st.session_state: st.session_state.selected_tools = ['calculator'] # Default

    currently_selected_in_ui = []
    tool_search = st.text_input('Filter Tools', placeholder='Filter by name or description', key='tool_search')
    filtered_tool_names = all_tool_names
    if tool_search:
        search_lower = tool_search.lower()
        filtered_tool_names = [name for name in all_tool_names if search_lower in name.lower() or (hasattr(all_tools_dict[name], 'description') and search_lower in all_tools_dict[name].description.lower())]

    st.markdown(f"**Available Tools ({len(filtered_tool_names)}/{len(all_tool_names)})**")
    cols = st.columns(3)
    for i, tool_name in enumerate(filtered_tool_names):
        tool = all_tools_dict.get(tool_name)
        if not tool: continue # Skip if tool not found
        col = cols[i % 3]
        with col:
            card = st.container(border=True)
            with card:
                checkbox_key = f"tool_checkbox_{tool_name}"
                default_value = tool_name in st.session_state.selected_tools
                # Checkbox label without icon
                is_selected = st.checkbox(f"{tool.title}", value=default_value, key=checkbox_key, help=tool.description)
                st.caption(f"({tool.name})") # Show internal name
                if is_selected:
                    currently_selected_in_ui.append(tool_name)
                    if hasattr(tool, '_ui') and callable(tool._ui): tool._ui()

    # Update selected tools state if changes were made
    if set(st.session_state.selected_tools) != set(currently_selected_in_ui):
         st.session_state.selected_tools = currently_selected_in_ui
         st.session_state.tools = st.session_state.tool_manager.get_selected_tools(st.session_state.selected_tools)
         st.experimental_rerun() # Rerun to update agent / info display

    # --- Create New Tool Section ---
    st.markdown("---")
    st.markdown("### Create New Tool")
    with st.expander("Add a Custom Tool (Python Function)"):
        new_tool_name = st.text_input("Tool File Name (e.g., `my_tool`)", key="new_tool_filename")
        new_tool_function = st.text_area("Tool Code (Python Function)", height=300, key="new_tool_code", placeholder="# Define function with docstring here...") # Simplified placeholder
        if st.button("Create Tool", key="create_tool_button"):
            if new_tool_name and new_tool_function:
                 if not new_tool_name.isidentifier(): st.error("Tool file name must be a valid Python identifier.")
                 else:
                    validation_result, has_doc, func_name = evaluate_function_string(new_tool_function)
                    if validation_result is True and has_doc is True:
                        # Use TOOLS_DIR consistently
                        ctools_dir = os.path.join(TOOLS_DIR, "custom_tools")
                        os.makedirs(ctools_dir, exist_ok=True)
                        tool_filename = f"{new_tool_name}.py" if not new_tool_name.endswith(".py") else new_tool_name
                        tool_path = os.path.join(ctools_dir, tool_filename)
                        try:
                             with open(tool_path, "w") as f: f.write(new_tool_function) # Use default encoding
                             st.success(f"Tool '{func_name}' saved as `{tool_filename}`!");
                             st.session_state.new_tool_filename = ""; st.session_state.new_tool_code = ""
                             # Reload tool manager state and rerun
                             st.session_state.tool_manager = ToolManager()
                             st.rerun()
                        except Exception as e: st.error(f"Error saving tool file: {e}")
                    else: # Handle validation errors
                        if validation_result is not True: st.error(f"Error in function code: {validation_result}")
                        if not has_doc: st.error("Function must include a docstring.")
            else: st.error("Please provide both file name and code.")

# --- Settings Page Removed ---
# def settings_page(): ...

def info_page():
    st.markdown("### About Omega") # Removed icon, updated name
    st.markdown("---")
    # Updated Markdown content (same as previous good version)
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
    *Session data is stored locally in the `omega_data` directory.*
    *Custom tools are located in the `omega_tools/custom_tools` directory.*
    *File operations use the system's default text encoding.*
    """) # Added note about encoding

# =============================================
# MAIN APPLICATION SETUP
# =============================================

# def option_menu_cb(): ... # Not needed if using buttons or default option_menu behaviour

def menusetup():
    """Set up the navigation menu (No Settings, No Icons)"""
    list_menu = ["Chat", "Tools", "Info"] # Removed "Settings"
    list_pages = [chat_page, tools_page, info_page] # Removed settings_page
    st.session_state.dictpages = dict(zip(list_menu, list_pages))
    # list_icons removed

    # --- Sticky Tabs CSS ---
    st.markdown("""
    <style>
        /* Target common Streamlit structure for sticky */
        div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stHorizontalBlock"] {
            position: sticky; top: 0; z-index: 999; background-color: white;
            padding: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;
        }
        /* Hide default header if using sticky element */
        .stApp > header { display: none; }
    </style>
    """, unsafe_allow_html=True)

    # --- Navigation Menu ---
    try: # Prioritize streamlit-option-menu if installed
        from streamlit_option_menu import option_menu
        if 'selected_page' not in st.session_state or st.session_state.selected_page not in list_menu:
             st.session_state.selected_page = "Chat" # Default to Chat

        selected = option_menu(
            None, list_menu,
            # icons=list_icons, # Icons parameter removed
            default_index=list_menu.index(st.session_state.selected_page),
            orientation="horizontal",
            key='menu_opt', # Consistent key
            styles={ # Basic styling without icons
                 "container": {"padding": "5px!important", "background-color": "#fafafa"},
                 "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
                 "nav-link-selected": {"background-color": "#02ab21", "font-weight": "bold"},
            }
        )
        if selected and selected != st.session_state.selected_page:
            st.session_state.selected_page = selected
            st.session_state.editing_session_id = None # Cancel edit on tab switch
            st.rerun()
    except ImportError: # Fallback to buttons
        st.warning("Optional: `pip install streamlit-option-menu` for better navigation tabs.")
        cols = st.columns(len(list_menu))
        for i, page_name in enumerate(list_menu):
             with cols[i]:
                 button_type = "primary" if st.session_state.selected_page == page_name else "secondary"
                 if st.button(page_name, use_container_width=True, type=button_type, key=f"nav_{page_name}"):
                     if st.session_state.selected_page != page_name:
                         st.session_state.selected_page = page_name
                         st.session_state.editing_session_id = None
                         st.rerun()

def pageselection():
    """Call the appropriate page function based on selection"""
    page_func = st.session_state.dictpages.get(st.session_state.selected_page)
    if page_func: page_func()
    else: # Fallback if page state is invalid
         st.error("Invalid page selected. Defaulting to Chat.")
         st.session_state.selected_page = "Chat"; st.rerun()

def ensure_session_state():
    """Initialize session state variables"""
    if "storage" not in st.session_state:
        st.session_state.storage = CSVStorage() # Initialize storage first

    if "session_id" not in st.session_state or st.session_state.session_id is None:
        all_sessions = st.session_state.storage.get_all_sessions() # Use stored sessions
        if all_sessions:
             # Try sorting (might fail if IDs aren't consistently formatted)
             try: sorted_ids = sorted(all_sessions, reverse=True); st.session_state.session_id = sorted_ids[0]
             except: st.session_state.session_id = all_sessions[0] # Fallback to first found
        else: # Create new if none exist
             st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
             st.session_state.storage.save_session_details(st.session_state.session_id, st.session_state.session_id, False)

    # --- Removed states related to Settings ---
    # if "model" not in st.session_state: st.session_state.model = DEFAULT_MODEL
    # if "prefix" / "suffix" not in st.session_state: ...
    # if "doc_manager" not in st.session_state: ...

    if "tool_manager" not in st.session_state: st.session_state.tool_manager = ToolManager()
    if "initial_tools" not in st.session_state: st.session_state.initial_tools = ['calculator'] # Default tool
    if "selected_tools" not in st.session_state: st.session_state.selected_tools = st.session_state.initial_tools
    if "tools" not in st.session_state: st.session_state.tools = st.session_state.tool_manager.get_selected_tools(st.session_state.selected_tools)
    # clicked_cards seems redundant if checkbox logic is handled correctly, remove for now
    # if "clicked_cards" not in st.session_state: ...

    if "memory" not in st.session_state: st.session_state.memory = AgentMemory()
    if "selected_page" not in st.session_state: st.session_state.selected_page = "Chat" # Default page
    if "editing_session_id" not in st.session_state: st.session_state.editing_session_id = None # For renaming UI state

def main():
    """Main application entry point"""
    # Set page config - Use Omega Branding
    icon_path = "appicon.ico"
    im = None
    try: # Try loading icon safely
        if os.path.exists(icon_path): im = Image.open(icon_path)
    except Exception as e: print(f"Warning: Could not load icon {icon_path}: {e}")

    st.set_page_config(
        page_title="Omega AI Agent", # Renamed
        page_icon=im, # Use loaded icon or None
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={ # Updated links/about text
            'Get Help': '[https://github.com/your-repo/omega](https://github.com/your-repo/omega)',
            'Report a bug': "[https://github.com/your-repo/omega/issues](https://github.com/your-repo/omega/issues)",
            'About': "# Omega: Advanced Multi-Network AI Agent"
        }
    )

    # Initialize session state
    ensure_session_state()

    # Ensure directories exist (ToolManager also does this)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(TOOLS_DIR, "custom_tools"), exist_ok=True)

    # --- Main UI Layout ---
    # Title removed from here, handled in Info page
    # st.title("Ω Omega AI Agent")

    sidebar() # Render sidebar first

    # Main content area
    main_col, _ = st.columns([3, 1]) # Give main content more space
    with main_col:
        menusetup() # Render navigation tabs
        pageselection() # Render the selected page

if __name__ == "__main__":
    main()
