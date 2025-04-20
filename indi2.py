# -*- coding: utf-8 -*- # Still good practice for source code encoding
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

# Config Constants
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_MODEL = 'abc_model'
MAX_ITERATIONS = 20

# =============================================
# STORAGE MANAGEMENT (No changes needed here)
# =============================================
class CSVStorage:
    def __init__(self, csv_path=None):
        if csv_path is None:
            data_dir = os.path.join(BASE_DIR, "omega_data")
            os.makedirs(data_dir, exist_ok=True)
            self.csv_path = os.path.join(data_dir, "chat_history.csv")
            self.sessions_path = os.path.join(data_dir, "sessions.csv")
        else:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            self.csv_path = csv_path
            self.sessions_path = os.path.join(os.path.dirname(csv_path), "sessions.csv")
        self._initialize_csv_files()

    def _initialize_csv_files(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["session_id", "timestamp", "role", "content", "session_name"])
        if not os.path.exists(self.sessions_path):
            with open(self.sessions_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["session_id", "session_name", "starred"])
        else:
            try: # Add try-except block for robustness
                with open(self.sessions_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header and "starred" not in header:
                        sessions_data = list(reader)
                if header and "starred" not in header:
                    with open(self.sessions_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(header + ["starred"])
                        for row in sessions_data:
                            writer.writerow(row + [0])
            except Exception as e:
                 st.error(f"Error checking/updating sessions file header: {e}")


    def save_chat_message(self, session_id, role, content, session_name=None):
        if session_name is None:
            session_name = self.get_session_details(session_id).get('name', session_id)
        try:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([session_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), role, content, session_name])
        except Exception as e:
             st.error(f"Error saving chat message: {e}")

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
        if not os.path.exists(self.sessions_path):
            return {}
        session_details = {}
        try:
            with open(self.sessions_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "session_id" in row and "session_name" in row:
                        starred_val = row.get("starred", "0")
                        try:
                            is_starred = bool(int(starred_val))
                        except (ValueError, TypeError):
                            is_starred = False
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
        all_details = self.get_all_session_details()
        return all_details.get(session_id, {"name": session_id, "starred": False})


    def save_session_details(self, session_id, session_name, starred):
        all_details = self.get_all_session_details()
        all_details[session_id] = {"name": session_name, "starred": starred}
        self._write_sessions_file(all_details)

    def toggle_session_star(self, session_id):
        all_details = self.get_all_session_details()
        if session_id in all_details:
            all_details[session_id]['starred'] = not all_details[session_id]['starred']
            self._write_sessions_file(all_details)
        else:
             if session_id in self.get_all_sessions():
                 self.save_session_details(session_id, session_id, True)

    def delete_session(self, session_id):
        # Delete from chat history
        all_messages = []
        fieldnames = ["session_id", "timestamp", "role", "content", "session_name"] # Default header
        if os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames: # Get actual header if file exists and has content
                        fieldnames = reader.fieldnames
                    all_messages = [row for row in reader if row.get("session_id") != session_id]

                with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_messages)
            except Exception as e:
                 st.error(f"Error updating chat history after delete: {e}")

        # Delete from sessions file
        all_details = self.get_all_session_details()
        if session_id in all_details:
            del all_details[session_id]
            self._write_sessions_file(all_details)

    def _write_sessions_file(self, session_details_dict):
        try:
            with open(self.sessions_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["session_id", "session_name", "starred"])
                for s_id, details in session_details_dict.items():
                    writer.writerow([s_id, details.get("name", s_id), 1 if details.get("starred", False) else 0])
        except Exception as e:
             st.error(f"Error writing sessions file: {e}")


# =============================================
# TOOLS SYSTEM (Removed icon attribute)
# =============================================

class BaseTool:
    name = 'Base_tool'
    link = 'https://github.com/your-repo/omega'
    # icon = '🔧' # Removed icon
    description = 'Base tool description'
    # --- Add title attribute for consistency ---
    title = 'Base Tool'

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        # --- Set title based on name if not provided ---
        if not hasattr(self, 'title'):
             self.title = self.name.replace("_", " ").title()

    def run(self, input_data):
        return self._run(input_data)

    def _run(self, input_data):
        print(f'Running base tool with input: {input_data}')
        return f'Success: {input_data}'

    def _ui(self):
        pass

class CalculatorTool(BaseTool):
    name = 'calculator'
    # icon = '🧮' # Removed icon
    title = 'Calculator' # Title remains
    description = 'Perform arithmetic calculations. Input format: mathematical expression (e.g., "2 + 2", "sin(30)", "sqrt(16)")'

    def _run(self, expression):
        try:
            safe_dict = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'len': len, 'pow': pow, 'int': int, 'float': float,
                'str': str, 'sorted': sorted, 'list': list, 'dict': dict,
                'set': set, 'tuple': tuple, 'range': range
            }
            import math
            for name in dir(math):
                 if not name.startswith("_"):
                    safe_dict[name] = getattr(math, name)
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

# --- Tool Helper Functions (Unchanged) ---
def has_required_attributes(cls):
    required_attributes = ['name', 'description']
    try:
        for attr in required_attributes:
            if not hasattr(cls, attr):
                return False
        return True
    except Exception:
        return False

def has_docstring(function_node):
    return (len(function_node.body) > 0 and
            isinstance(function_node.body[0], ast.Expr) and
            isinstance(function_node.body[0].value, ast.Str))


def evaluate_function_string(func_str):
    try:
        parsed_ast = ast.parse(func_str)
        function_node = next((node for node in parsed_ast.body if isinstance(node, ast.FunctionDef)), None)
        if not function_node:
             return False, False, None
        tool_name = function_node.name
        temp_globals = {}
        compiled_func = compile(func_str, '<string>', 'exec')
        exec(compiled_func, temp_globals)
        doc_exist = has_docstring(function_node)
        if tool_name not in temp_globals or not callable(temp_globals[tool_name]):
             raise ValueError(f"Function '{tool_name}' not found or not callable after execution.")
        return True, doc_exist, tool_name
    except Exception as e:
        return e, False, None


def get_class_func_from_module(module):
    members = inspect.getmembers(module)
    functions = []
    classes = []
    for name, member in members:
        try:
            if inspect.isfunction(member) and member.__module__ == module.__name__:
                if member.__doc__:
                    functions.append((name, member))
            if inspect.isclass(member) and member.__module__ == module.__name__:
                classes.append((name, member))
        except Exception:
             pass
    return classes, functions


def import_from_file(file_path, module_name=None):
    if module_name is None:
        module_name = os.path.basename(file_path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
         print(f"Error executing module {module_name} from {file_path}: {e}")
         if module_name in sys.modules:
             del sys.modules[module_name]
         return None
    return module


def monitor_folder(folder_path):
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
        except OSError as e:
             print(f"Error creating tools directory {folder_path}: {e}")
             return []
    if folder_path not in sys.path:
        sys.path.insert(0, folder_path)
    try:
        python_files = [f for f in os.listdir(folder_path)
                        if f.endswith('.py') and f != "__init__.py"]
    except FileNotFoundError:
         print(f"Tools directory not found during scan: {folder_path}")
         return []
    except Exception as e:
         print(f"Error listing files in tools directory {folder_path}: {e}")
         return []
    monitored_modules = []
    for py_file in python_files:
        file_path = os.path.join(folder_path, py_file)
        module_name = py_file[:-3]
        try:
            if module_name in sys.modules:
                 module = importlib.reload(sys.modules[module_name])
            else:
                 module = import_from_file(file_path, module_name)
            if module:
                monitored_modules.append(module)
        except Exception as e:
            st.warning(f"Error importing/reloading tool '{py_file}': {e}")
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
        base_tools = [CalculatorTool()]
        tools_root_dir = os.path.join(BASE_DIR, "omega_tools")
        custom_tools_dir = os.path.join(tools_root_dir, "custom_tools")
        os.makedirs(tools_root_dir, exist_ok=True)
        os.makedirs(custom_tools_dir, exist_ok=True)
        monitored_modules = monitor_folder(custom_tools_dir)
        processed_tool_names = {tool.name for tool in base_tools}

        for module in monitored_modules:
            try:
                classes, functions = get_class_func_from_module(module)
                # Add class-based tools
                for _, cls in classes:
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
                # Add function-based tools
                for name, func in functions:
                    if callable(func) and func.__doc__:
                         if name not in processed_tool_names:
                            # Create a tool from the function (No icon)
                            func_tool = type(
                                f"{name}_Tool",
                                (BaseTool,),
                                {
                                    "name": name,
                                    "description": func.__doc__.strip(),
                                    # "icon": "🛠️", # Removed icon
                                    "title": name.replace("_", " ").title(),
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
                st.warning(f"Error processing tool module {module.__name__}: {e}\n{traceback.format_exc()}")
        return base_tools


# =============================================
# AGENT SYSTEM (Unchanged)
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
        self.model = model
        self.tools = tools
        self.memory = memory
        self.max_iterations = MAX_ITERATIONS

    def run(self, input_text):
        self.memory.add_user_message(input_text)
        full_prompt = self._build_prompt()
        response = self._call_omega_model(full_prompt)
        self.memory.add_ai_message(response)
        return response

    def _build_prompt(self):
        history = ""
        for msg in self.memory.get_memory():
            history += f"{msg['role'].title()}: {msg['content']}\n\n"
        tools_prompt_part = ""
        if self.tools:
            tools_prompt_part = "You have access to the following tools:\n"
            for tool in self.tools:
                tools_prompt_part += f"- {tool.name}: {tool.description}\n"
            tools_prompt_part += "\nWhen you need to use a tool, structure your thought process clearly.\n"
        prompt = f"""
You are Omega, a multi-network advanced AI agent. Engage in helpful and informative conversation.
{tools_prompt_part}
Conversation History:
{history}Assistant:"""
        return prompt.strip()

    def _call_omega_model(self, prompt):
        print("-" * 50)
        print(f"--- Sending Prompt to {self.model} ---")
        print(prompt)
        print("-" * 50)
        response = f"Omega Placeholder Response: Received prompt for model '{self.model}'. Processing..."
        if "calculate 2+2" in prompt.lower() and any(t.name == 'calculator' for t in self.tools):
             response += "\nThought: The user asked for a calculation. I have a calculator tool. I should use it.\nAction: I'll use the calculator tool with input: 2 + 2"
             tool_result = "[Simulated Calculator Result: 4]"
             response += f"\nObservation: {tool_result}\nFinal Answer: The result of 2 + 2 is 4."
        elif "calculate" in prompt.lower() and not any(t.name == 'calculator' for t in self.tools):
             response += "\nThought: The user asked for a calculation, but the calculator tool is not available. I cannot perform the calculation directly."
             response += "\nFinal Answer: I can't perform the calculation because the calculator tool isn't enabled."
        return response


# =============================================
# UI COMPONENTS (Modified for text buttons)
# =============================================

def sidebar():
    with st.sidebar:
        # Assuming icon file is okay, if not, remove this line
        # st.image("appicon.ico" if os.path.exists("appicon.ico") else "path/to/default/omega_logo.png", width=60)
        st.markdown("## Omega Sessions")
        st.markdown("---")

        # Removed icon from button text
        if st.button("Start New Session", use_container_width=True):
            new_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            st.session_state.session_id = new_session_id
            st.session_state.selected_page = 'Chat'
            st.session_state.editing_session_id = None
            st.session_state.storage.save_session_details(new_session_id, new_session_id, False)
            if "agent_instance" in st.session_state:
                 st.session_state.agent_instance.memory.clear()
            st.rerun()

        st.markdown("---")

        all_sessions_details = st.session_state.storage.get_all_session_details()
        session_ids = list(all_sessions_details.keys())
        session_ids.sort(key=lambda s_id: (
            not all_sessions_details.get(s_id, {}).get('starred', False),
            s_id
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

                # Adjust column widths for text buttons
                col1, col2, col3, col4 = st.columns([0.5, 0.17, 0.18, 0.15]) # Name | Edit | Star/Unstar | Delete

                with col1:
                    if is_editing:
                        new_name = st.text_input(
                            "New Name",
                            value=session_name,
                            key=f"rename_{session_id}",
                            label_visibility="collapsed",
                            placeholder="Enter new name and press Enter",
                            on_change=handle_rename_session,
                            args=(session_id, is_starred),
                        )
                    else:
                        button_type = "primary" if session_id == st.session_state.session_id else "secondary"
                        display_name = session_name if len(session_name) < 30 else session_name[:27] + "..."
                        if st.button(display_name, key=f"session_{session_id}", use_container_width=True, type=button_type):
                             if st.session_state.session_id != session_id:
                                st.session_state.session_id = session_id
                                st.session_state.editing_session_id = None
                                st.session_state.messages = st.session_state.storage.get_chat_history(session_id)
                                st.session_state.memory.clear()
                                for msg in st.session_state.messages:
                                     if msg['role'] == 'user':
                                         st.session_state.memory.add_user_message(msg['content'])
                                     elif msg['role'] == 'assistant':
                                          st.session_state.memory.add_ai_message(msg['content'])
                                st.rerun()

                with col2:
                    if not is_editing:
                        # Replaced icon with text "Edit"
                        if st.button("Edit", key=f"edit_{session_id}", help="Rename Session", use_container_width=True):
                             st.session_state.editing_session_id = session_id
                             st.rerun()

                with col3:
                     # Replaced icon with text "Star" / "Unstar"
                     star_text = "Unstar" if is_starred else "Star"
                     if st.button(star_text, key=f"star_{session_id}", help="Toggle Favorite", use_container_width=True):
                         st.session_state.storage.toggle_session_star(session_id)
                         st.session_state.editing_session_id = None
                         st.rerun()

                with col4:
                     # Replaced icon with text "Delete"
                     if st.button("Delete", key=f"delete_{session_id}", help="Delete Session", use_container_width=True):
                         st.session_state.storage.delete_session(session_id)
                         st.session_state.editing_session_id = None
                         if st.session_state.session_id == session_id:
                             remaining_sessions = st.session_state.storage.get_all_sessions()
                             st.session_state.session_id = remaining_sessions[0] if remaining_sessions else None
                             if st.session_state.session_id is None:
                                 st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                                 st.session_state.storage.save_session_details(st.session_state.session_id, st.session_state.session_id, False)
                         st.rerun()

def handle_rename_session(session_id, current_starred_status):
    new_name_key = f"rename_{session_id}"
    new_name = st.session_state.get(new_name_key)
    if new_name and new_name.strip(): # Ensure name is not empty or just whitespace
        st.session_state.storage.save_session_details(session_id, new_name.strip(), current_starred_status)
        st.session_state.editing_session_id = None
    elif new_name is not None: # If user cleared the input, revert edit mode without saving
         st.warning("Session name cannot be empty.")
         # Keep editing mode active so user can correct it
         # st.session_state.editing_session_id = None # Don't reset here


def chat_page():
    # Removed icon from header
    st.markdown("### Chat")
    st.markdown("---")

    if st.session_state.selected_tools:
        s = ', '.join(st.session_state.selected_tools)
        st.info(f'Active tools: {s}')
    else:
         st.info('No tools selected. Go to the Tools page to enable them.')

    current_tool_names = {tool.name for tool in st.session_state.tools}
    agent_tool_names = set()
    if "agent_instance" in st.session_state:
        agent_tool_names = {tool.name for tool in st.session_state.agent_instance.tools}

    if "agent_instance" not in st.session_state or current_tool_names != agent_tool_names:
        st.session_state.agent_instance = SimpleAgent(
            model=DEFAULT_MODEL,
            tools=st.session_state.tools,
            memory=st.session_state.memory
        )
        print("Agent instance created or updated with new tools.")

    if not st.session_state.memory.get_memory():
         st.session_state.messages = st.session_state.storage.get_chat_history(st.session_state.session_id)
         for msg in st.session_state.messages:
             if msg['role'] == 'user':
                 st.session_state.memory.add_user_message(msg['content'])
             elif msg['role'] == 'assistant':
                  st.session_state.memory.add_ai_message(msg['content'])
         print(f"Loaded {len(st.session_state.messages)} messages into memory for session {st.session_state.session_id}")

    current_messages = st.session_state.memory.get_memory()
    for msg in current_messages:
        with st.chat_message(msg["role"]):
             st.write(msg["content"])

    if prompt := st.chat_input("Enter your message to Omega..."):
        with st.chat_message("user"):
            st.write(prompt)
        original_prompt = prompt
        with st.chat_message("assistant"):
            with st.spinner("Omega is thinking..."):
                response = st.session_state.agent_instance.run(original_prompt)
            st.write(response)

        session_details = st.session_state.storage.get_session_details(st.session_state.session_id)
        session_name = session_details.get('name', st.session_state.session_id)
        st.session_state.storage.save_chat_message(st.session_state.session_id, "user", original_prompt, session_name)
        st.session_state.storage.save_chat_message(st.session_state.session_id, "assistant", response, session_name)


def tools_page():
    # Removed icon from header
    st.markdown("### Tools")
    st.markdown("---")

    st.session_state.tool_manager = ToolManager()
    all_tool_names = st.session_state.tool_manager.get_tool_names()
    all_tools_dict = {tool.name: tool for tool in st.session_state.tool_manager.get_tools()}

    if 'selected_tools' not in st.session_state:
         st.session_state.selected_tools = st.session_state.initial_tools
         st.session_state.clicked_cards = {tool_name: (tool_name in st.session_state.selected_tools) for tool_name in all_tool_names}

    currently_selected_in_ui = []
    tool_search = st.text_input('Filter Tools', placeholder='Filter by name or description', key='tool_search')
    filtered_tool_names = all_tool_names
    if tool_search:
        search_lower = tool_search.lower()
        filtered_tool_names = [
            name for name in all_tool_names
            if search_lower in name.lower() or
               (hasattr(all_tools_dict[name], 'description') and search_lower in all_tools_dict[name].description.lower())
        ]

    st.markdown(f"**Available Tools ({len(filtered_tool_names)}/{len(all_tool_names)})**")

    cols = st.columns(3)
    for i, tool_name in enumerate(filtered_tool_names):
        tool = all_tools_dict[tool_name]
        col = cols[i % 3]

        with col:
            card = st.container(border=True)
            with card:
                checkbox_key = f"tool_checkbox_{tool_name}"
                default_value = tool_name in st.session_state.selected_tools
                # Removed icon from checkbox label
                is_selected = st.checkbox(
                    f"{tool.title}", # Display tool title only
                    value=default_value,
                    key=checkbox_key,
                    help=tool.description
                )
                st.caption(f"({tool.name})")
                if is_selected:
                    currently_selected_in_ui.append(tool_name)
                    if hasattr(tool, '_ui') and callable(tool._ui):
                        tool._ui()

    if set(st.session_state.selected_tools) != set(currently_selected_in_ui):
         st.session_state.selected_tools = currently_selected_in_ui
         st.session_state.tools = st.session_state.tool_manager.get_selected_tools(st.session_state.selected_tools)
         st.experimental_rerun()


    st.markdown("---")
    st.markdown("### Create New Tool")
    with st.expander("Add a Custom Tool (Python Function)"):
        new_tool_name = st.text_input("Tool File Name (e.g., `my_weather_tool`)", key="new_tool_filename")
        new_tool_function = st.text_area(
            "Tool Code (Python Function)", height=300, key="new_tool_code",
            placeholder="""# Required: Define a function with a docstring.
# The function name will be the tool name.
# The docstring is the tool description.
# It must accept one argument (input_data) and return a string.

import requests

def get_weather(location: str) -> str:
    \"\"\"Fetches the current weather for a given location.
    Input should be the city name (e.g., 'London').\"\"\"
    try:
        api_key = "YOUR_API_KEY" # Consider how to handle secrets
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        description = data['weather'][0]['description']
        temp = data['main']['temp']
        return f"Current weather in {location}: {description}, {temp}°C"
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather: {e}"
    except Exception as e:
        return f"Error processing weather data: {str(e)}"
"""
        )
        if st.button("Create Tool", key="create_tool_button"):
            if new_tool_name and new_tool_function:
                 if not new_tool_name.isidentifier():
                     st.error("Tool file name must be a valid Python identifier.")
                 else:
                    validation_result, has_doc, func_name = evaluate_function_string(new_tool_function)
                    if validation_result is True and has_doc is True:
                        tools_dir = os.path.join(BASE_DIR, "omega_tools", "custom_tools")
                        os.makedirs(tools_dir, exist_ok=True)
                        tool_filename = f"{new_tool_name}.py" if not new_tool_name.endswith(".py") else new_tool_name
                        tool_path = os.path.join(tools_dir, tool_filename)
                        try:
                             with open(tool_path, "w", encoding='utf-8') as f:
                                 f.write(new_tool_function)
                             st.success(f"Tool '{func_name}' saved successfully as `{tool_filename}`!")
                             st.session_state.new_tool_filename = ""
                             st.session_state.new_tool_code = ""
                             st.rerun()
                        except OSError as e:
                             st.error(f"Error saving tool file: {e}")
                        except Exception as e:
                             st.error(f"An unexpected error occurred: {e}")
                    else:
                        if validation_result is not True:
                            st.error(f"Error in function code: {validation_result}")
                        if not has_doc:
                            st.error("Function definition must include a docstring.")
            else:
                st.error("Please provide both a valid file name and the Python function code.")


def info_page():
    # Removed icon from header
    st.markdown("### About Omega")
    st.markdown("---")
    st.markdown("""
    # Omega: Advanced Multi-Network AI Agent

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
    """)


# =============================================
# MAIN APPLICATION SETUP
# =============================================

def menusetup():
    """Set up the navigation menu"""
    list_menu = ["Chat", "Tools", "Info"]
    list_pages = [chat_page, tools_page, info_page]
    st.session_state.dictpages = dict(zip(list_menu, list_pages))
    # list_icons = ['chat-left-text-fill', 'tools', 'info-circle-fill'] # Icons removed

    # Sticky Tabs CSS (Kept as it's layout, not character based)
    st.markdown("""
    <style>
        div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stHorizontalBlock"] {
            position: sticky;
            top: 0;
            z-index: 999;
            background-color: white;
            padding-top: 10px;
            padding-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .stApp > header { display: none; }
        .nav-container { position: sticky; top: 0; z-index: 999; background-color: white; }
    </style>
    """, unsafe_allow_html=True)

    try:
        from streamlit_option_menu import option_menu
        if 'selected_page' not in st.session_state or st.session_state.selected_page not in list_menu:
             st.session_state.selected_page = "Chat"

        selected = option_menu(
            None, list_menu,
            # icons=list_icons,  # Icons parameter removed
            default_index=list_menu.index(st.session_state.selected_page),
            orientation="horizontal",
            key='menu_opt',
            styles={ # Adjusted styles slightly without icons
                 "container": {"padding": "5px!important", "background-color": "#fafafa"},
                 # "icon": {"color": "orange", "font-size": "18px"}, # Icon style removed
                 "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
                 "nav-link-selected": {"background-color": "#02ab21", "font-weight": "bold"},
            }
        )
        if selected and selected != st.session_state.selected_page:
            st.session_state.selected_page = selected
            st.session_state.editing_session_id = None
            st.rerun()

    except ImportError:
        st.warning("`streamlit-option-menu` not found. Using basic buttons for navigation.")
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
    page_func = st.session_state.dictpages.get(st.session_state.selected_page)
    if page_func:
         page_func()
    else:
         st.error("Selected page not found!")
         st.session_state.selected_page = "Chat"
         st.rerun()

def ensure_session_state():
    """Initialize session state variables"""
    if "session_id" not in st.session_state or st.session_state.session_id is None:
        storage = CSVStorage() # Create instance to access storage
        all_sessions = storage.get_all_session_details()
        if all_sessions:
             sorted_ids = sorted(all_sessions.keys(), reverse=True)
             st.session_state.session_id = sorted_ids[0]
        else:
             st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
             storage.save_session_details(st.session_state.session_id, st.session_state.session_id, False)

    if "tool_manager" not in st.session_state:
        st.session_state.tool_manager = ToolManager()
    if "initial_tools" not in st.session_state:
        st.session_state.initial_tools = ['calculator']
    if "selected_tools" not in st.session_state:
         st.session_state.selected_tools = st.session_state.initial_tools
    if "tools" not in st.session_state:
         st.session_state.tools = st.session_state.tool_manager.get_selected_tools(st.session_state.selected_tools)
    if "clicked_cards" not in st.session_state: # Still needed for checkbox state logic
         st.session_state.clicked_cards = {tool_name: (tool_name in st.session_state.selected_tools)
                                          for tool_name in st.session_state.tool_manager.get_tool_names()}
    if "memory" not in st.session_state:
        st.session_state.memory = AgentMemory()
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Chat"
    if "storage" not in st.session_state:
        st.session_state.storage = CSVStorage()
    if "editing_session_id" not in st.session_state:
         st.session_state.editing_session_id = None


def main():
    """Main application entry point"""
    icon_path = "appicon.ico"
    im = None
    if os.path.exists(icon_path):
        try:
            im = Image.open(icon_path)
        except Exception as e:
            print(f"Warning: Could not load icon {icon_path}: {e}")

    st.set_page_config(
        page_title="Omega AI Agent",
        page_icon=im, # Can be None if loading fails
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo/omega',
            'Report a bug': "https://github.com/your-repo/omega/issues",
            'About': "# Omega: Advanced Multi-Network AI Agent"
        }
    )
    ensure_session_state()
    tools_dir = os.path.join(BASE_DIR, "omega_tools", "custom_tools")
    os.makedirs(tools_dir, exist_ok=True)

    sidebar()
    main_col, _ = st.columns([3, 1])
    with main_col:
        menusetup()
        pageselection()

if __name__ == "__main__":
    main()
