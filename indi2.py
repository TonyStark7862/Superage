import streamlit as st
import os
import inspect
import ast
import csv
import json
from datetime import datetime
import pandas as pd
from typing import List, Dict, Union, Callable, Any
import importlib.util
import sys
from PIL import Image
import tempfile
import shutil
import time

# Config Constants
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS = ['abc_model']
MAX_ITERATIONS = 20

# =============================================
# STORAGE MANAGEMENT
# =============================================

class CSVStorage:
    def __init__(self, csv_path=None):
        if csv_path is None:
            self.csv_path = os.path.join(BASE_DIR, "chat_history.csv")
        else:
            self.csv_path = csv_path
            
        self.sessions_path = os.path.join(BASE_DIR, "sessions.csv")
        
        # Initialize CSV files if they don't exist
        self._initialize_csv_files()
    
    def _initialize_csv_files(self):
        # Create chat history CSV if doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["session_id", "timestamp", "role", "content", "session_name", "favorite"])
        
        # Create sessions CSV if doesn't exist
        if not os.path.exists(self.sessions_path):
            with open(self.sessions_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["session_id", "session_name", "favorite", "create_time"])
    
    def save_chat_message(self, session_id, role, content, session_name=None):
        if session_name is None:
            session_name = session_id
            
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([session_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), role, content, session_name, False])
    
    def get_chat_history(self, session_id):
        if not os.path.exists(self.csv_path):
            return []
            
        messages = []
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["session_id"] == session_id:
                    messages.append({"role": row["role"], "content": row["content"]})
        return messages
    
    def get_all_sessions(self):
        if not os.path.exists(self.sessions_path):
            return []
            
        sessions = []
        with open(self.sessions_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sessions.append(row)
        
        # Sort by timestamp (newest first)
        sessions.sort(key=lambda x: x.get("create_time", ""), reverse=True)
        return [session["session_id"] for session in sessions]
    
    def get_all_sessions_data(self):
        if not os.path.exists(self.sessions_path):
            return []
            
        sessions = []
        with open(self.sessions_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sessions.append(row)
        
        # Sort by timestamp (newest first)
        sessions.sort(key=lambda x: x.get("create_time", ""), reverse=True)
        return sessions
    
    def get_all_sessions_names(self):
        if not os.path.exists(self.sessions_path):
            return {}
            
        session_names = {}
        with open(self.sessions_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "session_id" in row and "session_name" in row:
                    session_names[row["session_id"]] = row["session_name"]
        return session_names
    
    def get_favorites(self):
        if not os.path.exists(self.sessions_path):
            return {}
            
        favorites = {}
        with open(self.sessions_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "session_id" in row and "favorite" in row:
                    favorites[row["session_id"]] = row["favorite"] == "True"
        return favorites
    
    def create_new_session(self, session_name=None):
        session_id = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        if session_name is None:
            session_name = f"Chat {session_id}"
        
        # Add to sessions file
        with open(self.sessions_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([session_id, session_name, False, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        
        return session_id
    
    def save_session_name(self, session_id, session_name):
        # Read existing sessions
        sessions = []
        with open(self.sessions_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["session_id"] == session_id:
                    row["session_name"] = session_name
                sessions.append(row)
        
        # Write back all sessions
        with open(self.sessions_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["session_id", "session_name", "favorite", "create_time"])
            for session in sessions:
                writer.writerow([
                    session["session_id"], 
                    session["session_name"], 
                    session.get("favorite", False),
                    session.get("create_time", "")
                ])
    
    def toggle_favorite(self, session_id):
        # Read existing sessions
        sessions = []
        favorites = self.get_favorites()
        is_favorite = favorites.get(session_id, False)
        
        with open(self.sessions_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["session_id"] == session_id:
                    row["favorite"] = str(not is_favorite)
                sessions.append(row)
        
        # Write back all sessions
        with open(self.sessions_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["session_id", "session_name", "favorite", "create_time"])
            for session in sessions:
                writer.writerow([
                    session["session_id"], 
                    session["session_name"], 
                    session.get("favorite", False),
                    session.get("create_time", "")
                ])
        
        return not is_favorite
    
    def clear_chat(self, session_id):
        # Read all messages
        all_messages = []
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            all_messages = [row for row in reader if row["session_id"] != session_id]
        
        # Write back everything except the deleted session messages
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["session_id", "timestamp", "role", "content", "session_name", "favorite"])
            for msg in all_messages:
                writer.writerow([
                    msg["session_id"], 
                    msg.get("timestamp", ""), 
                    msg["role"], 
                    msg["content"], 
                    msg.get("session_name", ""),
                    msg.get("favorite", False)
                ])
    
    def delete_session(self, session_id):
        # Remove from chat history
        self.clear_chat(session_id)
        
        # Remove from sessions file
        sessions = []
        with open(self.sessions_path, 'r') as f:
            reader = csv.DictReader(f)
            sessions = [row for row in reader if row["session_id"] != session_id]
        
        with open(self.sessions_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["session_id", "session_name", "favorite", "create_time"])
            for session in sessions:
                writer.writerow([
                    session["session_id"], 
                    session["session_name"], 
                    session.get("favorite", False),
                    session.get("create_time", "")
                ])


class DocumentManager:
    def __init__(self):
        self.documents = {}
        self.database = None
    
    def add_document(self, name, content):
        self.documents[name] = content
    
    def list_documents(self):
        return list(self.documents.keys())
    
    def remove_document(self, name):
        if name in self.documents:
            del self.documents[name]
    
    def similarity_search(self, query, k=5):
        """Simple similarity search - just returns document chunks for now"""
        # In a real implementation, this would use embeddings to find similar documents
        results = []
        for doc_name, content in self.documents.items():
            results.append(SimpleDocument(f"Document: {doc_name}", content))
        return results[:k]


class SimpleDocument:
    def __init__(self, name, page_content):
        self.name = name
        self.page_content = page_content


# =============================================
# TOOLS SYSTEM
# =============================================

class BaseTool:
    name = 'Base_tool'
    link = 'https://github.com/example/simplified-omnitool'
    icon = '🧰'
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
                safe_dict[name] = getattr(math, name)
                
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"


class HelloWorldTool(BaseTool):
    name = 'print_hello_world'
    icon = '👋'
    title = 'Print Hello World'
    description = 'A simple function that returns the string "Hello World".'
    
    def _run(self, _):
        return "Hello World!"


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
    if len(function_node.body) > 0 and isinstance(function_node.body[0], ast.Expr) and isinstance(function_node.body[0].value, ast.Str):
        return True
    return False


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
        if not any(isinstance(node, ast.FunctionDef) for node in parsed_ast.body):
            return False, False, None

        function_node = next(node for node in parsed_ast.body if isinstance(node, ast.FunctionDef))
        
        # Extract tool name
        tool_name = function_node.name

        # Compiling the function string
        compiled_func = compile(func_str, '<string>', 'exec')
        exec(compiled_func, globals())

        # Check for docstring
        doc_exist = has_docstring(function_node)

        return True, doc_exist, tool_name

    except Exception as e:
        return e, False, None


def get_class_func_from_module(module):
    """Filter the classes and functions found inside a given module"""
    members = inspect.getmembers(module)
    # Filter and store only functions and classes defined inside module
    functions = []
    classes = []
    for name, member in members:
        if inspect.isfunction(member) and member.__module__ == module.__name__:
            if member.__doc__:  # Only include functions with docstrings
                functions.append((name, member))
            
        if inspect.isclass(member) and member.__module__ == module.__name__:
            classes.append((name, member))
        
    return classes, functions


def import_from_file(file_path, module_name=None):
    """Import a module from a file path"""
    if module_name is None:
        module_name = os.path.basename(file_path).replace(".py", "")
        
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        return None
        
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def monitor_folder(folder_path):
    """Monitor a folder path and return the list of modules from .py files inside"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    # Make sure folder is in path
    if folder_path not in sys.path:
        sys.modules.pop(folder_path, None)  # Remove existing module if any
        sys.path.append(folder_path)

    # List all .py files in the directory (excluding __init__.py)
    python_files = [f for f in os.listdir(folder_path) 
                    if f.endswith('.py') and f != "__init__.py"]

    # Dynamically import all modules
    monitored_modules = []
    for py_file in python_files:
        file_path = os.path.join(folder_path, py_file)
        try:
            # Clear module from cache if it exists
            module_name = py_file.replace(".py", "")
            if module_name in sys.modules:
                del sys.modules[module_name]
                
            module = import_from_file(file_path)
            if module:
                monitored_modules.append(module)
        except Exception as e:
            print(f"Error importing {py_file}: {e}")
            
    return monitored_modules


class ToolManager:
    def __init__(self):
        self.structured_tools = self.make_tools_list()
        self.tools_description = self.make_tools_description()
    
    def make_tools_description(self):
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
    
    def make_tools_list(self):
        """Build the list of available tools"""
        # Built-in tools
        base_tools = [CalculatorTool(), HelloWorldTool()]
        
        # Create tools directory if it doesn't exist
        tools_dir = os.path.join(BASE_DIR, "tools")
        if not os.path.exists(tools_dir):
            os.makedirs(tools_dir)
            
        # Monitor custom tools directory
        custom_tools_dir = os.path.join(tools_dir, "custom_tools")
        if not os.path.exists(custom_tools_dir):
            os.makedirs(custom_tools_dir)
            
        monitored_modules = monitor_folder(custom_tools_dir)
        
        # Process tool modules
        for module in monitored_modules:
            try:
                # Get classes and functions from the module
                classes, functions = get_class_func_from_module(module)
                
                # Add class-based tools
                for _, cls in classes:
                    if has_required_attributes(cls) and issubclass(cls, BaseTool):
                        base_tools.append(cls())
                
                # Add function-based tools
                for name, func in functions:
                    if callable(func) and func.__doc__:
                        # Create a tool from the function
                        func_tool = type(
                            f"{name}_Tool",
                            (BaseTool,),
                            {
                                "name": name,
                                "description": func.__doc__,
                                "icon": "🔧",
                                "title": name.replace("_", " ").title(),
                                "_run": lambda self, input_data, fn=func: str(fn(input_data))
                            }
                        )
                        base_tools.append(func_tool())
            except Exception as e:
                print(f"Error processing module: {e}")
                
        return base_tools
        
    def reload_tools(self):
        """Reload all tools"""
        self.structured_tools = self.make_tools_list()
        self.tools_description = self.make_tools_description()
        return self.structured_tools


# =============================================
# AGENT SYSTEM
# =============================================

class AgentMemory:
    def __init__(self):
        self.chat_memory = []
    
    def add_user_message(self, message):
        self.chat_memory.append({"role": "user", "content": message})
    
    def add_ai_message(self, message):
        self.chat_memory.append({"role": "assistant", "content": message})
    
    def clear(self):
        self.chat_memory = []


class SimpleAgent:
    def __init__(self, model, tools, memory):
        self.model = model
        self.tools = tools
        self.memory = memory
        self.max_iterations = MAX_ITERATIONS
    
    def run(self, input_text, callbacks=None):
        """Process the input and return a response"""
        # First, check if we should use a tool
        tool_result = None
        
        # Build full prompt with memory
        full_prompt = self._build_prompt(input_text)
        
        # Send to LLM
        response = self._call_abc_model(full_prompt)
        
        # Tool processing logic
        if any(f"I'll use the {tool.name} tool" in response for tool in self.tools):
            # Extract tool name and input
            for tool in self.tools:
                if f"I'll use the {tool.name} tool" in response:
                    # Simple way to extract tool input - could be more sophisticated
                    tool_input_marker = f"I'll use the {tool.name} tool with input:"
                    if tool_input_marker in response:
                        parts = response.split(tool_input_marker)
                        if len(parts) > 1:
                            tool_input = parts[1].strip()
                            tool_result = tool.run(tool_input)
                            
                            # Build a new prompt with the tool result
                            full_prompt = self._build_prompt(
                                input_text, 
                                f"I used the {tool.name} tool with input: {tool_input}\nResult: {tool_result}"
                            )
                            
                            # Get final response
                            response = self._call_abc_model(full_prompt)
        
        return response
    
    def _build_prompt(self, input_text, tool_info=None):
        """Build a full prompt including memory, tools info, and input"""
        # Format chat history
        history = ""
        for msg in self.memory.chat_memory:
            history += f"{msg['role'].title()}: {msg['content']}\n\n"
        
        # Format tools info
        tools_info = "You have access to the following tools:\n"
        for tool in self.tools:
            tools_info += f"- {tool.name}: {tool.description}\n"
        
        # Add guidance on tool usage
        tools_info += "\nWhen you need to use a tool, write: 'I'll use the [tool_name] tool with input: [tool_input]'"
        
        # Build the full prompt
        prompt = f"""
You are an AI assistant having a conversation with a human.

{tools_info}

Chat History:
{history}

"""
        
        # Add tool info if provided
        if tool_info:
            prompt += f"Tool usage information:\n{tool_info}\n\n"
        
        # Add the user's input
        prompt += f"User: {input_text}\nAssistant:"
        
        return prompt
    
    def _call_abc_model(self, prompt):
        """Call our custom model"""
        # This is where you'd call your abc_response function
        return abc_response(prompt)


# =============================================
# UI COMPONENTS
# =============================================

def replace_emojis_with_text(text, emoji_dict=None):
    """Replace emojis with text equivalents for Linux compatibility"""
    if emoji_dict is None:
        emoji_dict = {
            '🧮': '[CALC]',
            '👋': '[HELLO]',
            '🔧': '[TOOL]',
            '📊': '[CHART]',
            '📝': '[NOTE]',
            '💻': '[CODE]',
            '🔍': '[SEARCH]',
            '🔄': '[REFRESH]',
            '❌': '[DELETE]',
            '⭐': '[STAR]',
            '🧰': '[TOOLBOX]',
            '❓': '[HELP]',
            '📁': '[FILE]',
        }
    
    for emoji, replacement in emoji_dict.items():
        text = text.replace(emoji, replacement)
    
    return text


def get_platform_compatible_icon(icon):
    """Get a platform-compatible icon (text or emoji)"""
    # Check if we're on Linux
    is_linux = sys.platform.startswith('linux')
    
    # If on Linux, replace emojis with FontAwesome or text equivalents
    if is_linux:
        emoji_to_fa = {
            '🧮': 'calculator',
            '👋': 'hand-wave',
            '🔧': 'wrench',
            '📊': 'chart-bar',
            '📝': 'pen',
            '💻': 'laptop-code',
            '🔍': 'search',
            '🔄': 'sync',
            '❌': 'times',
            '⭐': 'star',
            '🧰': 'toolbox',
            '❓': 'question',
            '📁': 'folder',
        }
        
        return emoji_to_fa.get(icon, 'tools')
    
    return icon


def sidebar():
    with st.sidebar:
        st.write("## OmniTool")
        st.markdown("---")  # Separator
        
        st.write("### Chat Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("New Chat", use_container_width=True, type="primary"):
                # Create a new session
                new_session_id = st.session_state.storage.create_new_session("New Chat")
                st.session_state.session_id = new_session_id
                st.session_state.memory.clear()
                st.rerun()
        
        with col2:
            if st.button("Clear Chat", use_container_width=True):
                if "session_id" in st.session_state:
                    st.session_state.storage.clear_chat(st.session_state.session_id)
                    st.session_state.memory.clear()
                    st.rerun()
                    
        st.markdown("---")
        
        st.write("### Saved Chats")
        st.write("Previous chats will be automatically deleted after one week")
        
        # Get sessions data
        sessions = st.session_state.storage.get_all_sessions_data()
        favorites = st.session_state.storage.get_favorites()
        
        # Display favorites first
        if any(favorites.values()):
            st.write("#### Favorites")
            for session in sessions:
                session_id = session["session_id"]
                if favorites.get(session_id, False):
                    session_name = session["session_name"]
                    
                    col1, col2, col3 = st.columns([6, 1, 1])
                    with col1:
                        if st.button(f"⭐ {session_name}", key=f"fav_session_{session_id}", use_container_width=True):
                            st.session_state.session_id = session_id
                            st.session_state.memory.clear()  # Clear memory to reload from storage
                            st.rerun()
                    
                    with col2:
                        if st.button("⭐", key=f"unfav_{session_id}", help="Remove from favorites"):
                            st.session_state.storage.toggle_favorite(session_id)
                            st.rerun()
                    
                    with col3:
                        if st.button("🗑️", key=f"del_fav_{session_id}", help="Delete chat"):
                            st.session_state.storage.delete_session(session_id)
                            # If current session was deleted, create a new one
                            if st.session_state.session_id == session_id:
                                new_session_id = st.session_state.storage.create_new_session("New Chat")
                                st.session_state.session_id = new_session_id
                                st.session_state.memory.clear()
                            st.rerun()
        
        # Display regular sessions
        st.write("#### Recent Chats")
        for session in sessions:
            session_id = session["session_id"]
            if not favorites.get(session_id, False):
                session_name = session["session_name"]
                
                col1, col2, col3 = st.columns([6, 1, 1])
                with col1:
                    if st.button(f"{session_name}", key=f"session_{session_id}", use_container_width=True):
                        st.session_state.session_id = session_id
                        st.session_state.memory.clear()  # Clear memory to reload from storage
                        st.rerun()
                
                with col2:
                    if st.button("☆", key=f"fav_{session_id}", help="Add to favorites"):
                        st.session_state.storage.toggle_favorite(session_id)
                        st.rerun()
                
                with col3:
                    if st.button("🗑️", key=f"del_{session_id}", help="Delete chat"):
                        st.session_state.storage.delete_session(session_id)
                        # If current session was deleted, create a new one
                        if st.session_state.session_id == session_id:
                            new_session_id = st.session_state.storage.create_new_session("New Chat")
                            st.session_state.session_id = new_session_id
                            st.session_state.memory.clear()
                        st.rerun()


def rename_session_callback(new_name):
    if new_name:
        st.session_state.storage.save_session_name(st.session_state.session_id, new_name)
        st.success(f"Chat renamed to {new_name}")
        time.sleep(0.5)
        st.rerun()


def chat_page():
    # Check if we have a session
    if "session_id" not in st.session_state:
        new_session_id = st.session_state.storage.create_new_session("New Chat")
        st.session_state.session_id = new_session_id
    
    # Get session name
    session_names = st.session_state.storage.get_all_sessions_names()
    current_session_name = session_names.get(st.session_state.session_id, "Chat")
    
    # Header with rename option
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown(f"## {current_session_name}")
    
    with col2:
        if st.button("Rename", type="secondary", use_container_width=True):
            st.session_state.renaming_session = True
    
    if st.session_state.get("renaming_session", False):
        new_name = st.text_input("New name", value=current_session_name)
        col1, col2 = st.columns([6, 1])
        with col1:
            if st.button("Save", type="primary", use_container_width=True):
                rename_session_callback(new_name)
                st.session_state.renaming_session = False
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.renaming_session = False
    
    # Show selected tools
    selected_tools = [tool.name for tool in st.session_state.tools]
    if selected_tools:
        st.info(f"Selected tools: {', '.join(selected_tools)}")
    
    # Initialize agent if needed
    if "agent_instance" not in st.session_state:
        st.session_state.agent_instance = SimpleAgent(
            st.session_state.model,
            st.session_state.tools,
            st.session_state.memory
        )
    
    # Load chat history
    messages = st.session_state.storage.get_chat_history(st.session_state.session_id)
    
    # Initialize memory from storage if needed
    if not st.session_state.memory.chat_memory:
        for msg in messages:
            if msg["role"] == "user":
                st.session_state.memory.add_user_message(msg["content"])
            else:
                st.session_state.memory.add_ai_message(msg["content"])
    
    # Display messages
    for msg in messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])
    
    # Input
    if prompt := st.chat_input("Type your message here..."):
        # Search documents if database exists
        if "database" in st.session_state and st.session_state.database:
            # Do a similarity search in the loaded documents with the user's input
            similar_docs = st.session_state.database.similarity_search(prompt)
            # Insert the content of the most similar document into the prompt
            if similar_docs:
                prompt = f"Relevant documentation:\n{similar_docs[0].page_content}\n\nUser prompt:\n{prompt}"
        
        # Add prefix and suffix
        original_prompt = prompt
        prompt = f"{st.session_state.prefix}{prompt}{st.session_state.suffix}"
        
        # Display user message
        st.chat_message("user").write(original_prompt)
        
        # Add to memory
        st.session_state.memory.add_user_message(original_prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent_instance.run(prompt)
            st.write(response)
        
        # Add to memory
        st.session_state.memory.add_ai_message(response)
        
        # Save to storage
        session_name = st.session_state.storage.get_all_sessions_names().get(st.session_state.session_id, st.session_state.session_id)
        st.session_state.storage.save_chat_message(st.session_state.session_id, "user", original_prompt, session_name)
        st.session_state.storage.save_chat_message(st.session_state.session_id, "assistant", response, session_name)


def tools_page():
    st.markdown("## Tools")
    
    # Tool selection
    st.session_state.selected_tools = []
    
    # Search bar
    search_term = st.text_input("🔍 Search tools", placeholder="Search tools by name or description")
    
    # Filter tools based on search
    if search_term:
        st.session_state.tool_filtered = [tool for tool in st.session_state.tool_manager.get_tool_names() 
                                        if search_term.lower() in tool.lower() or 
                                          search_term.lower() in st.session_state.tool_manager.tools_description[tool].lower()]
    else:
        st.session_state.tool_filtered = st.session_state.tool_manager.get_tool_names()
    
    # Display tools as cards in a grid layout with consistent sizing
    st.markdown("### Available Tools")
    
    # Create columns for tools grid
    tool_count = len(st.session_state.tool_filtered)
    if tool_count > 0:
        cols = st.columns(min(3, tool_count))
        
        for i, tool_name in enumerate(st.session_state.tool_filtered):
            tool = st.session_state.tool_manager.get_selected_tools([tool_name])[0]
            col_idx = i % len(cols)
            
            with cols[col_idx]:
                with st.container(border=True, height=180):  # Fixed height for consistent card sizing
                    is_selected = st.checkbox(
                        f"{get_platform_compatible_icon(tool.icon)} {tool.title}", 
                        value=tool_name in st.session_state.clicked_cards,
                        key=f"tool_{tool_name}"
                    )
                    
                    # Description with limited height
                    st.markdown(f"<div style='height: 100px; overflow-y: auto;'><small>{tool.description}</small></div>", unsafe_allow_html=True)
                    
                    if is_selected:
                        st.session_state.clicked_cards[tool_name] = True
                        st.session_state.selected_tools.append(tool_name)
                    else:
                        st.session_state.clicked_cards[tool_name] = False
    else:
        st.info("No tools found matching your search criteria.")
    
    # Update the tools in session state
    st.session_state.tools = st.session_state.tool_manager.get_selected_tools(st.session_state.selected_tools)
    
    # Add new tool section
    st.markdown("---")
    st.markdown("## Create New Tool")
    
    new_tool_name = st.text_input("Tool filename (without .py extension)")
    
    new_tool_function = st.text_area(
        "Tool code", 
        height=250,
        placeholder="""def my_tool(input_data):
    \"\"\"Description of what this tool does and how to use it\"\"\"
    # Tool implementation here
    return f"Result: {input_data}"
"""
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        create_tool = st.button("Create Tool", type="primary", use_container_width=True)
    
    if create_tool:
        if new_tool_name and new_tool_function:
            # Validate function
            runs, has_doc, func_name = evaluate_function_string(new_tool_function)
            
            if runs is True and has_doc is True:
                # Create the tools directory if it doesn't exist
                tools_dir = os.path.join(BASE_DIR, "tools", "custom_tools")
                os.makedirs(tools_dir, exist_ok=True)
                
                # Save the tool
                tool_path = os.path.join(tools_dir, f"{new_tool_name}.py")
                with open(tool_path, "w") as f:
                    f.write(new_tool_function)
                
                # Force clean tool cache for reload
                import importlib
                if new_tool_name in sys.modules:
                    del sys.modules[new_tool_name]
                
                # Reload tool manager
                st.session_state.tool_manager.reload_tools()
                st.session_state.tool_list = st.session_state.tool_manager.structured_tools
                
                st.success(f"Tool '{new_tool_name}' created successfully!")
                time.sleep(0.5)  # Brief pause for feedback
                st.rerun()
            else:
                if not runs is True:
                    st.error(f"Error in function: {runs}")
                if not has_doc is True:
                    st.error("Function must have a docstring.")
        else:
            st.error("Please provide both a name and function code for the tool.")
    
    # Display information about created tools
    custom_tools_dir = os.path.join(BASE_DIR, "tools", "custom_tools")
    if os.path.exists(custom_tools_dir):
        custom_tools = [f for f in os.listdir(custom_tools_dir) if f.endswith('.py')]
        if custom_tools:
            st.markdown("### Custom Tools")
            for tool_file in custom_tools:
                st.markdown(f"- **{tool_file}**")


def info_page():
    st.markdown("## About OmniTool")
    st.markdown("---")
    
    st.markdown("""
    # Simplified OmniTool
    
    This is a streamlined version of OmniTool, a framework for creating customizable 
    chatbot interfaces that can use language models and access different tools to perform tasks.
    
    ## Features
    
    - Custom language model integration with `abc_response(prompt)`
    - Tool selection and management
    - Session history management with favorites and organization
    - Document reference integration
    - Custom tool creation
    
    ## Usage Guidelines
    
    ### Select tools in the tools page:
    
    - Search for tools by name and description
    - Select tools you want to use
    - Create custom tools directly from the interface
    
    ### Chat with the AI in the chat page:
    
    - Start a new session or continue an existing one
    - The AI can use tools when needed
    - View chat history and manage sessions
    
    ## Creating Custom Tools
    
    You can create custom tools by:
    
    1. Going to the Tools page
    2. Entering a name for your tool file
    3. Writing a Python function with a proper docstring
    
    Example:
    ```python
    def weather_lookup(location):
        """Look up the weather for a specific location"""
        # Implementation would go here
        return f"The weather in {location} is sunny."
    ```
    """)


# =============================================
# MAIN APPLICATION SETUP
# =============================================

def setup_navigation():
    """Set up the navigation menu"""
    st.markdown(
        """
        <style>
        .stButton button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            font-weight: bold;
        }
        .stTextInput > div > div > input {
            border-radius: 5px;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    # Define the tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔧 Tools", "ℹ️ Info"])
    
    with tab1:
        if st.session_state.selected_page == "Chat":
            chat_page()
    
    with tab2:
        if st.session_state.selected_page == "Tools":
            tools_page()
    
    with tab3:
        if st.session_state.selected_page == "Info":
            info_page()
    
    # Set the active tab based on selected page
    active_tab = 0
    if st.session_state.selected_page == "Tools":
        active_tab = 1
    elif st.session_state.selected_page == "Info":
        active_tab = 2
    
    # Use JavaScript to select the active tab
    st.markdown(
        f"""
        <script>
            var tabs = window.parent.document.querySelectorAll('.stTabs button[role="tab"]');
            tabs[{active_tab}].click();
        </script>
        """,
        unsafe_allow_html=True
    )


def ensure_session_state():
    """Initialize session state variables"""
    if "session_id" not in st.session_state:
        # Create an initial session
        storage = CSVStorage()
        new_session_id = storage.create_new_session("New Chat")
        st.session_state.session_id = new_session_id
    
    if "model" not in st.session_state:
        st.session_state.model = "abc_model"
    
    if "tool_manager" not in st.session_state:
        st.session_state.tool_manager = ToolManager()
        st.session_state.tool_list = st.session_state.tool_manager.structured_tools
    
    if "initial_tools" not in st.session_state:
        st.session_state.initial_tools = ['calculator', 'print_hello_world']
    
    if "selected_tools" not in st.session_state:
        st.session_state.selected_tools = st.session_state.initial_tools
    
    if "tools" not in st.session_state:
        st.session_state.tools = st.session_state.tool_manager.get_selected_tools(st.session_state.initial_tools)
    
    if "clicked_cards" not in st.session_state:
        st.session_state.clicked_cards = {tool_name: True for tool_name in st.session_state.initial_tools}
    
    if "memory" not in st.session_state:
        st.session_state.memory = AgentMemory()
    
    if "doc_manager" not in st.session_state:
        st.session_state.doc_manager = DocumentManager()
    
    if "documents" not in st.session_state:
        st.session_state.documents = st.session_state.doc_manager.list_documents()
    
    if "database" not in st.session_state:
        st.session_state.database = st.session_state.doc_manager.database
    
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Chat"
    
    if "prefix" not in st.session_state:
        st.session_state.prefix = ''
    
    if "suffix" not in st.session_state:
        st.session_state.suffix = ''
    
    if "storage" not in st.session_state:
        st.session_state.storage = CSVStorage()
    
    if "renaming_session" not in st.session_state:
        st.session_state.renaming_session = False


def main():
    """Main application entry point"""
    # Set page config
    st.set_page_config(
        page_title="OmniTool",
        page_icon="🧰",
        initial_sidebar_state="expanded",
        layout="wide",
        menu_items={
            'About': "Simplified version of OmniTool for custom LLM integration"
        }
    )
    
    # Make sure directories exist
    os.makedirs(os.path.join(BASE_DIR, "tools", "custom_tools"), exist_ok=True)
    
    # Initialize session state
    ensure_session_state()
    
    # Set up UI
    sidebar()
    setup_navigation()


if __name__ == "__main__":
    main()
