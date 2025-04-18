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
import tempfile
import time

# Config Constants
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS = ['abc_model']  # Our custom model
MAX_ITERATIONS = 20

# Custom emoji map to ensure consistency across platforms
EMOJI_MAP = {
    "calculator": "🧮",
    "chat": "💬",
    "tools": "🔧",
    "settings": "⚙️",
    "info": "ℹ️",
    "star": "⭐",
    "trash": "🗑️",
    "new": "➕",
    "clear": "🔄",
    "save": "💾",
    "search": "🔍",
    "warning": "⚠️",
    "success": "✅",
    "error": "❌"
}

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
                writer.writerow(["session_id", "timestamp", "role", "content", "session_name"])
        
        # Create sessions CSV if doesn't exist
        if not os.path.exists(self.sessions_path):
            with open(self.sessions_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["session_id", "session_name"])
    
    def save_chat_message(self, session_id, role, content, session_name=None):
        if session_name is None:
            session_name = session_id
            
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([session_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), role, content, session_name])
    
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
        if not os.path.exists(self.csv_path):
            return []
            
        session_ids = set()
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                session_ids.add(row["session_id"])
        return list(session_ids)
    
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
    
    def save_session_name(self, session_id, session_name):
        # Read existing sessions
        session_names = self.get_all_sessions_names()
        session_names[session_id] = session_name
        
        # Write back all sessions
        with open(self.sessions_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["session_id", "session_name"])
            for s_id, s_name in session_names.items():
                writer.writerow([s_id, s_name])
    
    def delete_session(self, session_id):
        # Read all messages
        all_messages = []
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            all_messages = [row for row in reader if row["session_id"] != session_id]
        
        # Write back everything except the deleted session
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["session_id", "timestamp", "role", "content", "session_name"])
            for msg in all_messages:
                writer.writerow([
                    msg["session_id"], 
                    msg.get("timestamp", ""), 
                    msg["role"], 
                    msg["content"], 
                    msg.get("session_name", "")
                ])
        
        # Also remove from sessions file
        session_names = self.get_all_sessions_names()
        if session_id in session_names:
            del session_names[session_id]
            
            with open(self.sessions_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["session_id", "session_name"])
                for s_id, s_name in session_names.items():
                    writer.writerow([s_id, s_name])


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
    icon = '🔧'
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
    icon = EMOJI_MAP["calculator"]
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


class PrintHelloWorldTool(BaseTool):
    name = 'print_hello_world'
    icon = "💻"
    title = 'Print Hello World'
    description = 'A simple function that returns the string "Hello World".'
    
    def _run(self, input_data=None):
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
        sys.path.append(folder_path)

    # List all .py files in the directory (excluding __init__.py)
    python_files = [f for f in os.listdir(folder_path) 
                    if f.endswith('.py') and f != "__init__.py"]

    # Dynamically import all modules
    monitored_modules = []
    for py_file in python_files:
        file_path = os.path.join(folder_path, py_file)
        try:
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
        base_tools = [CalculatorTool()]
        
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
                                "icon": "🛠️",
                                "title": name.replace("_", " ").title(),
                                "_run": lambda self, input_data, fn=func: str(fn(input_data))
                            }
                        )
                        base_tools.append(func_tool())
            except Exception as e:
                print(f"Error processing module: {e}")
                
        return base_tools


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
        """Call our custom model - implementation would be provided separately"""
        # This is where you'd call your abc_response function
        # For now, we'll return a placeholder
        return "This would be a response from the abc_response(prompt) function. In a real implementation, you would call your model here."


# =============================================
# UI COMPONENTS
# =============================================

def sidebar():
    with st.sidebar:
        # Quick actions section
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"{EMOJI_MAP['new']} Start new chat", use_container_width=True):
                st.session_state.session_id = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                st.session_state.memory.clear()
                st.session_state.selected_page = 'Chat'
                # Set default name
                st.session_state.storage.save_session_name(
                    st.session_state.session_id, 
                    "Initial Basic Greeting"
                )
                st.rerun()
        
        with col2:
            if st.button(f"{EMOJI_MAP['clear']} Clear current chat", use_container_width=True):
                if "session_id" in st.session_state:
                    st.session_state.memory.clear()
                    st.session_state.storage.delete_session(st.session_state.session_id)
                    st.rerun()
        
        st.markdown("### Saved Chats")
        
        # Show the pinned chats first
        st.markdown("#### Previous Chats")
        st.markdown("*Deleted after one week*")
        
        session_id_list = st.session_state.storage.get_all_sessions()
        st.session_state.session_name = st.session_state.storage.get_all_sessions_names()
        
        # Add star for favorite sessions
        for session_id in reversed(session_id_list):
            session_name = st.session_state.session_name.get(session_id, session_id)
            
            # Create a container for each session with hover effect
            session_container = st.container()
            with session_container:
                col1, col2, col3 = st.columns([1, 5, 1])
                
                # Star icon for favorites
                with col1:
                    star_key = f"star_{session_id}"
                    if star_key not in st.session_state:
                        st.session_state[star_key] = False
                    
                    if st.button(
                        EMOJI_MAP["star"], 
                        key=star_key,
                        help="Mark as favorite"
                    ):
                        st.session_state[star_key] = not st.session_state[star_key]
                        st.rerun()
                
                # Session name/button
                with col2:
                    if st.button(session_name, key=f"session_{session_id}", use_container_width=True):
                        st.session_state.session_id = session_id
                        st.session_state.memory.clear()
                        # Load chat history into memory
                        messages = st.session_state.storage.get_chat_history(session_id)
                        for msg in messages:
                            if msg["role"] == "user":
                                st.session_state.memory.add_user_message(msg["content"])
                            else:
                                st.session_state.memory.add_ai_message(msg["content"])
                        st.session_state.selected_page = 'Chat'
                        st.rerun()
                
                # Delete button
                with col3:
                    if st.button(
                        EMOJI_MAP["trash"], 
                        key=f"delete_{session_id}",
                        help="Delete this chat"
                    ):
                        st.session_state.storage.delete_session(session_id)
                        st.rerun()
            
            # Rename input field - only show when clicked
            if f"rename_active_{session_id}" not in st.session_state:
                st.session_state[f"rename_active_{session_id}"] = False
                
            if st.session_state.get(f"rename_active_{session_id}", False):
                with session_container:
                    rename_col1, rename_col2 = st.columns([6, 1])
                    with rename_col1:
                        st.text_input(
                            "New name", 
                            key=f"new_name_{session_id}",
                            value=session_name,
                            label_visibility="collapsed"
                        )
                    with rename_col2:
                        if st.button("Save", key=f"save_rename_{session_id}"):
                            if st.session_state[f"new_name_{session_id}"]:
                                st.session_state.storage.save_session_name(
                                    session_id, 
                                    st.session_state[f"new_name_{session_id}"]
                                )
                            st.session_state[f"rename_active_{session_id}"] = False
                            st.rerun()
            else:
                # Show "Rename" option on right-click or long press (simulated with double click)
                if session_container.button(
                    "Rename", 
                    key=f"rename_btn_{session_id}",
                    help="Rename this chat"
                ):
                    st.session_state[f"rename_active_{session_id}"] = True
                    st.rerun()


def change_session_name(session_id):
    new_name_key = f'new_name_{session_id}'
    if new_name_key in st.session_state and st.session_state[new_name_key]:
        st.session_state.storage.save_session_name(session_id, st.session_state[new_name_key])


def chat_page():
    # Show selected tools
    tools_str = ', '.join(st.session_state.selected_tools)
    st.info(f'Selected tools: {tools_str}')
    
    # Initialize agent if needed
    if "agent_instance" not in st.session_state or st.session_state.agent_needs_update:
        st.session_state.agent_instance = SimpleAgent(
            st.session_state.model,
            st.session_state.tools,
            st.session_state.memory
        )
        st.session_state.agent_needs_update = False
    
    # Get chat history
    st.session_state.messages = st.session_state.storage.get_chat_history(st.session_state.session_id)
    
    # Display messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    # Input
    if prompt := st.chat_input("Type your message here..."):
        # Search documents if database exists
        if hasattr(st.session_state, "database") and st.session_state.database:
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
                # This is where your custom LLM function would be called
                response = st.session_state.agent_instance.run(prompt)
            st.write(response)
        
        # Add to memory
        st.session_state.memory.add_ai_message(response)
        
        # Save to storage
        session_name = st.session_state.session_name.get(st.session_state.session_id, st.session_state.session_id)
        st.session_state.storage.save_chat_message(st.session_state.session_id, "user", original_prompt, session_name)
        st.session_state.storage.save_chat_message(st.session_state.session_id, "assistant", response, session_name)
        st.session_state.storage.save_session_name(st.session_state.session_id, session_name)


def tools_page():
    # Tool selection
    st.session_state.selected_tools = []
    
    # Tool search with icon
    search_col1, search_col2 = st.columns([1, 6])
    with search_col1:
        st.markdown(f"### {EMOJI_MAP['search']}")
    with search_col2:
        tool_search = st.text_input('Search', placeholder='Search tools by name or description', 
                                  key='tool_search', label_visibility="collapsed")
    
    # Filter tools based on search
    if tool_search:
        st.session_state.tool_filtered = [tool for tool in st.session_state.tool_manager.get_tool_names() 
                                        if tool_search.lower() in tool.lower() or 
                                          tool_search.lower() in st.session_state.tool_manager.tools_description[tool].lower()]
    else:
        st.session_state.tool_filtered = st.session_state.tool_manager.get_tool_names()
    
    # Display tools as cards with consistent styling
    st.markdown("""
    <style>
    .tool-card {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #e0e0e0;
        height: 200px;
        overflow: hidden;
        position: relative;
        transition: all 0.3s ease;
    }
    .tool-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .tool-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .tool-title {
        font-size: 18px;
        font-weight: bold;
        margin: 0;
    }
    .tool-description {
        font-size: 14px;
        color: #555;
        margin-bottom: 10px;
        flex: 1;
        overflow: hidden;
    }
    .tool-checkbox {
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create a grid layout for tools
    cols = st.columns(3)
    
    for i, tool_name in enumerate(st.session_state.tool_filtered):
        tool = st.session_state.tool_manager.get_selected_tools([tool_name])[0]
        col = cols[i % 3]
        
        with col:
            # Create a custom styled card
            card = st.container()
            with card:
                # Use a custom HTML/CSS card for consistent styling
                card_html = f"""
                <div class="tool-card">
                    <div class="tool-header">
                        <h3 class="tool-title">{tool.icon} {tool.title}</h3>
                    </div>
                    <div class="tool-description">
                        {tool.description}
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Add the checkbox below the card for selection
                is_selected = st.checkbox(
                    "Select", 
                    value=tool_name in st.session_state.clicked_cards,
                    key=f"tool_{tool_name}"
                )
                
                if is_selected:
                    st.session_state.clicked_cards[tool_name] = True
                    st.session_state.selected_tools.append(tool_name)
                else:
                    st.session_state.clicked_cards[tool_name] = False
                
                # Call the tool's UI method if it exists
                if hasattr(tool, '_ui') and callable(tool._ui):
                    tool._ui()
    
    # Update the tools in session state
    st.session_state.tools = st.session_state.tool_manager.get_selected_tools(st.session_state.selected_tools)
    st.session_state.agent_needs_update = True
    
    # Create new tool section
    st.markdown("---")
    
    # Use columns for better layout
    tool_col1, tool_col2 = st.columns([1, 3])
    
    with tool_col1:
        st.subheader("Create New Tool")
    
    with tool_col2:
        new_tool_btn = st.button("+ Create Tool", key="create_tool_btn")
    
    if "show_tool_creator" not in st.session_state:
        st.session_state.show_tool_creator = False
        
    if new_tool_btn:
        st.session_state.show_tool_creator = True
    
    if st.session_state.show_tool_creator:
        # Tool creation form with improved styling
        with st.container(border=True):
            st.subheader("Create New Tool")
            
            new_tool_name = st.text_input("Tool file name", key="new_tool_name")
            
            # Code editor with syntax highlighting
            new_tool_function = st.text_area(
                "Tool code", 
                height=300,
                key="new_tool_code",
                placeholder="""def my_tool(input_data):
    \"\"\"Description of what this tool does and how to use it\"\"\"
    # Tool implementation here
    return f"Result: {input_data}"
"""
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                submit_btn = st.button("Create", use_container_width=True)
            with col2:
                cancel_btn = st.button("Cancel", use_container_width=True)
            
            if cancel_btn:
                st.session_state.show_tool_creator = False
                st.rerun()
                
            if submit_btn:
                if new_tool_name and new_tool_function:
                    # Show spinner while processing
                    with st.spinner("Creating tool..."):
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
                            
                            # Reinitialize tool manager without requiring full page refresh
                            st.session_state.tool_manager = ToolManager()
                            st.session_state.tool_list = st.session_state.tool_manager.structured_tools
                            
                            # Success message
                            st.success(f"Tool '{new_tool_name}' created successfully!")
                            
                            # Hide the tool creator form
                            st.session_state.show_tool_creator = False
                            
                            # Force the UI to update with the new tool
                            time.sleep(0.5)  # Small delay to ensure file is processed
                            st.rerun()
                        else:
                            if not runs is True:
                                st.error(f"Error in function: {runs}")
                            if not has_doc is True:
                                st.error("Function must have a docstring.")
                else:
                    st.error("Please provide both a name and function code for the tool.")


def settings_page():
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    
    # Model selection
    st.session_state.model = st.selectbox(
        "Select a model", 
        options=MODELS,
        index=MODELS.index(st.session_state.model)
    )
    
    # Prompt prefix and suffix
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.prefix = st.text_area(
            'Prefix',
            value=st.session_state.prefix,
            placeholder='Text to add before user input',
            height=150
        )
    
    with col2:
        st.session_state.suffix = st.text_area(
            'Suffix',
            value=st.session_state.suffix,
            placeholder='Text to add after user input',
            height=150
        )
    
    # Document management
    with st.expander("Document Management", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload Documents", 
            type=['txt', 'pdf'], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.type == 'text/plain':
                    # Read text file
                    content = uploaded_file.getvalue().decode('utf-8')
                    st.session_state.doc_manager.add_document(uploaded_file.name, content)
                
                elif uploaded_file.type == 'application/pdf':
                    # For PDFs, we'd use a PDF library in a real app
                    # Here we'll just store the name for demo purposes
                    st.session_state.doc_manager.add_document(
                        uploaded_file.name, 
                        f"[PDF Content from {uploaded_file.name}]"
                    )
            
            st.success("Documents loaded successfully.")
        
        if st.button("Clear Documents"):
            for doc_name in st.session_state.doc_manager.list_documents():
                st.session_state.doc_manager.remove_document(doc_name)
            st.success("All documents removed.")
        
        st.write("Loaded documents:")
        if st.session_state.doc_manager.list_documents():
            for doc in st.session_state.doc_manager.list_documents():
                st.write(f"- {doc}")
        else:
            st.write("No documents loaded.")


def info_page():
    st.markdown("""
    # Simplified OmniTool
    
    This is a streamlined version of OmniTool, a framework for creating customizable 
    chatbot interfaces that can use language models and access different tools to perform tasks.
    
    ## Features
    
    - Custom language model integration via `abc_response` function
    - Tool selection and management
    - Session history management with CSV storage
    - Custom tool creation
    
    ## Usage Guidelines
    
    ### Select tools in the tools page:
    
    - Search tools by name and description
    - Select tools you want to use in your chat
    - Create custom tools directly from the interface
    
    ### Chat with the AI in the chat page:
    
    - Start a new session or continue an existing one
    - The AI can use tools when needed
    - View chat history and manage sessions
    
    ## Creating Custom Tools
    
    You can create custom tools by:
    
    1. Clicking the "+ Create Tool" button in the Tools page
    2. Writing a Python function with a proper docstring
    
    Example:
    ```python
    def weather_lookup(location):
        \"\"\"Look up the weather for a specific location\"\"\"
        # Implementation would go here
        return f"The weather in {location} is sunny."
    ```
    
    ## About the LLM Integration
    
    This application uses a custom language model via the `abc_response(prompt)` function.
    The actual implementation of this function needs to be provided separately.
    """)


def ensure_session_state():
    """Initialize session state variables"""
    if "session_id" not in st.session_state:
        session_id = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        st.session_state.session_id = session_id
        # Set default name for new session
        if "storage" in st.session_state:
            st.session_state.storage.save_session_name(session_id, "Initial Basic Greeting")
    
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
    
    if "agent_needs_update" not in st.session_state:
        st.session_state.agent_needs_update = True


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
        
        # Send to LLM using the custom abc_response function
        # Note: The actual implementation of abc_response should be provided
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
        """Call the custom abc_response function"""
        # This is where the abc_response function would be called
        # For testing purposes, we'll just return a placeholder response
        # In production, replace this with: return abc_response(prompt)
        
        # For demonstration, return a message that shows we'd call abc_response
        return f"This would be a response from calling abc_response() with the prompt. In production, replace this line with: return abc_response(prompt)"
    - Create custom tools directly from the interface
    
    ### Chat with the AI in the chat page:
    
    - Start a new session or continue an existing one
    - The AI can use tools when needed
    - View chat history and manage sessions
    
    ## Getting Started
    
    1. Navigate to the Settings page to configure the model
    2. Visit the Tools page to select which tools you want to enable
    3. Go to the Chat page to start interacting with the AI
    
    ## Creating Custom Tools
    
    You can create custom tools by:
    
    1. Using the "Create New Tool" section in the Tools page
    2. Writing a Python function with a proper docstring
    
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

def option_menu_cb():
    """Callback for menu selection"""
    st.session_state.selected_page = st.session_state.menu_opt

def menusetup():
    """Set up the navigation menu"""
    list_menu = ["Chat", "Tools", "Info"]  # Removed Settings page
    list_pages = [chat_page, tools_page, info_page]  # Removed settings_page
    st.session_state.dictpages = dict(zip(list_menu, list_pages))
    
    # Create a simple tab-based navigation
    col1, col2, col3 = st.columns(3)
    
    btn_style = """
<style>
div.stButton > button {
    background-color: transparent;
    color: #4F8BF9;
    border: none;
    border-radius: 0;
    font-weight: normal;
    width: 100%;
    height: 50px;
}
div.stButton > button:hover {
    background-color: rgba(79, 139, 249, 0.1);
    color: #4F8BF9;
}
div.stButton > button:focus {
    background-color: rgba(79, 139, 249, 0.2);
    color: #4F8BF9;
    box-shadow: none;
}
</style>
"""
    st.markdown(btn_style, unsafe_allow_html=True)
    
    # Highlight active tab
    active_style = """
    <style>
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child({}) div.stButton > button {{
        border-bottom: 2px solid #4F8BF9;
        font-weight: bold;
        background-color: rgba(79, 139, 249, 0.1);
    }}
    </style>
    """
    
    active_index = list_menu.index(st.session_state.selected_page) + 1
    st.markdown(active_style.format(active_index), unsafe_allow_html=True)
    
    # Render the buttons
    with col1:
        if st.button(f"{EMOJI_MAP['chat']} Chat", use_container_width=True, key='btn_chat'):
            st.session_state.selected_page = "Chat"
            st.rerun()
    
    with col2:
        if st.button(f"{EMOJI_MAP['tools']} Tools", use_container_width=True, key='btn_tools'):
            st.session_state.selected_page = "Tools"
            st.rerun()
    
    with col3:
        if st.button(f"{EMOJI_MAP['info']} Info", use_container_width=True, key='btn_info'):
            st.session_state.selected_page = "Info"
            st.rerun()

def pageselection(): 
    """Call the appropriate page function based on selection"""
    st.session_state.dictpages[st.session_state.selected_page]()

def ensure_session_state():
    """Initialize session state variables"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    if "model" not in st.session_state:
        st.session_state.model = "abc_model"
    
    if "tool_manager" not in st.session_state:
        st.session_state.tool_manager = ToolManager()
        st.session_state.tool_list = st.session_state.tool_manager.structured_tools
    
    if "initial_tools" not in st.session_state:
        st.session_state.initial_tools = ['calculator']
    
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
        st.session_state.selected_page = "Settings"
    
    if "prefix" not in st.session_state:
        st.session_state.prefix = ''
    
    if "suffix" not in st.session_state:
        st.session_state.suffix = ''
    
    if "storage" not in st.session_state:
        st.session_state.storage = CSVStorage()

def main():
    """Main application entry point"""
    # Set page config
    im = Image.open("appicon.ico") if os.path.exists("appicon.ico") else None
    
    st.set_page_config(
        page_title="Simplified OmniTool",
        page_icon=im,
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/example/simplified-omnitool',
            'Report a bug': "https://github.com/example/simplified-omnitool",
            'About': "Simplified version of OmniTool for custom LLM integration"
        }
    )
    
    # Initialize session state
    ensure_session_state()
    
    # Create tools directory if it doesn't exist
    tools_dir = os.path.join(BASE_DIR, "tools", "custom_tools")
    os.makedirs(tools_dir, exist_ok=True)
    
    # Set up UI
    st.title("Simplified OmniTool")
    menusetup()
    sidebar()
    pageselection()

if __name__ == "__main__":
    main()
