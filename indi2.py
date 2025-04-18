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

# Config Constants
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS = ['abc_model']  # Our custom model
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
        base_tools = [CalculatorTool(), PrintHelloWorldTool()]
        
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
        # This is where you'd call the abc_response function
        # The function should be provided externally
        return abc_response(prompt)


# =============================================
# UI COMPONENTS
# =============================================

def sidebar():
    with st.sidebar:
        # Add action buttons at the top
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("➕ Start new chat", use_container_width=True):
                st.session_state.session_id = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                # Initialize empty memory for the new session
                st.session_state.memory = AgentMemory()
                st.session_state.selected_page = 'Chat'
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear current chat", use_container_width=True):
                # Clear memory but keep the session
                st.session_state.memory = AgentMemory()
                st.rerun()
        
        st.markdown("### Saved Chats")
        
        # Add labels for sections
        st.markdown("#### Previous Chats")
        st.caption("Deleted after one week")
        
        # Get session data
        session_id_list = st.session_state.storage.get_all_sessions()
        st.session_state.session_name = st.session_state.storage.get_all_sessions_names()
        
        # Show starred/favorite sessions first
        # In a real app, you would track favorites in your storage system
        # For now, we'll use a simple convention: if a session name starts with "⭐", we'll consider it starred
        
        # Create favorite session display
        starred_sessions = []
        regular_sessions = []
        
        for session_id in reversed(session_id_list):
            session_name = st.session_state.session_name.get(session_id, session_id)
            if session_name.startswith("⭐"):
                starred_sessions.append((session_id, session_name))
            else:
                regular_sessions.append((session_id, session_name))
        
        # Display all sessions
        for session_id, session_name in starred_sessions + regular_sessions:
            with st.container():
                col1, col2 = st.columns([6, 1])
                
                # Session button
                with col1:
                    if st.button(f"{session_name}", key=f"session_btn_{session_id}", use_container_width=True):
                        st.session_state.session_id = session_id
                        # Load the session memory
                        st.session_state.memory = AgentMemory()
                        messages = st.session_state.storage.get_chat_history(session_id)
                        for msg in messages:
                            if msg["role"] == "user":
                                st.session_state.memory.add_user_message(msg["content"])
                            else:
                                st.session_state.memory.add_ai_message(msg["content"])
                        st.session_state.selected_page = 'Chat'
                        st.rerun()
                
                # Delete button
                with col2:
                    if st.button("🗑️", key=f"delete_btn_{session_id}"):
                        st.session_state.storage.delete_session(session_id)
                        st.rerun()
            
            # Rename field and Star toggle in a separate row for better UX
            with st.container():
                col1, col2 = st.columns([6, 1])
                with col1:
                    # Create a unique key for each rename field
                    rename_key = f"rename_{session_id}"
                    if rename_key not in st.session_state:
                        st.session_state[rename_key] = session_name.replace("⭐ ", "")
                        
                    new_name = st.text_input(
                        "Rename", 
                        value=st.session_state[rename_key],
                        key=rename_key,
                        label_visibility="collapsed"
                    )
                    
                    # Only update if the name actually changed
                    if new_name != st.session_state[rename_key]:
                        st.session_state[rename_key] = new_name
                        # Preserve star if it was there
                        if session_name.startswith("⭐"):
                            new_name = f"⭐ {new_name}"
                        st.session_state.storage.save_session_name(session_id, new_name)
                
                with col2:
                    # Star/unstar toggle
                    is_starred = session_name.startswith("⭐")
                    star_label = "★" if is_starred else "☆"
                    if st.button(star_label, key=f"star_btn_{session_id}"):
                        if is_starred:
                            # Remove star
                            new_name = session_name.replace("⭐ ", "")
                        else:
                            # Add star
                            new_name = f"⭐ {session_name}"
                        st.session_state.storage.save_session_name(session_id, new_name)
                        st.rerun()
            
            # Add a separator between sessions
            st.markdown("---")


def change_session_name(session_id):
    new_name_key = f'new_name_{session_id}'
    if new_name_key in st.session_state and st.session_state[new_name_key]:
        st.session_state.storage.save_session_name(session_id, st.session_state[new_name_key])


def chat_page():
    # Header
    st.markdown("### 💬 Chat")
    st.markdown("---")
    
    # Show selected tools with better styling
    if st.session_state.selected_tools:
        tools_str = ', '.join(st.session_state.selected_tools)
        st.markdown(f"""
        <div style="background-color: #EFF8FF; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
            <span style="font-weight: bold;">Selected tools:</span> {tools_str}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No tools selected. Visit the Tools page to enable tools.")
    
    # Initialize or update agent if tools changed
    if "agent_instance" not in st.session_state or st.session_state.tools != getattr(st.session_state.agent_instance, "tools", None):
        st.session_state.agent_instance = SimpleAgent(
            st.session_state.model,
            st.session_state.tools,
            st.session_state.memory
        )
    
    # Get chat history
    st.session_state.messages = st.session_state.storage.get_chat_history(st.session_state.session_id)
    
    # Set up the chat container with some styling
    chat_container = st.container()
    
    # Display messages with enhanced styling
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
    
    # Input
    if prompt := st.chat_input("Type your message here..."):
        # Search documents if database exists
        if "database" in st.session_state and st.session_state.database:
            # Do a similarity search in the loaded documents with the user's input
            similar_docs = st.session_state.database.similarity_search(prompt)
            # Insert the content of the most similar document into the prompt
            if similar_docs:
                prompt_with_docs = f"Relevant documentation:\n{similar_docs[0].page_content}\n\nUser prompt:\n{prompt}"
            else:
                prompt_with_docs = prompt
        else:
            prompt_with_docs = prompt
        
        # Add prefix and suffix
        original_prompt = prompt
        full_prompt = f"{st.session_state.prefix}{prompt_with_docs}{st.session_state.suffix}"
        
        # Display user message
        with chat_container:
            with st.chat_message("user"):
                st.write(original_prompt)
        
        # Add to memory
        st.session_state.memory.add_user_message(original_prompt)
        
        # Get response
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.agent_instance.run(full_prompt)
                st.write(response)
        
        # Add to memory
        st.session_state.memory.add_ai_message(response)
        
        # Save to storage
        session_name = st.session_state.session_name.get(st.session_state.session_id, st.session_state.session_id)
        st.session_state.storage.save_chat_message(st.session_state.session_id, "user", original_prompt, session_name)
        st.session_state.storage.save_chat_message(st.session_state.session_id, "assistant", response, session_name)
        st.session_state.storage.save_session_name(st.session_state.session_id, session_name)


def tools_page():
    st.markdown("### 🔧 Tools")
    st.markdown("---")
    
    # Tool selection
    st.session_state.selected_tools = []
    
    # Tool search - improved label and styling
    st.markdown("##### Search tools")
    tool_search = st.text_input('', placeholder='Search by name or description...', key='tool_search', label_visibility="collapsed")
    
    # Filter tools based on search
    if tool_search:
        st.session_state.tool_filtered = [tool for tool in st.session_state.tool_manager.get_tool_names() 
                                        if tool_search.lower() in tool.lower() or 
                                          tool_search.lower() in st.session_state.tool_manager.tools_description[tool].lower()]
    else:
        st.session_state.tool_filtered = st.session_state.tool_manager.get_tool_names()
    
    # Display tools as cards with consistent styling
    # Apply custom CSS for tool cards
    st.markdown("""
    <style>
    .tool-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        height: 200px;
        overflow-y: auto;
    }
    .tool-header {
        margin-bottom: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .tool-description {
        color: #495057;
        font-size: 14px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create grid for tools
    cols = st.columns(2)
    
    # Function to create tool card with consistent styling
    def create_tool_card(col, tool, index):
        with col:
            with st.container(border=True):
                # Use a minimum height for all cards
                st.markdown(f"""
                <div style="min-height: 160px;">
                    <h4>{tool.title}</h4>
                    <p style="color: #666; font-size: 0.9rem;">{tool.description}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Tool checkbox
                is_selected = st.checkbox(
                    f"Enable this tool", 
                    value=tool.name in st.session_state.clicked_cards,
                    key=f"tool_{tool.name}"
                )
                
                if is_selected:
                    st.session_state.clicked_cards[tool.name] = True
                    st.session_state.selected_tools.append(tool.name)
                else:
                    st.session_state.clicked_cards[tool.name] = False
                
                # Call the tool's UI method if it exists
                if hasattr(tool, '_ui') and callable(tool._ui):
                    tool._ui()
    
    # Create tool cards
    for i, tool_name in enumerate(st.session_state.tool_filtered):
        tool = st.session_state.tool_manager.get_selected_tools([tool_name])[0]
        create_tool_card(cols[i % 2], tool, i)
    
    # Update the tools in session state
    st.session_state.tools = st.session_state.tool_manager.get_selected_tools(st.session_state.selected_tools)
    
    # Add new tool - improved UI
    st.markdown("---")
    st.markdown("### Create New Tool")
    
    # Direct tool creation interface (without expander)
    col1, col2 = st.columns([1, 3])
    
    with col1:
        new_tool_name = st.text_input("Tool filename", placeholder="my_tool.py")
    
    with col2:
        st.write("Tool code:")
    
    # Code editor for tool function
    new_tool_function = st.text_area(
        "",
        height=250,
        key="new_tool_code",
        placeholder="""def my_tool(input_data):
    \"\"\"Description of what this tool does and how to use it\"\"\"
    # Tool implementation here
    return f"Result: {input_data}"
""",
        label_visibility="collapsed"
    )
    
    if st.button("Create Tool", type="primary"):
        if new_tool_name and new_tool_function:
            # Validate function
            runs, has_doc, func_name = evaluate_function_string(new_tool_function)
            
            if runs is True and has_doc is True:
                # Create the tools directory if it doesn't exist
                tools_dir = os.path.join(BASE_DIR, "tools", "custom_tools")
                os.makedirs(tools_dir, exist_ok=True)
                
                # Make sure the filename has .py extension
                if not new_tool_name.endswith('.py'):
                    new_tool_name += '.py'
                
                # Save the tool
                tool_path = os.path.join(tools_dir, new_tool_name)
                with open(tool_path, "w") as f:
                    f.write(new_tool_function)
                
                # Directly import the new tool module to avoid requiring a restart
                try:
                    module_name = new_tool_name.replace('.py', '')
                    new_module = import_from_file(tool_path, module_name)
                    
                    # Get any new tools from the module
                    classes, functions = get_class_func_from_module(new_module)
                    
                    # If we have new function tools, add them immediately
                    if functions:
                        for name, func in functions:
                            if func.__doc__:
                                # Create a dynamic tool class
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
                                # Add to the current tool list
                                st.session_state.tool_manager.structured_tools.append(func_tool())
                    
                    # Also add any class-based tools
                    for name, cls in classes:
                        if has_required_attributes(cls) and issubclass(cls, BaseTool):
                            st.session_state.tool_manager.structured_tools.append(cls())
                    
                    st.success(f"Tool '{new_tool_name}' created and activated!")
                    # Clear the input fields
                    st.session_state["new_tool_code"] = ""
                except Exception as e:
                    # Fall back to reinitialization
                    st.session_state.tool_manager = ToolManager()
                    st.session_state.tool_list = st.session_state.tool_manager.structured_tools
                    st.success(f"Tool '{new_tool_name}' created successfully!")
                    st.warning("Please refresh to see the new tool")
            else:
                if not runs is True:
                    st.error(f"Error in function: {runs}")
                if not has_doc is True:
                    st.error("Function must have a docstring to describe the tool")
        else:
            st.error("Please provide both a name and function code for the tool")


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
    st.markdown("### ℹ️ About")
    st.markdown("---")
    
    st.markdown("""
    # Simplified OmniTool
    
    This is a streamlined version of OmniTool, a framework for creating customizable 
    chatbot interfaces that can use language models and access different tools to perform tasks.
    
    ## Features
    
    - Custom language model integration
    - Tool selection and management
    - Session history management
    - Document reference integration
    - Custom tool creation
    
    ## Usage Guidelines
    
    ### Setup the chatbot in the settings page:
    
    - Choose your model
    - Define prefix and suffix for prompts
    - Load documents for reference
    
    ### Select tools in the tools page:
    
    - Filter tools by name and description
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
    # Removed Settings page
    list_menu = ["Chat", "Tools", "Info"]
    list_pages = [chat_page, tools_page, info_page]
    st.session_state.dictpages = dict(zip(list_menu, list_pages))
    
    # Create a clean horizontal menu with Streamlit components
    col1, col2, col3 = st.columns(3)
    
    # Define button styles
    button_style = """
    <style>
    div[data-testid="column"] button {
        background-color: transparent;
        color: #4F8BF9;
        border-radius: 5px;
        border: none;
        font-weight: normal;
        width: 100%;
    }
    div[data-testid="column"] button:hover {
        background-color: #EFF8FF;
    }
    div.nav-active button {
        border-bottom: 3px solid #4F8BF9 !important;
        font-weight: bold !important;
    }
    </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)
    
    # Add button active state markers
    if st.session_state.selected_page == "Chat":
        st.markdown('<div class="nav-active">', unsafe_allow_html=True)
    with col1:
        if st.button("💬 Chat", use_container_width=True):
            st.session_state.selected_page = "Chat"
            st.rerun()
    if st.session_state.selected_page == "Chat":
        st.markdown('</div>', unsafe_allow_html=True)
        
    if st.session_state.selected_page == "Tools":
        st.markdown('<div class="nav-active">', unsafe_allow_html=True)
    with col2:
        if st.button("🔧 Tools", use_container_width=True):
            st.session_state.selected_page = "Tools"
            st.rerun()
    if st.session_state.selected_page == "Tools":
        st.markdown('</div>', unsafe_allow_html=True)
        
    if st.session_state.selected_page == "Info":
        st.markdown('<div class="nav-active">', unsafe_allow_html=True)
    with col3:
        if st.button("ℹ️ Info", use_container_width=True):
            st.session_state.selected_page = "Info"
            st.rerun()
    if st.session_state.selected_page == "Info":
        st.markdown('</div>', unsafe_allow_html=True)

def pageselection(): 
    """Call the appropriate page function based on selection"""
    st.session_state.dictpages[st.session_state.selected_page]()

# Add a simple Hello World tool to demonstrate the tool creation functionality
class PrintHelloWorldTool(BaseTool):
    name = 'print_hello_world'
    icon = '👋'
    title = 'Print Hello World'
    description = 'A simple function that returns the string "Hello World".'
    
    def _run(self, input_data):
        """Return Hello World"""
        return "Hello World!"


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
