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
        st.markdown("---")  # Separator
        
        if st.button("Start New Session", use_container_width=True):
            st.session_state.session_id = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            st.session_state.selected_page = 'Chat'
            st.rerun()
           
        st.markdown("---")

        session_id_list = st.session_state.storage.get_all_sessions()
        st.session_state.session_name = st.session_state.storage.get_all_sessions_names()
        session_expander = st.expander(label='Sessions', expanded=True)

        with session_expander:
            if session_id_list:
                for session_id in reversed(session_id_list):
                    button_container = st.empty()
                    session_name = st.session_state.session_name.get(session_id, session_id)
                    col1, col2 = st.columns([5,1])
                    with button_container:
                        with st.container():
                            if st.button(session_name, use_container_width=True, key=f"session_{session_id}"):
                                st.session_state.session_id = session_id
                                st.rerun()
                    with col1:
                        st.text_input('Rename', key=f'new_name_{session_id}', 
                                     placeholder=f'Rename: {str(session_id)[5:]}', 
                                     label_visibility="collapsed",
                                     on_change=lambda s_id=session_id: change_session_name(s_id))
                    with col2:
                        if st.button('❌', use_container_width=True, key=f"delete_{session_id}"):
                            st.session_state.storage.delete_session(session_id)
                            st.rerun()


def change_session_name(session_id):
    new_name_key = f'new_name_{session_id}'
    if new_name_key in st.session_state and st.session_state[new_name_key]:
        st.session_state.storage.save_session_name(session_id, st.session_state[new_name_key])


def chat_page():
    # Header
    st.markdown("### 💭 Chat")
    st.markdown("---")
    
    # Show selected tools
    s = ', '.join(st.session_state.selected_tools)
    st.info(f'Selected tools: {s}')
    
    # Initialize agent if needed
    if "agent_instance" not in st.session_state:
        st.session_state.agent_instance = SimpleAgent(
            st.session_state.model,
            st.session_state.tools,
            st.session_state.memory
        )
    
    # Get chat history
    st.session_state.messages = st.session_state.storage.get_chat_history(st.session_state.session_id)
    
    # Display messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
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
        session_name = st.session_state.session_name.get(st.session_state.session_id, st.session_state.session_id)
        st.session_state.storage.save_chat_message(st.session_state.session_id, "user", original_prompt, session_name)
        st.session_state.storage.save_chat_message(st.session_state.session_id, "assistant", response, session_name)
        st.session_state.storage.save_session_name(st.session_state.session_id, session_name)


def tools_page():
    st.markdown("### 🛠️ Tools")
    st.markdown("---")
    
    # Tool selection
    st.session_state.selected_tools = []
    
    # Tool search
    tool_search = st.text_input('Filter', placeholder='Filter tools', key='tool_search')
    
    # Filter tools based on search
    if tool_search:
        st.session_state.tool_filtered = [tool for tool in st.session_state.tool_manager.get_tool_names() 
                                        if tool_search.lower() in tool.lower() or 
                                          tool_search.lower() in st.session_state.tool_manager.tools_description[tool].lower()]
    else:
        st.session_state.tool_filtered = st.session_state.tool_manager.get_tool_names()
    
    # Display tools as cards
    cols = st.columns(3)
    for i, tool_name in enumerate(st.session_state.tool_filtered):
        tool = st.session_state.tool_manager.get_selected_tools([tool_name])[0]
        col = cols[i % 3]
        
        with col:
            card = st.container(border=True)
            with card:
                is_selected = st.checkbox(
                    f"{tool.icon} {tool.title}", 
                    value=tool_name in st.session_state.clicked_cards,
                    key=f"tool_{tool_name}"
                )
                
                st.markdown(f"**{tool.description}**")
                
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
    
    # Add new tool
    st.markdown("---")
    st.markdown("### Create New Tool")
    
    with st.expander("Create a new tool"):
        new_tool_name = st.text_input("Tool file name")
        new_tool_function = st.text_area(
            "Tool code", 
            height=300,
            placeholder="""def my_tool(input_data):
    \"\"\"Description of what this tool does and how to use it\"\"\"
    # Tool implementation here
    return f"Result: {input_data}"
"""
        )
        
        if st.button("Create Tool"):
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
                    
                    # Reinitialize tool manager
                    st.session_state.tool_manager = ToolManager()
                    st.session_state.tool_list = st.session_state.tool_manager.structured_tools
                    
                    st.success(f"Tool '{new_tool_name}' created successfully!")
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
    list_menu = ["Chat", "Tools", "Settings", "Info"]
    list_pages = [chat_page, tools_page, settings_page, info_page]
    st.session_state.dictpages = dict(zip(list_menu, list_pages))
    list_icons = ['chat-left-text', 'tools', 'gear', 'info-circle']
    
    # Create a horizontal menu
    menu_items = ""
    for i, (menu, icon) in enumerate(zip(list_menu, list_icons)):
        active = "active" if menu == st.session_state.selected_page else ""
        menu_items += f"""
        <div class="nav-item {active}">
            <a href="#" onclick="setPage('{menu}'); return false;" class="nav-link">
                <i class="bi bi-{icon}"></i> {menu}
            </a>
        </div>
        """
    
    # JavaScript to handle page selection
    st.markdown(f"""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.0/font/bootstrap-icons.css">
    <style>
        .nav-container {{
            display: flex;
            justify-content: space-around;
            background-color: #f8f9fa;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .nav-item {{
            flex: 1;
            text-align: center;
            padding: 10px;
        }}
        .nav-item.active {{
            background-color: #e9ecef;
            border-bottom: 3px solid #1f77b4;
        }}
        .nav-link {{
            text-decoration: none;
            color: #495057;
            display: block;
        }}
        .nav-item.active .nav-link {{
            font-weight: bold;
        }}
        .nav-link i {{
            margin-right: 5px;
        }}
    </style>
    <div class="nav-container">
        {menu_items}
    </div>
    <script>
        function setPage(page) {{
            // Use Streamlit's custom component communication
            const data = {{selected_page: page}};
            window.parent.postMessage({{
                type: "streamlit:setComponentValue",
                value: data
            }}, "*");
        }}
    </script>
    """, unsafe_allow_html=True)
    
    # Check if the page was changed via the custom HTML/JS
    if st.session_state.get("selected_page_from_js"):
        st.session_state.selected_page = st.session_state.selected_page_from_js
        st.session_state.selected_page_from_js = None
        st.rerun()
    
    # Alternatively, use streamlit-option-menu package if available
    # This is a fallback if you have it installed
    try:
        from streamlit_option_menu import option_menu
        st.session_state.selected_page = option_menu(
            "", list_menu,
            icons=list_icons,
            menu_icon="",
            orientation="horizontal",
            key='menu_opt',
            default_index=list_menu.index(st.session_state.selected_page)
        )
    except ImportError:
        # Simplified fallback
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Chat", use_container_width=True):
                st.session_state.selected_page = "Chat"
                st.rerun()
        with col2:
            if st.button("Tools", use_container_width=True):
                st.session_state.selected_page = "Tools"
                st.rerun()
        with col3:
            if st.button("Settings", use_container_width=True):
                st.session_state.selected_page = "Settings"
                st.rerun()
        with col4:
            if st.button("Info", use_container_width=True):
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
