import streamlit as st
from streamlit_elements import elements, mui, html
from streamlit_ace import st_ace
from datetime import datetime
import os
import inspect
import ast
import csv
from typing import List, Dict, Union, Any, Optional
from dataclasses import dataclass
from PIL import Image

# =============================================
# CONFIGURATION
# =============================================

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS = ['abc_model']  # Single model as requested

# Mock LLM API
def abc_response(prompt: str) -> str:
    """Mock LLM response"""
    return f"OMEGA: I am analyzing your request: {prompt}"

# System prompt
SYSTEM_PROMPT = """You are OMEGA, a state-of-the-art neural architecture built to harness the power of multiple specialized networks.
You excel at complex problem-solving, mathematical analysis, and logical reasoning. Your responses are precise and data-driven.
When using tools, clearly indicate the tool name and input in your response.
Format: "Using [tool_name] with input: [input]" """

# =============================================
# STORAGE SYSTEM
# =============================================

class CSVManager:
    """Handles all CSV storage operations"""
    
    def __init__(self, file_path: Optional[str] = None):
        self.chat_file = file_path or os.path.join(BASE_DIR, "chat_history.csv")
        self.sessions_file = os.path.join(BASE_DIR, "sessions.csv")
        self._initialize_files()
    
    def _initialize_files(self) -> None:
        """Initialize CSV files with headers"""
        if not os.path.exists(self.chat_file):
            with open(self.chat_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["session_id", "timestamp", "role", "content", "session_name"])
                
        if not os.path.exists(self.sessions_file):
            with open(self.sessions_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["session_id", "session_name"])

    def save_message(self, session_id: str, role: str, content: str, session_name: Optional[str] = None) -> None:
        """Save a chat message"""
        with open(self.chat_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                session_id,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                role,
                content,
                session_name or session_id
            ])

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get chat history for a session"""
        messages = []
        with open(self.chat_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["session_id"] == session_id:
                    messages.append({"role": row["role"], "content": row["content"]})
        return messages

    def get_sessions(self) -> List[str]:
        """Get all session IDs"""
        sessions = set()
        with open(self.chat_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sessions.add(row["session_id"])
        return list(sessions)

    def get_session_names(self) -> Dict[str, str]:
        """Get mapping of session IDs to names"""
        names = {}
        with open(self.sessions_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "session_id" in row and "session_name" in row:
                    names[row["session_id"]] = row["session_name"]
        return names

    def save_session_name(self, session_id: str, name: str) -> None:
        """Update session name"""
        names = self.get_session_names()
        names[session_id] = name
        
        with open(self.sessions_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["session_id", "session_name"])
            for s_id, s_name in names.items():
                writer.writerow([s_id, s_name])

    def delete_session(self, session_id: str) -> None:
        """Delete a session and its messages"""
        # Remove messages
        messages = []
        with open(self.chat_file, 'r') as f:
            reader = csv.DictReader(f)
            messages = [row for row in reader if row["session_id"] != session_id]
        
        with open(self.chat_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["session_id", "timestamp", "role", "content", "session_name"])
            for msg in messages:
                writer.writerow([
                    msg["session_id"],
                    msg.get("timestamp", ""),
                    msg["role"],
                    msg["content"],
                    msg.get("session_name", "")
                ])
        
        # Remove from sessions file
        names = self.get_session_names()
        if session_id in names:
            del names[session_id]
            with open(self.sessions_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["session_id", "session_name"])
                for s_id, s_name in names.items():
                    writer.writerow([s_id, s_name])

# =============================================
# TOOL SYSTEM
# =============================================

@dataclass
class BaseTool:
    """Base class for all tools"""
    name: str = "base_tool"
    description: str = "Base tool description"
    
    def run(self, input_data: str) -> str:
        """Execute the tool"""
        return self._run(input_data)
    
    def _run(self, input_data: str) -> str:
        """Implementation method to be overridden"""
        return f"Base tool executed with: {input_data}"
    
    def _ui(self) -> None:
        """Optional UI method to be overridden"""
        pass

class Calculator(BaseTool):
    """Calculator tool implementation"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Performs mathematical calculations. Input: mathematical expression (e.g. '2 + 2', 'sqrt(16)')"
        )
    
    def _run(self, expression: str) -> str:
        """Execute calculation"""
        try:
            # Create safe math environment
            safe_dict = {
                'abs': abs, 'round': round,
                'pow': pow, 'sum': sum,
                'min': min, 'max': max
            }
            
            # Add math functions
            import math
            for name in dir(math):
                if not name.startswith('_'):
                    safe_dict[name] = getattr(math, name)
            
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

def evaluate_tool_code(code: str) -> tuple[bool, bool, Optional[str]]:
    """Validate tool code"""
    try:
        tree = ast.parse(code)
        if not any(isinstance(node, ast.FunctionDef) for node in tree.body):
            return False, False, None
            
        func_def = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        has_doc = (len(func_def.body) > 0 and 
                  isinstance(func_def.body[0], ast.Expr) and 
                  isinstance(func_def.body[0].value, ast.Str))
                  
        # Test execution
        exec(code, globals())
        return True, has_doc, func_def.name
    except Exception as e:
        return False, False, str(e)

class ToolManager:
    """Manages tool registration and execution"""
    
    def __init__(self):
        self.tools: List[BaseTool] = [Calculator()]
        self._load_custom_tools()
    
    def _load_custom_tools(self) -> None:
        """Load tools from custom_tools directory"""
        tools_dir = os.path.join(BASE_DIR, "custom_tools")
        os.makedirs(tools_dir, exist_ok=True)
        
        for file in os.listdir(tools_dir):
            if file.endswith('.py'):
                try:
                    with open(os.path.join(tools_dir, file)) as f:
                        code = f.read()
                        success, has_doc, name = evaluate_tool_code(code)
                        if success and has_doc and name:
                            # Create tool class
                            exec(code)
                            new_tool = type(
                                f"{name}_Tool",
                                (BaseTool,),
                                {
                                    "name": name,
                                    "description": eval(name).__doc__,
                                    "_run": staticmethod(eval(name))
                                }
                            )
                            self.tools.append(new_tool())
                except Exception as e:
                    print(f"Error loading tool {file}: {e}")
    
    def get_tool_names(self) -> List[str]:
        """Get list of tool names"""
        return [tool.name for tool in self.tools]
    
    def get_tools(self, names: List[str]) -> List[BaseTool]:
        """Get tools by names"""
        return [tool for tool in self.tools if tool.name in names]

# =============================================
# AGENT SYSTEM
# =============================================

class Memory:
    """Manages conversation memory"""
    
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str) -> None:
        """Add message to memory"""
        self.messages.append({"role": role, "content": content})
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages"""
        return self.messages
    
    def clear(self) -> None:
        """Clear memory"""
        self.messages = []

class Agent:
    """Main agent implementation"""
    
    def __init__(self, memory: Memory, tools: List[BaseTool]):
        self.memory = memory
        self.tools = tools
    
    def run(self, prompt: str) -> str:
        """Process input and generate response"""
        # Build context from memory
        context = "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in self.memory.get_messages()
        )
        
        # Add tools info
        tools_info = "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        )
        
        # Build full prompt
        full_prompt = f"""
{SYSTEM_PROMPT}

Available tools:
{tools_info}

Conversation history:
{context}

User: {prompt}
"""
        
        # Get initial response
        response = abc_response(full_prompt)
        
        # Check for tool usage
        for tool in self.tools:
            tool_marker = f"Using {tool.name} with input:"
            if tool_marker in response:
                # Extract tool input
                tool_input = response.split(tool_marker)[1].split("\n")[0].strip()
                # Execute tool
                tool_result = tool.run(tool_input)
                # Add result to response
                response += f"\n\nTool result: {tool_result}"
        
        return response

# =============================================
# UI COMPONENTS
# =============================================

def setup_session_state():
    """Initialize session state variables"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    if "storage" not in st.session_state:
        st.session_state.storage = CSVManager()
    if "tool_manager" not in st.session_state:
        st.session_state.tool_manager = ToolManager()
    if "memory" not in st.session_state:
        st.session_state.memory = Memory()
    if "selected_tools" not in st.session_state:
        st.session_state.selected_tools = ["calculator"]
    if "tools" not in st.session_state:
        st.session_state.tools = st.session_state.tool_manager.get_tools(st.session_state.selected_tools)
    if "model" not in st.session_state:
        st.session_state.model = MODELS[0]
    if "prefix" not in st.session_state:
        st.session_state.prefix = ""
    if "suffix" not in st.session_state:
        st.session_state.suffix = ""
    if "page" not in st.session_state:
        st.session_state.page = "Chat"

def render_sidebar():
    """Render sidebar with session management"""
    with st.sidebar:
        st.markdown("---")
        if st.button("New Session", use_container_width=True):
            st.session_state.session_id = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            st.rerun()
        
        st.markdown("---")
        session_ids = st.session_state.storage.get_sessions()
        names = st.session_state.storage.get_session_names()
        
        with st.expander("Sessions", expanded=True):
            for session_id in reversed(session_ids):
                name = names.get(session_id, session_id)
                col1, col2 = st.columns([5,1])
                
                if col1.button(name, key=f"session_{session_id}", use_container_width=True):
                    st.session_state.session_id = session_id
                    st.rerun()
                
                new_name = col1.text_input(
                    "Rename",
                    key=f"rename_{session_id}",
                    label_visibility="collapsed",
                    placeholder=f"Rename: {session_id[5:]}"
                )
                
                if new_name:
                    st.session_state.storage.save_session_name(session_id, new_name)
                    st.rerun()
                
                if col2.button("❌", key=f"delete_{session_id}"):
                    st.session_state.storage.delete_session(session_id)
                    st.rerun()

def render_tools_page():
    """Render tools management page"""
    st.title("Tools")
    
    # Tool search and selection
    search = st.text_input("Search tools", "")
    tools = st.session_state.tool_manager.tools
    
    if search:
        tools = [
            t for t in tools
            if search.lower() in t.name.lower() or
               search.lower() in t.description.lower()
        ]
    
    # Display tools as cards
    cols = st.columns(3)
    for i, tool in enumerate(tools):
        with cols[i % 3]:
            with st.container(border=True):
                selected = st.checkbox(
                    tool.name,
                    value=tool.name in st.session_state.selected_tools,
                    key=f"tool_{tool.name}"
                )
                
                st.markdown(f"**{tool.description}**")
                
                if selected and tool.name not in st.session_state.selected_tools:
                    st.session_state.selected_tools.append(tool.name)
                elif not selected and tool.name in st.session_state.selected_tools:
                    st.session_state.selected_tools.remove(tool.name)
    
    # Update tools list
    st.session_state.tools = st.session_state.tool_manager.get_tools(st.session_state.selected_tools)
    
    # Tool creation
    st.markdown("---")
    with st.expander("Create New Tool"):
        name = st.text_input("Tool name")
        code = st_ace(
            value="def my_tool(input_data):\n    \"\"\"Tool description\"\"\"\n    return f\"Result: {input_data}\"",
            language="python",
            theme="monokai",
            height=300
        )
        
        if st.button("Create Tool"):
            if name and code:
                success, has_doc, _ = evaluate_tool_code(code)
                if success and has_doc:
                    tool_path = os.path.join(BASE_DIR, "custom_tools", f"{name}.py")
                    os.makedirs(os.path.dirname(tool_path), exist_ok=True)
                    
                    with open(tool_path, "w") as f:
                        f.write(code)
                    
                    st.success("Tool created successfully!")
                    st.session_state.tool_manager = ToolManager()
                    st.rerun()
                else:
                    st.error("Invalid tool code or missing docstring")

def render_chat_page():
    """Render chat interface"""
    st.title("Chat")
    
    # Show selected tools
    tools = ", ".join(st.session_state.selected_tools)
    st.info(f"Selected tools: {tools}")
    
    # Initialize agent if needed
    if "agent" not in st.session_state:
        st.session_state.agent = Agent(
            st.session_state.memory,
            st.session_state.tools
        )
    
    # Display chat history
    messages = st.session_state.storage.get_history(st.session_state.session_id)
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Message OMEGA..."):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process with agent
        full_prompt = f"{st.session_state.prefix}{prompt}{st.session_state.suffix}"
        with st.chat_message("assistant"):
            response = st.session_state.agent.run(full_prompt)
            st.write(response)
        
        # Save to storage and memory
        st.session_state.storage.save_message(
            st.session_state.session_id,
            "user",
            prompt
        )
        st.session_state.storage.save_message(
            st.session_state.session_id,
            "assistant",
            response
        )
        
        # Update memory
        st.session_state.memory.add_message("user", prompt)
        st.session_state.memory.add_message("assistant", response)

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="OMEGA",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize
    setup_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Navigation
    tab1, tab2 = st.tabs(["Chat", "Tools"])
    
    with tab1:
        render_chat_page()
    with tab2:
        render_tools_page()

if __name__ == "__main__":
    main()
