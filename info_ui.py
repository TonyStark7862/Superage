import streamlit as st

def info_page():
    """
    Render the information page with docs, examples, and guidance.
    """
    st.markdown("## ℹ️ OmniTool Information")
    
    # Create tabs for different information sections
    tab1, tab2, tab3 = st.tabs(["Overview", "Tool Development", "Examples"])
    
    with tab1:
        overview_section()
    
    with tab2:
        tool_development_section()
    
    with tab3:
        examples_section()

def overview_section():
    """Display general information about OmniTool."""
    st.markdown("""
    # Welcome to OmniTool
    
    OmniTool is a next-generation agentic interface with a pluggable tool system. It provides a modern and intuitive way to interact with AI agents that can use various tools to accomplish tasks.
    
    ## Key Features
    
    ### 🧠 Plan and Execute Agent
    OmniTool features a sophisticated Plan and Execute agent that:
    - Analyzes your request and formulates a detailed plan
    - Selects appropriate tools based on your query
    - Executes steps in sequence to complete complex tasks
    - Provides transparent reasoning for its actions
    
    ### 🧰 Pluggable Tool System
    - Tools can be easily added, removed, or modified
    - Custom tools can be created with minimal coding
    - Tool settings can be configured via the user interface
    
    ### 💬 Enhanced Chat Interface
    - View the agent's thinking process in real-time
    - Track execution progress of multi-step plans
    - Access chat history across multiple sessions
    
    ## Getting Started
    
    1. **Select Tools**: Visit the Tools tab to enable tools for your agent
    2. **Start Chatting**: Go to the Chat tab and start a conversation
    3. **Create Custom Tools**: Extend functionality by creating your own tools
    
    ## System Requirements
    
    OmniTool is built with Python and Streamlit, making it cross-platform and easy to deploy.
    """)

def tool_development_section():
    """Provide guidance on creating custom tools."""
    st.markdown("""
    # Tool Development Guide
    
    OmniTool's pluggable tool system makes it easy to create and integrate custom tools. Here's how to create your own tools:
    
    ## Using the Tool Creator Interface
    
    The simplest way to create a tool is to use the built-in Tool Creator in the Tools tab:
    
    1. Expand the "Create New Tool" section
    2. Enter a name for your tool
    3. Write a Python function with a descriptive docstring
    4. Click "Create Tool"
    
    ## Creating Advanced Tools
    
    For more advanced tools with custom UI components, you can create a Python class:
    
    ```python
    from tools.base_tools import Ui_Tool
    import streamlit as st
    
    class MyAdvancedTool(Ui_Tool):
        name = 'my_advanced_tool'
        icon = '🔧'
        title = 'My Advanced Tool'
        description = 'Description of what this tool does'
        category = 'Custom'
        
        def __init__(self):
            super().__init__()
            # Initialize any tool-specific settings here
            self.settings = {
                'parameter1': 'default_value',
                'parameter2': 42
            }
        
        def _run(self, input_str):
            # Tool implementation logic
            result = f"Processed {input_str} with {self.settings['parameter1']}"
            return result
        
        def _ui(self):
            # Custom UI for tool configuration
            with st.expander("Tool Settings"):
                # Create UI components for settings
                self.settings['parameter1'] = st.text_input(
                    "Parameter 1",
                    value=self.settings['parameter1']
                )
                self.settings['parameter2'] = st.number_input(
                    "Parameter 2",
                    value=self.settings['parameter2']
                )
    ```
    
    Save your tool class in a Python file in the `tools/tools_list/` directory, and it will be automatically loaded when OmniTool starts.
    
    ## Tool Integration Best Practices
    
    - **Clear Description**: Write a clear docstring that explains what your tool does
    - **Error Handling**: Include robust error handling in your `_run` method
    - **Input Validation**: Validate user input before processing
    - **Informative Output**: Return well-formatted, informative results
    - **Consistent UI**: Follow OmniTool's UI patterns for settings
    
    ## Testing Your Tools
    
    After creating a tool, you can test it by:
    
    1. Enabling it in the Tools tab
    2. Going to the Chat tab
    3. Asking the agent to use your tool
    
    The agent will incorporate your tool into its planning and execution based on the tool's description.
    """)

def examples_section():
    """Provide usage examples."""
    st.markdown("""
    # Usage Examples
    
    Here are some examples of how to use OmniTool effectively:
    
    ## Basic Calculations
    
    You can ask the agent to perform calculations:
    
    > "Calculate the square root of 144"
    
    The agent will:
    1. Analyze your request
    2. Determine that the Calculator tool is appropriate
    3. Use the tool to perform the calculation
    4. Return the result: 12
    
    ## Multi-step Tasks
    
    OmniTool excels at breaking down complex tasks:
    
    > "Calculate the average of 15, 27, 42, and 31, then find the square root of the result"
    
    The agent will:
    1. Plan a sequence of steps
    2. Calculate the average: (15 + 27 + 42 + 31) / 4 = 28.75
    3. Find the square root of 28.75: 5.36
    4. Provide the final result along with the intermediate steps
    
    ## Adding Custom Tools
    
    You can extend OmniTool's capabilities with custom tools:
    
    1. Create a temperature conversion tool using the Tool Creator
    2. Enable the tool in the Tools tab
    3. Ask: "Convert 32 degrees Fahrenheit to Celsius"
    
    The agent will use your custom tool to perform the conversion.
    
    ## Customizing Agent Behavior
    
    You can influence the agent's behavior using settings:
    
    1. Go to the Settings tab
    2. Add a prefix like "You are a helpful assistant specialized in mathematics"
    3. Ask a math-related question
    
    The agent will incorporate this guidance into its responses.
    """)
