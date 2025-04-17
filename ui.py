import os
from datetime import datetime
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit_shadcn_ui as ui

from tools.tool_manager import ToolManager
from storage.storage_csv import CSVStorage
from storage.logger_config import logger
from ui.sidebar_ui import sidebar
from ui.chat_ui import chat_page
from ui.settings_ui import settings_page
from ui.tools_ui import tools_page
from ui.info_ui import info_page
from ui.login_ui import login_page, logout_user
from ui.theme import apply_theme
from PIL import Image
import random

# Set up base directory
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
logger.debug('BASE_DIR: ' + BASE_DIR)

# Set page configuration
im = Image.open(BASE_DIR + '/assets/appicon.ico')
st.set_page_config(
    page_title="OmniTool",
    page_icon=im,
    initial_sidebar_state="expanded",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/user/OmniTool/tree/master',
        'Report a bug': "https://github.com/user/OmniTool/tree/master",
        'About': "Next-gen agent interface with pluggable tools"
    }
)

# Apply theme
apply_theme()

# Session state management
def ensure_session_state():
    logger.debug('Ensure sessions states')
    # Ensure there are defaults for the session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    if "agent" not in st.session_state:
        st.session_state.agent = "PlanAndExecute"
    if 'tool_manager' not in st.session_state:
        st.session_state.tool_manager = ToolManager()
        st.session_state.tool_list = st.session_state.tool_manager.structured_tools
    if "initial_tools" not in st.session_state:
        # Default selected tool
        st.session_state.initial_tools = ['Calculator']
    if "selected_tools" not in st.session_state:
        st.session_state.selected_tools = st.session_state.initial_tools
    if "tools" not in st.session_state:
        st.session_state.tools = st.session_state.tool_manager.get_selected_tools(st.session_state.initial_tools)
    if "clicked_cards" not in st.session_state:
        st.session_state.clicked_cards = {tool_name: True for tool_name in st.session_state.initial_tools}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "memory" not in st.session_state:
        st.session_state.memory = []
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Chat"
    if "prefix" not in st.session_state:
        st.session_state.prefix = ''
    if "suffix" not in st.session_state:
        st.session_state.suffix = ''
    if "session_name" not in st.session_state:
        st.session_state.session_name = {}
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "light"
    if "agent_thinking" not in st.session_state:
        st.session_state.agent_thinking = []
    if "current_plan" not in st.session_state:
        st.session_state.current_plan = None
    if "plan_steps" not in st.session_state:
        st.session_state.plan_steps = []
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    if "show_thinking" not in st.session_state:
        st.session_state.show_thinking = True
    if "user_authenticated" not in st.session_state:
        st.session_state.user_authenticated = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""

# Menu callback
def option_menu_cb():
    st.session_state.selected_page = st.session_state.menu_opt

# Initialize storage
def init_storage():
    logger.info('Building storage')
    # Create CSV storage
    storage = CSVStorage(BASE_DIR)
    return storage

# Option Menu
def menusetup():
    list_menu = ["Chat", "Tools", "Settings", "Info"]
    list_pages = [chat_page, tools_page, settings_page, info_page]
    st.session_state.dictpages = dict(zip(list_menu, list_pages))
    list_icons = ['chat', 'tools', 'gear', 'info-circle']
    
    # Add logout option
    if st.session_state.user_authenticated:
        list_menu.append("Logout")
        list_pages.append(handle_logout)
        list_icons.append('box-arrow-right')
    
    st.session_state.selected_page = option_menu(
        "",
        list_menu,
        icons=list_icons, 
        menu_icon="", 
        orientation="horizontal",
        on_change=option_menu_cb,
        key='menu_opt',
        default_index=list_menu.index(st.session_state.selected_page) if st.session_state.selected_page in list_menu else 0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "var(--primary-color)", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "padding": "10px 15px",
                "--hover-color": "var(--hover-bg)"
            },
            "nav-link-selected": {"background-color": "var(--primary-color)", "color": "white"},
        }
    )

def pageselection():
    if st.session_state.selected_page in st.session_state.dictpages:
        st.session_state.dictpages[st.session_state.selected_page]()

def handle_logout():
    """Handle user logout."""
    logout_user()
    st.success("You have been logged out successfully!")
    st.rerun()

# Main application
def main():
    ensure_session_state()
    
    # Check if user is authenticated
    if not st.session_state.user_authenticated:
        login_page()
        return
    
    # Initialize application components
    menusetup()
    st.session_state.storage = init_storage()
    sidebar()
    pageselection()

if __name__ == "__main__":
    main()
def ensure_session_state():
    logger.debug('Ensure sessions states')
    # Ensure there are defaults for the session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    if "agent" not in st.session_state:
        st.session_state.agent = "PlanAndExecute"
    if 'tool_manager' not in st.session_state:
        st.session_state.tool_manager = ToolManager()
        st.session_state.tool_list = st.session_state.tool_manager.structured_tools
    if "initial_tools" not in st.session_state:
        # Default selected tool
        st.session_state.initial_tools = ['Calculator']
    if "selected_tools" not in st.session_state:
        st.session_state.selected_tools = st.session_state.initial_tools
    if "tools" not in st.session_state:
        st.session_state.tools = st.session_state.tool_manager.get_selected_tools(st.session_state.initial_tools)
    if "clicked_cards" not in st.session_state:
        st.session_state.clicked_cards = {tool_name: True for tool_name in st.session_state.initial_tools}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "memory" not in st.session_state:
        st.session_state.memory = []
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Chat"
    if "prefix" not in st.session_state:
        st.session_state.prefix = ''
    if "suffix" not in st.session_state:
        st.session_state.suffix = ''
    if "session_name" not in st.session_state:
        st.session_state.session_name = {}
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "light"
    if "agent_thinking" not in st.session_state:
        st.session_state.agent_thinking = []
    if "current_plan" not in st.session_state:
        st.session_state.current_plan = None
    if "plan_steps" not in st.session_state:
        st.session_state.plan_steps = []
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    if "show_thinking" not in st.session_state:
        st.session_state.show_thinking = True
    if "user_authenticated" not in st.session_state:
        st.session_state.user_authenticated = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""

# Menu callback
def option_menu_cb():
    st.session_state.selected_page = st.session_state.menu_opt

# Initialize storage
def init_storage():
    logger.info('Building storage')
    # Create CSV storage
    storage = CSVStorage(BASE_DIR)
    return storage

# Option Menu
def menusetup():
    list_menu = ["Chat", "Tools", "Settings", "Info"]
    list_pages = [chat_page, tools_page, settings_page, info_page]
    st.session_state.dictpages = dict(zip(list_menu, list_pages))
    list_icons = ['chat', 'tools', 'gear', 'info-circle']
    
    # Add logout option
    if st.session_state.user_authenticated:
        list_menu.append("Logout")
        list_pages.append(handle_logout)
        list_icons.append('box-arrow-right')
    
    st.session_state.selected_page = option_menu(
        "",
        list_menu,
        icons=list_icons, 
        menu_icon="", 
        orientation="horizontal",
        on_change=option_menu_cb,
        key='menu_opt',
        default_index=list_menu.index(st.session_state.selected_page) if st.session_state.selected_page in list_menu else 0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "var(--primary-color)", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "padding": "10px 15px",
                "--hover-color": "var(--hover-bg)"
            },
            "nav-link-selected": {"background-color": "var(--primary-color)", "color": "white"},
        }
    )

def pageselection():
    if st.session_state.selected_page in st.session_state.dictpages:
        st.session_state.dictpages[st.session_state.selected_page]()

def handle_logout():
    """Handle user logout."""
    logout_user()
    st.success("You have been logged out successfully!")
    st.rerun()

# Main application
def main():
    ensure_session_state()
    
    # Check if user is authenticated
    if not st.session_state.user_authenticated:
        login_page()
        return
    
    # Initialize application components
    menusetup()
    st.session_state.storage = init_storage()
    sidebar()
    pageselection()

if __name__ == "__main__":
    main()
Next-gen agent interface with pluggable tools"
    }
)

# Apply theme
apply_theme()

# Session state management
def ensure_session_state():
    logger.debug('Ensure sessions states')
    # Ensure there are defaults for the session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    if "agent" not in st.session_state:
        st.session_state.agent = "PlanAndExecute"
    if 'tool_manager' not in st.session_state:
        st.session_state.tool_manager = ToolManager()
        st.session_state.tool_list = st.session_state.tool_manager.structured_tools
    if "initial_tools" not in st.session_state:
        # Default selected tool
        st.session_state.initial_tools = ['Calculator']
    if "selected_tools" not in st.session_state:
        st.session_state.selected_tools = st.session_state.initial_tools
    if "tools" not in st.session_state:
        st.session_state.tools = st.session_state.tool_manager.get_selected_tools(st.session_state.initial_tools)
    if "clicked_cards" not in st.session_state:
        st.session_state.clicked_cards = {tool_name: True for tool_name in st.session_state.initial_tools}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "memory" not in st.session_state:
        st.session_state.memory = []
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Chat"
    if "prefix" not in st.session_state:
        st.session_state.prefix = ''
    if "suffix" not in st.session_state:
        st.session_state.suffix = ''
    if "session_name" not in st.session_state:
        st.session_state.session_name = {}
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "light"
    if "agent_thinking" not in st.session_state:
        st.session_state.agent_thinking = []
    if "current_plan" not in st.session_state:
        st.session_state.current_plan = None
    if "plan_steps" not in st.session_state:
        st.session_state.plan_steps = []
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    if "show_thinking" not in st.session_state:
        st.session_state.show_thinking = True

# Menu callback
def option_menu_cb():
    st.session_state.selected_page = st.session_state.menu_opt

# Initialize storage
def init_storage(db_url='sqlite:///' + BASE_DIR + '//storage//app_session_history.db'):
    logger.info('Building storage')
    # Create or connect to db
    storage = PersistentStorage(db_url)
    return storage

# Option Menu
def menusetup():
    list_menu = ["Chat", "Tools", "Settings", "Info"]
    list_pages = [chat_page, tools_page, settings_page, info_page]
    st.session_state.dictpages = dict(zip(list_menu, list_pages))
    list_icons = ['chat', 'tools', 'gear', 'info-circle']
    
    st.session_state.selected_page = option_menu(
        "",
        list_menu,
        icons=list_icons, 
        menu_icon="", 
        orientation="horizontal",
        on_change=option_menu_cb,
        key='menu_opt',
        default_index=list_menu.index(st.session_state.selected_page),
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "var(--primary-color)", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "padding": "10px 15px",
                "--hover-color": "var(--hover-bg)"
            },
            "nav-link-selected": {"background-color": "var(--primary-color)", "color": "white"},
        }
    )

def pageselection():
    st.session_state.dictpages[st.session_state.selected_page]()

# Main application
def main():
    ensure_session_state()
    menusetup()
    st.session_state.storage = init_storage()
    sidebar()
    pageselection()

if __name__ == "__main__":
    main()
