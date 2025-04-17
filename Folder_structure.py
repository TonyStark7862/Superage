import os

def create_file_structure(base_dir):
    """
    Create the complete file structure for the OmniTool application.
    
    Args:
        base_dir: The base directory where OmniTool will be created
    """
    # Ensure base directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Create main directories
    directories = [
        'agents',
        'agents/agents_list',
        'storage',
        'storage/logs',
        'tools',
        'tools/tools_list',
        'ui',
        'data',
        'assets',
        'workspace'
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)
        print(f"Created directory: {os.path.join(base_dir, directory)}")
    
    # Create empty files
    files = [
        # Main application files
        'Omnitool_UI.py',
        'config.py',
        
        # Agent files
        'agents/__init__.py',
        'agents/agent.py',
        'agents/agents_list/__init__.py',
        'agents/agents_list/new_agent.py',
        
        # Storage files
        'storage/__init__.py',
        'storage/storage_csv.py',
        'storage/logger_config.py',
        
        # Tools files
        'tools/__init__.py',
        'tools/base_tools.py',
        'tools/tool_manager.py',
        'tools/utils.py',
        'tools/tools_list/__init__.py',
        'tools/tools_list/calculator.py',
        'tools/tools_list/test_tool.py',
        
        # UI files
        'ui/__init__.py',
        'ui/callbacks_ui.py',
        'ui/chat_ui.py',
        'ui/info_ui.py',
        'ui/login_ui.py',
        'ui/settings_ui.py',
        'ui/sidebar_ui.py',
        'ui/theme.py',
        'ui/tools_ui.py',
        
        # Data files (will be populated by the application)
        'data/.gitkeep',
        
        # Example icon
        'assets/appicon.ico'
    ]
    
    for file_path in files:
        full_path = os.path.join(base_dir, file_path)
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        # Create empty file if it doesn't exist
        if not os.path.exists(full_path):
            with open(full_path, 'w') as f:
                pass
            print(f"Created file: {full_path}")

if __name__ == "__main__":
    # Change this to your desired installation directory
    installation_dir = "./OmniTool"
    create_file_structure(installation_dir)
    
    print("\nFile structure created successfully!")
    print(f"Now you can copy and paste the code into the respective files in {installation_dir}")
