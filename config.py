# Configuration settings for OmniTool

# Logger level 
import logging
LOGGER_LEVEL = logging.INFO

# Agent settings
MAX_ITERATIONS = 20  # Maximum thoughts iteration per query

# UI Settings
DEFAULT_THEME = "light"  # Default theme mode (light/dark)
SHOW_THINKING_DEFAULT = True  # Default setting for showing agent thinking
DEFAULT_AGENT = "PlanAndExecute"  # Default agent type

# Tool settings
DEFAULT_TOOLS = ["Calculator", "TestTool"]  # Default activated tools

# User pattern validation
EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'  # Valid email pattern

# Paths (relative to base directory)
DATA_DIR = "data"  # Directory for CSV storage
ASSETS_DIR = "assets"  # Directory for images and other assets
TOOLS_LIST_DIR = "tools/tools_list"  # Directory for tool implementations

# UI appearance
UI_COLORS = {
    "primary": "#3a86ff",      # Vibrant Blue
    "secondary": "#8338ec",    # Deep Purple 
    "success": "#06d6a0",      # Teal
    "warning": "#ffbe0b",      # Amber
    "error": "#ef476f",        # Rose
}
