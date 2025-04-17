import logging
import os

# Set logging level
LOGGER_LEVEL = logging.INFO

def configure_logger():
    """
    Configure the application logger.
    Sets up logging to both file and console.
    
    Returns:
        Logger instance
    """
    # Get the directory where this file is located
    local_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(local_dir, "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Log file path
    log_file = os.path.join(logs_dir, "omnitool.log")
    
    # Configure basic logging settings
    logging.basicConfig(
        level=LOGGER_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create and return logger
    return logging.getLogger("OmniTool")

# Create and configure the logger
logger = configure_logger()
