import streamlit as st

def apply_theme():
    """
    Apply a professional theme with consistent colors throughout the application.
    Includes CSS styling for cards, chat messages, and various UI components.
    """
    # Define theme colors
    colors = {
        "primary": "#3a86ff",      # Vibrant Blue
        "secondary": "#8338ec",    # Deep Purple 
        "success": "#06d6a0",      # Teal
        "warning": "#ffbe0b",      # Amber
        "error": "#ef476f",        # Rose
        "bg_light": "#f8f9fa",     # Light Gray
        "bg_dark": "#212529",      # Dark Gray
        "card": "#ffffff",         # White
        "text_primary": "#212529", # Dark Gray
        "text_secondary": "#6c757d", # Medium Gray
    }
    
    # Get the theme mode from session state
    theme_mode = st.session_state.get("theme_mode", "light")
    
    # Adjust colors based on theme mode
    if theme_mode == "dark":
        colors.update({
            "bg_light": "#212529",
            "bg_dark": "#121212",
            "card": "#2d3238",
            "text_primary": "#f8f9fa",
            "text_secondary": "#ced4da",
        })
    
    # Create CSS with theme colors
    css = f"""
    <style>
        /* Set CSS variables for colors */
        :root {{
            --primary-color: {colors["primary"]};
            --secondary-color: {colors["secondary"]};
            --success-color: {colors["success"]};
            --warning-color: {colors["warning"]};
            --error-color: {colors["error"]};
            --bg-light: {colors["bg_light"]};
            --bg-dark: {colors["bg_dark"]};
            --card-bg: {colors["card"]};
            --text-primary: {colors["text_primary"]};
            --text-secondary: {colors["text_secondary"]};
            --hover-bg: {colors["primary"] + "20"};
        }}
        
        /* Overall page styling */
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        
        /* Header styling */
        h1, h2, h3, h4, h5, h6 {{
            color: var(--primary-color);
            font-weight: 600;
        }}
        
        /* Card styling */
        div.stCard {{
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
            margin-bottom: 1rem;
            background-color: var(--card-bg);
            border: 1px solid rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        
        div.stCard:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }}
        
        /* Expander styling */
        .streamlit-expanderHeader {{
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        /* Chat message styling */
        [data-testid="stChatMessage"] {{
            background-color: var(--card-bg);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(0, 0, 0, 0.1);
        }}
        
        /* User message styling */
        [data-testid="stChatMessage"][data-testid="user"] {{
            border-left: 4px solid var(--primary-color);
        }}
        
        /* Assistant message styling */
        [data-testid="stChatMessage"][data-testid="assistant"] {{
            border-left: 4px solid var(--secondary-color);
        }}
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background-color: var(--bg-light);
            border-right: 1px solid rgba(0, 0, 0, 0.1);
        }}
        
        /* Button styling */
        .stButton > button {{
            border-radius: 4px;
            font-weight: 500;
            transition: all 0.2s ease;
            border: 1px solid rgba(0, 0, 0, 0.1);
        }}
        
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }}
        
        /* Primary button */
        .stButton > button.primary {{
            background-color: var(--primary-color);
            color: white;
        }}
        
        /* Secondary button */
        .stButton > button.secondary {{
            background-color: transparent;
            border: 1px solid var(--primary-color);
            color: var(--primary-color);
        }}
        
        /* Success button */
        .stButton > button.success {{
            background-color: var(--success-color);
            color: white;
        }}
        
        /* Warning button */
        .stButton > button.warning {{
            background-color: var(--warning-color);
            color: white;
        }}
        
        /* Input fields */
        div[data-baseweb="input"] {{
            border-radius: 4px;
        }}
        
        /* Progress bar */
        div.stProgress > div > div > div {{
            background-color: var(--primary-color);
        }}
        
        /* Tool card styling */
        .card-hover {{
            transition: transform 0.2s ease, box-shadow 0.2s ease, border 0.2s ease !important;
        }}
        
        .card-hover:hover {{
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1) !important;
            border-color: var(--primary-color) !important;
        }}
        
        /* Agent thinking process styling */
        .stExpander[data-testid="stExpander"] {{
            border-radius: 8px;
            border: 1px solid rgba(0, 0, 0, 0.1);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }}
        
        /* Plan execution styling */
        .plan-step {{
            padding: 8px 12px;
            border-radius: 4px;
            margin-bottom: 8px;
            background-color: var(--bg-light);
            border-left: 3px solid var(--primary-color);
        }}
        
        .plan-step-complete {{
            border-left: 3px solid var(--success-color);
        }}
        
        .plan-step-active {{
            border-left: 3px solid var(--warning-color);
            background-color: var(--warning-color) + "10";
        }}
        
        /* Tool icons styling */
        .tool-icon {{
            font-size: 24px;
            margin-right: 8px;
        }}
    </style>
    """
    
    # Apply the CSS
    st.markdown(css, unsafe_allow_html=True)
    
    # Add theme toggle in sidebar
    with st.sidebar:
        if st.button("🌓 Toggle Theme"):
            st.session_state.theme_mode = "dark" if theme_mode == "light" else "light"
            st.rerun()
