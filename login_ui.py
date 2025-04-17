import streamlit as st
import os
import csv
import pandas as pd
import re
from datetime import datetime
from storage.logger_config import logger

def login_page():
    """
    Render the login page with email authentication.
    """
    st.markdown(
        """
        <style>
        .login-container {
            max-width: 500px;
            margin: 0 auto;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-color: var(--card-bg);
        }
        .login-title {
            text-align: center;
            color: var(--primary-color);
            margin-bottom: 2rem;
        }
        .login-image {
            display: block;
            margin: 0 auto 2rem auto;
            max-width: 100px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Display login container
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    # App logo and title
    st.markdown(
        """
        <div style="text-align: center;">
            <span style="font-size: 3rem;">🤖</span>
            <h1 class="login-title">OmniTool</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Login form
    st.markdown("### Sign In")
    
    # Email input
    email = st.text_input("Email Address", placeholder="your.email@xyz.com")
    
    # Password input
    password = st.text_input("Password", type="password")
    
    # Remember me checkbox
    remember_me = st.checkbox("Remember me")
    
    # Login button
    col1, col2 = st.columns([1, 1])
    with col1:
        login_button = st.button("Sign In", type="primary", use_container_width=True)
    with col2:
        register_button = st.button("Register", use_container_width=True)
    
    # Forgot password link
    st.markdown(
        '<div style="text-align: center; margin-top: 1rem;">'
        '<a href="#" style="color: var(--primary-color);">Forgot password?</a>'
        '</div>',
        unsafe_allow_html=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle login logic
    if login_button:
        if validate_login(email, password):
            st.success("Login successful!")
            st.session_state.user_authenticated = True
            st.session_state.user_email = email
            st.rerun()
        else:
            st.error("Invalid email or password. Please try again.")
    
    # Handle registration logic
    if register_button:
        if validate_email_format(email):
            if register_user(email, password):
                st.success("Registration successful! You can now sign in.")
            else:
                st.error("This email is already registered. Please use a different email or sign in.")
        else:
            st.error("Please enter a valid email address with the format 'user@xyz.com'")
    
    # If user is already authenticated, redirect to main app
    if st.session_state.get("user_authenticated", False):
        st.rerun()

def validate_login(email, password):
    """
    Validate login credentials.
    
    Args:
        email: User's email address
        password: User's password
        
    Returns:
        True if login is valid, False otherwise
    """
    if not email or not password:
        return False
    
    # Validate email format
    if not validate_email_format(email):
        return False
    
    # Check if user exists in users.csv
    users_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data', 'users.csv')
    
    # Create users file if it doesn't exist
    if not os.path.exists(os.path.dirname(users_file)):
        os.makedirs(os.path.dirname(users_file), exist_ok=True)
    
    if not os.path.exists(users_file):
        with open(users_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'email', 'password', 'created_at', 'last_login'])
    
    try:
        df = pd.read_csv(users_file)
        user = df[df['email'] == email]
        
        if user.empty:
            return False
        
        # Check password (in a real app, you would use proper password hashing)
        if user.iloc[0]['password'] != password:
            return False
        
        # Update last login time
        df.loc[df['email'] == email, 'last_login'] = datetime.now().isoformat()
        df.to_csv(users_file, index=False)
        
        return True
    except Exception as e:
        logger.error(f"Error validating login: {e}")
        return False

def register_user(email, password):
    """
    Register a new user.
    
    Args:
        email: User's email address
        password: User's password
        
    Returns:
        True if registration is successful, False otherwise
    """
    if not email or not password:
        return False
    
    # Check email format
    if not validate_email_format(email):
        return False
    
    users_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data', 'users.csv')
    
    # Create users file if it doesn't exist
    if not os.path.exists(os.path.dirname(users_file)):
        os.makedirs(os.path.dirname(users_file), exist_ok=True)
    
    if not os.path.exists(users_file):
        with open(users_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'email', 'password', 'created_at', 'last_login'])
    
    try:
        df = pd.read_csv(users_file)
        
        # Check if email already exists
        if not df[df['email'] == email].empty:
            return False
        
        # Add new user
        next_id = 1 if df.empty else df['id'].max() + 1
        created_at = datetime.now().isoformat()
        
        with open(users_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([next_id, email, password, created_at, created_at])
        
        logger.info(f"Registered new user: {email}")
        return True
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        return False

def validate_email_format(email):
    """
    Validate email format to ensure it follows the pattern user@xyz.com.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if email is valid, False otherwise
    """
    if not email:
        return False
    
    # Check for @xyz.com pattern (allowing any domain that ends with .com)
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def logout_user():
    """Log out the current user."""
    if "user_authenticated" in st.session_state:
        del st.session_state.user_authenticated
    if "user_email" in st.session_state:
        del st.session_state.user_email
