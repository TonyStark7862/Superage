import streamlit as st
import re
import math
from tools.base_tools import Ui_Tool

class Calculator(Ui_Tool):
    """
    Calculator tool that can evaluate mathematical expressions.
    """
    name = 'Calculator'
    icon = '🧮'
    title = 'Calculator'
    description = 'Perform mathematical calculations and evaluate expressions'
    category = 'Utilities'
    version = '1.0'
    author = 'System'
    
    def __init__(self):
        super().__init__()
        self.settings = {
            'precision': 4,
            'allow_complex': True,
            'scientific_notation': False
        }
    
    def _run(self, input_str):
        """
        Parse and evaluate a mathematical expression.
        
        Args:
            input_str: A string containing a mathematical expression
            
        Returns:
            The result of the calculation as a string
        """
        try:
            # Clean the input
            expression = self._clean_expression(input_str)
            
            # Check if the expression is empty after cleaning
            if not expression:
                return "Please provide a valid mathematical expression."
            
            # Evaluate the expression
            result = self._safe_eval(expression)
            
            # Format the result based on settings
            formatted_result = self._format_result(result)
            
            return f"Result: {formatted_result}"
        except Exception as e:
            return f"Error calculating: {str(e)}"
    
    def _clean_expression(self, expression):
        """
        Clean and normalize the mathematical expression.
        
        Args:
            expression: Input expression string
            
        Returns:
            Cleaned expression string
        """
        # Handle common word problem patterns
        expression = self._handle_word_problems(expression)
        
        # Remove anything that's not a valid part of a mathematical expression
        # Allow digits, operators, parentheses, decimals, and common math functions
        clean_expr = re.sub(r'[^0-9+\-*/^().,%\s\w]', '', expression)
        
        # Replace ^ with ** for exponentiation
        clean_expr = clean_expr.replace('^', '**')
        
        # Replace % with /100* for percentage
        clean_expr = re.sub(r'(\d+)%', r'\1/100', clean_expr)
        
        # Replace common math function names
        replacements = {
            'sqrt': 'math.sqrt',
            'sin': 'math.sin',
            'cos': 'math.cos',
            'tan': 'math.tan',
            'log': 'math.log10',
            'ln': 'math.log',
            'exp': 'math.exp',
            'pi': 'math.pi',
            'e': 'math.e'
        }
        
        for word, replacement in replacements.items():
            clean_expr = re.sub(r'\b' + word + r'\b', replacement, clean_expr)
        
        return clean_expr
    
    def _handle_word_problems(self, expression):
        """
        Try to extract mathematical expressions from word problems.
        
        Args:
            expression: Input word problem string
            
        Returns:
            Extracted mathematical expression
        """
        # Look for patterns like "what is 5 plus 3" or "calculate 10 divided by 2"
        patterns = [
            (r'what\s+is\s+(\d+)\s+plus\s+(\d+)', r'\1+\2'),
            (r'what\s+is\s+(\d+)\s+minus\s+(\d+)', r'\1-\2'),
            (r'what\s+is\s+(\d+)\s+times\s+(\d+)', r'\1*\2'),
            (r'what\s+is\s+(\d+)\s+multiplied\s+by\s+(\d+)', r'\1*\2'),
            (r'what\s+is\s+(\d+)\s+divided\s+by\s+(\d+)', r'\1/\2'),
            (r'calculate\s+(\d+)\s+minus\s+(\d+)', r'\1-\2'),
            (r'calculate\s+(\d+)\s+times\s+(\d+)', r'\1*\2'),
            (r'calculate\s+(\d+)\s+divided\s+by\s+(\d+)', r'\1/\2'),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                return re.sub(pattern, replacement, expression, flags=re.IGNORECASE)
        
        return expression
    
    def _safe_eval(self, expression):
        """
        Safely evaluate the mathematical expression.
        
        Args:
            expression: Cleaned expression string
            
        Returns:
            Result of the evaluation
        """
        # Create a safe local environment with only math functions
        safe_locals = {
            'math': math,
            '__builtins__': {}
        }
        
        # Evaluate the expression in the safe environment
        result = eval(expression, {"__builtins__": {}}, safe_locals)
        return result
    
    def _format_result(self, result):
        """
        Format the result based on user settings.
        
        Args:
            result: The raw calculation result
            
        Returns:
            Formatted result string
        """
        # Handle complex numbers if not allowed
        if isinstance(result, complex) and not self.settings['allow_complex']:
            return "Error: Complex result not allowed"
        
        # Format based on settings
        if isinstance(result, (int, float)):
            # Round to specified precision
            rounded = round(result, self.settings['precision'])
            
            # Convert to integer if it's a whole number
            if rounded == int(rounded):
                rounded = int(rounded)
                
            # Use scientific notation if enabled and appropriate
            if self.settings['scientific_notation'] and (abs(rounded) > 1e6 or 0 < abs(rounded) < 1e-4):
                return f"{rounded:.{self.settings['precision']}e}"
            
            # Regular formatting
            if isinstance(rounded, int):
                return str(rounded)
            else:
                return f"{rounded:.{self.settings['precision']}f}".rstrip('0').rstrip('.')
        
        # Return as is for other types (like complex numbers)
        return str(result)
    
    def _ui(self):
        """Render custom UI controls for calculator settings."""
        with st.expander("Calculator Settings"):
            # Precision setting
            st.slider(
                "Precision (decimal places)",
                min_value=0,
                max_value=10,
                value=self.settings['precision'],
                key=f"calc_precision_{id(self)}",
                on_change=self._update_precision
            )
            
            # Complex numbers setting
            st.checkbox(
                "Allow complex numbers",
                value=self.settings['allow_complex'],
                key=f"calc_complex_{id(self)}",
                on_change=self._update_complex_setting
            )
            
            # Scientific notation setting
            st.checkbox(
                "Use scientific notation for large/small numbers",
                value=self.settings['scientific_notation'],
                key=f"calc_scientific_{id(self)}",
                on_change=self._update_scientific_setting
            )
    
    def _update_precision(self):
        """Update precision setting from UI."""
        self.settings['precision'] = st.session_state[f"calc_precision_{id(self)}"]
    
    def _update_complex_setting(self):
        """Update complex numbers setting from UI."""
        self.settings['allow_complex'] = st.session_state[f"calc_complex_{id(self)}"]
    
    def _update_scientific_setting(self):
        """Update scientific notation setting from UI."""
        self.settings['scientific_notation'] = st.session_state[f"calc_scientific_{id(self)}"]
+plus\s+(\d+)', r'\1+\2'),
            (r'calculate\s+(\d+)\s
