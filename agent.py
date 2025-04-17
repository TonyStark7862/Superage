import streamlit as st
from storage.logger_config import logger
from ui.callbacks_ui import Custom_chat_callback, ToolCallback

class AgentConfig:
    """
    Main agent configuration class that handles initialization and execution
    of Plan and Execute agent.
    """
    def __init__(self, agent_type, tools, memory):
        self.agent_type = agent_type
        self.tools = tools
        self.memory = memory
        self.thinking_steps = []
        
    def _generate_plan(self, query):
        """
        Generate a plan based on the user query.
        Returns a list of steps to execute.
        """
        # Store the thinking process
        thinking = []
        
        # Add initial thinking step
        thinking.append("Analyzing the query to understand the user's request...")
        thinking.append(f"Query: {query}")
        
        # Generate a simple plan based on available tools
        tool_names = [tool.name for tool in self.tools]
        thinking.append(f"Available tools: {', '.join(tool_names)}")
        
        # Decide which tools are relevant
        relevant_tools = []
        plan_steps = []
        
        # Simple logic to determine relevant tools
        for tool in self.tools:
            tool_relevance = self._check_tool_relevance(query, tool)
            thinking.append(f"Analyzing if tool '{tool.name}' is relevant: {tool_relevance}")
            if tool_relevance:
                relevant_tools.append(tool)
                
        thinking.append(f"Relevant tools identified: {[t.name for t in relevant_tools]}")
        
        # If no relevant tools are found, respond directly
        if not relevant_tools:
            thinking.append("No specific tools needed for this query. Will provide a direct response.")
            plan_steps = [{"type": "response", "description": "Provide direct response to the query"}]
        else:
            # Generate plan steps for each relevant tool
            thinking.append("Creating execution plan with identified tools:")
            for idx, tool in enumerate(relevant_tools):
                step = {
                    "type": "tool",
                    "tool": tool.name,
                    "description": f"Use {tool.name} to {self._get_tool_usage(tool, query)}",
                    "input": self._generate_tool_input(tool, query)
                }
                plan_steps.append(step)
                thinking.append(f"Step {idx+1}: {step['description']}")
            
            # Add final response step
            plan_steps.append({
                "type": "response",
                "description": "Synthesize results and provide final response"
            })
            thinking.append(f"Step {len(plan_steps)}: Synthesize results and provide final response")
        
        # Store thinking process
        st.session_state.agent_thinking = thinking
        
        return plan_steps
    
    def _check_tool_relevance(self, query, tool):
        """Determine if a tool is relevant for the given query."""
        # Simple keyword matching for now
        # This could be replaced with a more sophisticated relevance check
        keywords = tool.description.lower().split()
        query_words = query.lower().split()
        
        # Check if any keyword from the tool description appears in the query
        for word in query_words:
            if word in keywords or any(keyword in word for keyword in keywords):
                return True
                
        return False
    
    def _get_tool_usage(self, tool, query):
        """Generate a description of how to use the tool for this query."""
        # This is a simplistic approach - could be enhanced with better understanding
        return f"process the query: '{query}'"
    
    def _generate_tool_input(self, tool, query):
        """Generate appropriate input for the tool based on the query."""
        # Simple approach - just use the query as input
        return query
        
    def _execute_plan(self, plan_steps, query):
        """Execute each step in the plan and return the final result."""
        results = []
        
        for i, step in enumerate(plan_steps):
            # Update current step in session state
            st.session_state.current_step = i
            
            if step["type"] == "tool":
                # Find the tool
                tool = next((t for t in self.tools if t.name == step["tool"]), None)
                if tool:
                    # Execute the tool
                    try:
                        result = tool._run(step["input"])
                        results.append({
                            "step": i,
                            "tool": step["tool"],
                            "result": result
                        })
                    except Exception as e:
                        results.append({
                            "step": i,
                            "tool": step["tool"],
                            "error": str(e)
                        })
            elif step["type"] == "response":
                # This is the final response step - we'll handle this outside the loop
                pass
        
        # Generate final response based on all results
        final_response = self._synthesize_results(results, query)
        return final_response
    
    def _synthesize_results(self, results, query):
        """Combine all results to generate a final response."""
        # Simplified synthesis - just combine the results
        if not results:
            # If no results, use abc_response directly
            prompt = f"The user asked: {query}. Please provide a helpful response."
            return self._abc_response(prompt)
        
        # Build a prompt with all the tool results
        prompt = f"The user asked: {query}\n\n"
        prompt += "I used the following tools to help answer:\n\n"
        
        for res in results:
            if "error" in res:
                prompt += f"- {res['tool']}: Error occurred - {res['error']}\n"
            else:
                prompt += f"- {res['tool']}: {res['result']}\n"
        
        prompt += "\nBased on these results, please provide a final comprehensive answer to the user's query."
        
        return self._abc_response(prompt)
    
    def _abc_response(self, prompt):
        """Interface to the LLM service."""
        # This is a placeholder function - should be imported
        # In a real implementation, this would call the actual LLM service
        return f"Based on your query, here's what I found: [Response would be generated by LLM]"
    
    def _handle_error(self, error):
        """Handle errors during agent execution."""
        return f'Error in agent execution: \n{error}'

    def initialize_agent(self):
        """Initialize and return the agent executor."""
        return PlanAndExecuteAgent(self)


class PlanAndExecuteAgent:
    """
    Agent that follows a plan-then-execute approach.
    First plans a series of steps, then executes them in sequence.
    """
    def __init__(self, config):
        self.config = config
        
    def run(self, input, callbacks=None):
        """Run the agent on the given input."""
        try:
            # Store the query in the agent config
            query = input
            
            # Generate a plan
            plan_steps = self.config._generate_plan(query)
            st.session_state.current_plan = "Generated plan for execution"
            st.session_state.plan_steps = plan_steps
            
            # Execute the plan
            response = self.config._execute_plan(plan_steps, query)
            
            # Reset the current step
            st.session_state.current_step = -1
            
            return response
        except Exception as e:
            return self.config._handle_error(e)
