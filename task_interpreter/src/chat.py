from autogen import ConversableAgent
from config import *

class ChatAgent:
    def __init__(self, llm_config, system_message):
        self.agent = ConversableAgent(
            "chatbot",
            llm_config=llm_config,
            system_message="You ONLY discuss about Earth Observation data. If the user wants to discuss something irrelevant to your service you kindly and briefly say that you can NOT help them with that.",
            code_execution_config=False,  # Disable code execution
            function_map=None,  # No registered functions
            human_input_mode="NEVER"  # No human input required
        )
    
    def generate_reply(self, messages):
        # Generate and return the reply using the ChatAgent
        reply = self.agent.generate_reply(messages=messages)
        return reply