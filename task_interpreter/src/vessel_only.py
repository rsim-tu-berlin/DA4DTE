from config import llm_config_35
from autogen import AssistantAgent, UserProxyAgent
import json

# Constants
vessel_prompt_instructions= """
The label must be a JSON of the format:
{
    "vessel_topic": bool,
    "level_of_certainty": float
}
"""

class VesselAgent : 
    def __init__(self, llm_config):
        self.llm_config = llm_config
        self.user_proxy = UserProxyAgent("user", human_input_mode="ALWAYS", code_execution_config=False)
        self.vessel_agent = AssistantAgent(
            "vessel_agent",
            llm_config=llm_config_35,
            system_message="You decide whether the user's input relates to the vessel domain.",
            code_execution_config=False,  # Disable code execution
            function_map=None,  # No registered functions
            human_input_mode="NEVER"  # No human input required
        )
    def analyze_vessel_topic(self, user_input):
        self.user_proxy.initiate_chat(
            self.vessel_agent,
            max_turns=1,
            message=f"""
            Determine whether the input relates to the vessel domain, including stating the absence of them.
            Follow these labeling instructions:
            {vessel_prompt_instructions}
            User Input: {user_input}
            """
        )
        response_content = json.loads(self.user_proxy.chat_messages[self.vessel_agent][-1]["content"])
        return response_content["vessel_topic"], response_content["level_of_certainty"]