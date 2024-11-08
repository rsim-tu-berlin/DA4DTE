from autogen import AssistantAgent, UserProxyAgent
import os
import json
from config import *
from utils import *

# Prompt instructions to guide the model
image_description_instructions = """The label must be a JSON of the format:
{
    "isImageTypeSpecified": bool,
    "ImageType": str
}"""

image_types = "Sentinel-1, Sentinel-2, None"

# RequestClassifierAgent Class Definition
class ImageTypeAgent:
    def __init__(self, llm_config):
        self.llm_config = llm_config
        self.user_proxy = UserProxyAgent("user", human_input_mode="ALWAYS", code_execution_config=False)
        self.image_type_agent = AssistantAgent("image_type_determiner", llm_config=self.llm_config,
                                             system_message="You decide if the user specifies the type of the image(s) he/she requests.")

    def determinate_image_type(self, user_input):
        # Generate the full message based on the input type
        full_message = f"""
            Possible image types are: {image_types}.
            Decide about the image(s) type requested by the USER_INPUT via the following instructions:

            {image_description_instructions}

            USER_INPUT: {user_input}
        """

        # Initiate chat with the appropriate prompt
        self.user_proxy.initiate_chat(
            self.image_type_agent,
            max_turns=1,
            message=full_message
        )
        
        # Parse the last message's content from the guidance agent's response
        content = json.loads(self.user_proxy.chat_messages[self.image_type_agent][-1]["content"])
        if "ImageType" not in content.keys():
            content["ImageType"]="None"
        return content["isImageTypeSpecified"], content["ImageType"]