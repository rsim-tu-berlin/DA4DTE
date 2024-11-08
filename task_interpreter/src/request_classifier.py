from autogen import AssistantAgent, UserProxyAgent
import os
import json
from config import *
from utils import *

# Prompt instructions to guide the model
response_prompt_instructions = """The label must be a JSON of the format:
{
    "request_existence": bool,
    "explanation": str,
    "request_category": str
}"""

# RequestClassifierAgent Class Definition
class RequestClassifierAgent:
    def __init__(self, llm_config):
        self.llm_config = llm_config
        self.user_proxy = UserProxyAgent("user", human_input_mode="ALWAYS", code_execution_config=False)
        self.guidance_agent = AssistantAgent("guidance_labeler", llm_config=self.llm_config,
                                             system_message="You decide if the user is making a request or not, as far as the category of it.")

    # Unified method to classify both image and non-image requests
    def request_existence_and_classification(self, user_input, contains_image=False):
        # Select the appropriate categories based on whether the input contains an image
        if contains_image:
            request_types = "IMAGE_RETRIEVAL_BY_IMAGE, BINARY_VISUAL_QA, IMAGE_SEGMENTATION, OBJECT_COUNTING, None"
            input_message = f"Given this satellite image: {user_input}"
        else:
            request_types = "IMAGE_RETRIEVAL_BY_CAPTION, IMAGE_RETRIEVAL_BY_METADATA, GEOGRAPHY_QA, None"
            input_message = f"{user_input}.\n Geoentities are considered metadata."

        # Generate the full message based on the input type
        full_message = f"""
            Possible request types are: {request_types}.
            Label the USER_INPUT via the following instructions:

            {response_prompt_instructions}

            USER_INPUT: {input_message}
        """

        # Initiate chat with the appropriate prompt
        self.user_proxy.initiate_chat(
            self.guidance_agent,
            max_turns=1,
            message=full_message
        )
        
        # Parse the last message's content from the guidance agent's response
        content = json.loads(self.user_proxy.chat_messages[self.guidance_agent][-1]["content"])
        
        a = content["request_category"]
        if content["request_category"]=="GEOGRAPHY_QA":
            a = "GEOSPATIAL_QA"
        content["request_category"] = a

        if content["request_category"] in ["OBJECT_COUNTING", "IMAGE_SEGMENTATION", "IMAGE_RETRIEVAL_BY_IMAGE"]:
            full_message_2 = f"""
                Possible request types are: {content["request_category"]}, BINARY_VISUAL_QA.
                If the request can be answered with Yes/No then make the request category BINARY_VISUAL_QA.
                Label the USER_INPUT via the following instructions:

                {response_prompt_instructions}

                USER_INPUT: {input_message}
            """
                    # Initiate chat with the appropriate prompt
            self.user_proxy.initiate_chat(
                self.guidance_agent,
                max_turns=1,
                message=full_message_2 
            )
            content = json.loads(self.user_proxy.chat_messages[self.guidance_agent][-1]["content"])


        return content["request_existence"], content["request_category"]