import logging
import json
import os
import sys
sys.path.append(os.getcwd() + '/src') # Add the 'src' directory to the system path to allow imports
from config import llm_config_35, log_config, vessel_msg
from request_classifier import RequestClassifierAgent
from chat import ChatAgent
from vessel_only import VesselAgent
from image_type import ImageTypeAgent

# Configure logging as per the configuration in config.py
logging.basicConfig(**log_config)

# Initialize the Request Classifier Agent
request_classifier = RequestClassifierAgent(llm_config=llm_config_35)

# Initialize the Chat Agent
chat_agent = ChatAgent(llm_config=llm_config_35, system_message="Welcome to Digital Assistant for Digital Twin Earth!")

#Initialize the Vessel Agent
vessel_agent = VesselAgent(llm_config_35)

#Initialize the Image Type Agent
image_type_agent = ImageTypeAgent(llm_config_35)

def main():
    messages = [{"role":"assistant","content": "I am DA4DTE. How can I help you?"}]
    dialog_messages = []
    save_dialog = True
    # Simulating user input
    user_input = 'hi'
    while user_input!='exit':
        answer = ""
        user_input = input(messages[-1]["content"])

        # Check if the input contains an image context
        contains_image_user = input('Contains image(yes/no)')
        if contains_image_user == 'yes':
            contains_image = True
        elif contains_image_user == 'no':
            contains_image = False
        messages.append({"content":user_input,"role":"user"})

        # Request Classifier Agent tries to classify the request
        request_existence, detected_category = request_classifier.request_existence_and_classification(user_input, contains_image=contains_image)

        if not request_existence or detected_category == 'None':
            # If no specific request or category detected, activate Chat Agent
            final_answer = chat_agent.generate_reply(messages)
        else:
            a = detected_category
            #SEARCH BY METADATA -> ONLY FOR VESSELS
            if detected_category == 'IMAGE_RETRIEVAL_BY_CAPTION':
                is_vessel, certainty = vessel_agent.analyze_vessel_topic(user_input)
            # If a specific request is classified, show the engine's answer with a placeholder for link
                if not is_vessel:
                    answer = vessel_msg 

            # IF VESSELS -> SEARCH BY METADATA
            elif detected_category == 'IMAGE_RETRIEVAL_BY_METADATA':
                is_vessel, certainty = vessel_agent.analyze_vessel_topic(user_input)
                if is_vessel:
                    a = 'IMAGE_RETRIEVAL_BY_CAPTION'
            

            elif detected_category == 'IMAGE_RETRIEVAL_BY_IMAGE':
                is_type_specified, image_type = image_type_agent.determinate_image_type(user_input)
                if is_type_specified == True:
                    a = image_type +"_" +detected_category

            detected_category = a
            if answer !="":
                final_answer=answer
            else:
                final_answer = f""" {detected_category} answer [Link]"""
        messages.append({"role":"assistant","content":final_answer})
        dialog_messages.append({"input":user_input, "image":contains_image, "answer":final_answer})

    if save_dialog:
        if user_input.lower() == 'exit':
            dialogue_dir = 'dialogues'
            if not os.path.exists(dialogue_dir):
                os.makedirs(dialogue_dir)
        # List only entries in the directory that are files
            files = [file for file in os.listdir(dialogue_dir) if os.path.isfile(os.path.join(dialogue_dir, file))]
            file_number = len(files)+1
            dialog_file = 'dialog_'+ str(file_number)

            file_path = dialogue_dir +'/'+dialog_file
            with open(file_path, 'w') as f:
                json.dump(dialog_messages, f, indent=4)

    return messages

if __name__ == "__main__":
    main()