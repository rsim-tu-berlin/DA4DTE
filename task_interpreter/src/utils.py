from config import *
# from guidance import assistant, gen, models,system, user
import json
import os
import pandas as pd
import logging

def configure_logging(filename, level, format):
    logging.basicConfig(filename=filename, level=level, format=format)

def process_requests_and_log_to_excel(dataset_path, classifier_agent, output_file):
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        logging.error(f"Dataset file not found: {dataset_path}")
        return

    # Read the dataset
    with open(dataset_path, 'r') as file:
        try:
            dataset = json.load(file)
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON file: {str(e)}")
            return

    # Prepare results list to store the processed results
    results = []

    # Process each request in the dataset
    for entry in dataset:
        user_input = entry.get('request', '')
        expected_category = entry.get('category', 'None')

        # Determine if request has an image context or not
        has_image = expected_category in ["IMAGE_RETRIEVAL_BY_IMAGE", "BINARY_VISUAL_QA", "IMAGE_SEGMENTATION", "OBJECT_COUNTING"]
        try:
            # Call the classifier agent to process the request
            request_existence, detected_category = classifier_agent.request_existence_and_classification(user_input, contains_image=has_image)

            # Append the result to the list
            results.append({
                'Request': user_input,
                'Expected Category': expected_category,
                'Detected Category': detected_category,
                'Request Exists': request_existence
            })

        except Exception as e:
            # Log the error in a log file and add an error entry to the results
            logging.error(f"Error processing request '{user_input}': {str(e)}")
            results.append({
                'Request': user_input,
                'Expected Category': expected_category,
                'Detected Category': 'Error',
                'Request Exists': 'Error'
            })

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Save the results to an Excel file
    df.to_excel(output_file, index=False)
    print(f"Results have been saved to {output_file}")