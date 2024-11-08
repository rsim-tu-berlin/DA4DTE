# Task Interpreter

The Task Interpreter main funcionalities are: the engine selection to fulfill the user's requests, and the assistant's conversational functionality. Additionally, for the project's needs, it is responsible for limiting the requests sent to the *Image Retrieval by Caption* engine within the vessel domain and determining whether the *Image Retrieval by Image* requests are cross-modal.
The Task Interpreter is implemented as a multi-agent system, with the agents being: a request classifier, a conversational (chat) agent and the helper ones about vessels and image types.  

## Performance
- Dataset and Documentation: `requests_dataset` directory.

- Evaluation Results: `classifier_evaluation.ipynb`.

### Prerequisites
1. **Create and activate the Conda environment:**

   ```
   conda your_env_name create -f env.yaml
   conda activate your_env_name
   ```

2. **Set the OpenAI API key:**

   ```
   export OPENAI_API_KEY='your_openai_api_key_here'
   ```

## Run 

The main script is `main.py`. For each request, you'll be asked whether it includes an accompanying image. Type 'exit' to finish and save the dialog.
You can find the dialogues saved in the `dialogues` directory.

