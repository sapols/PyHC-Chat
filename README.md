# PyHC-Chat Prototype

Explore and ask questions about the [Python in Heliophysics Community](https://pyhc.org) and its [core packages](https://heliopython.org/projects/) using OpenAI's GPT-4 language model.

[![PyHC-Chat-Diagram.png](https://i.postimg.cc/ZqPfkmWZ/Py-HC-Chat-Diagram.png)](https://postimg.cc/XZJKFTKD)

## Prerequisites

- OpenAI API key, set in the environment variable `OPENAI_API_KEY`
- Activeloop token, set in the environment variable `ACTIVELOOP_TOKEN` (get one [here](https://docs.activeloop.ai/storage-and-credentials/user-authentication#authentication-in-programmatic-interfaces))

## Usage
1. Set the environment variables `OPENAI_API_KEY` and `ACTIVELOOP_TOKEN`
2. (Optional) Set your `deeplake_username` in `config.py` if you want your vector store online instead of local
2. Run the script: `pyhc_chat.py`
3. Ask your questions! Type `exit()` to quit

## Key Features
- Has up-to-date knowledge of PyHC and its core packages, facilitated by context retrieval from a DeepLake vector store (this is why an Activeloop token is required)
- Generates detailed answers to user queries based on package repositories' contents
- Spawns helper bots to determine which repos are relevant to the user's prompts and what information should be retrieved from the vector store
- Vector store can be either online or local to your machine
- Uses OpenAI's language model for generating responses
- Optional `verbose` mode to display intermediate model reasoning before responses

## Caveats
- Monitor your OpenAI API usage closely when using the GPT-4 model because it's pretty expensive. It's not hard to rack up a few dollars in usage after just a few conversations.
- When using an online vector store, startup time can take about a minute due to network delays.
- Likewise, responses can be slow because of delays in both querying OpenAI's API and retrieving from the vector store, especially when the helper bots are doing lots of heavy lifting.
- Vector store retrieval doesn't always get the necessary context, so sometimes the model gives responses like _"I can't answer that question based on the provided documents."_
- Chat history length is not intelligently managed yet (that's a TODO), so messages can sometimes exceed the model's max token limit which results in errors. 
- PyHC-Chat is currently only designed to discuss PyHC itself and the seven core packages. 
    - GPT-4 has (outdated) knowledge of other PyHC packages baked into its training data, so it may answer some questions about other packages, but results will vary.
