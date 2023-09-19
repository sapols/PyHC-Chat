# repo_prompter_bot.py
import re
import json
from typing import Dict
from config import model_name, secondary_model_name
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage


class RepoPrompterBot:
    def __init__(self, repos_to_prompt):
        self.repos_to_prompt = repos_to_prompt
        self.chat = ChatOpenAI(model_name=model_name, temperature=0.3)
        self.chat_list = [SystemMessage(content=f"""
You are RepoPrompterBot, a pivotal component of the PyHC-Chat system—a custom chatbot designed to provide users with up-to-date information about the Python in Heliophysics Community (PyHC) and its core Python packages.

Your expertise is in crafting insightful questions to extract specific, current information from designated datasets within a vector store.

Your critical assignment is:

1. Examine Contextual Inputs:
   - Review the chat history of the session.
   - Pay special attention to the latest user prompt and its relevance to the provided dataset name(s): {', '.join(self.repos_to_prompt)}.

2. Understand the Dataset(s):
   - Recognize that the name(s) you've been given map to a dataset in the vector store. These datasets encapsulate vector embeddings of files from the corresponding package's GitHub repo or, in the case of 'pyhc', the source code files of PyHC's website.

3. Formulate Targeted Questions for Retrieval:
   - For the given dataset name(s) ({', '.join(self.repos_to_prompt)}), craft a concise and relevant question. This question will guide a semantic search within the vector store, aiming to retrieve the most pertinent information from the dataset in relation to the user's query.

4. Structure Your Response:
   - Arrange your answers as:
     ```
     {{first dataset name}}: {{question for first dataset}}
     {{second dataset name}}: {{question for second dataset}}
     ... and so on.
     ```
""")]

    def formulate_repo_questions(self, chat_history, prompt) -> Dict[str, str]:
        # TODO: catch error "This model's maximum context length is 4097 tokens..." (see: https://github.com/search?q=%22This+model%27s+maximum+context+length+is%22&type=code)
        convo = self.chat_list + chat_history + [HumanMessage(content=prompt)]
        response = self.chat(convo).content
        try:
            package_questions = self.parse_output_dict_without_gpt(response)
        except Exception as e:
            package_questions = self.parse_output_dict_with_gpt(response)
        return package_questions

    def parse_output_dict_without_gpt(self, prompter_response) -> Dict[str, str]:
        # Try to parse the response without GPT first
        lines = prompter_response.strip().split('\n')
        parsed_dict = {}
        # Regular expression to match lines with the format "package: question"
        # Allows optional single or double quotes around package names/questions, is flexible with whitespace and case
        pattern = re.compile(r'^\s*["\']?(\w+)["\']?\s*:\s*["\']?(.*?)["\']?\s*$', re.IGNORECASE)
        for line in lines:
            match = pattern.match(line)
            if match:
                # Case-insensitive matching
                key = match.group(1).lower()  # Assumes pyhc_bot REPO_NAMEs are lowercase
                value = match.group(2)
                if key in [repo.lower() for repo in self.repos_to_prompt]:
                    parsed_dict[key] = value
        if self.repo_questions_are_valid(parsed_dict):
            return parsed_dict
        else:
            raise ValueError("Invalid dict parsed without GPT.")

    def parse_output_dict_with_gpt(self, prompter_response) -> Dict[str, str]:
        # Try a couple times to parse the response with GPT (using secondary_model_name)
        model = ChatOpenAI(model_name=secondary_model_name, temperature=0)
        chat_list = [SystemMessage(content=f"""
You will be given a poorly formatted string containing questions for helper bots about particular Python packages. They'll likely be in the following format: "{{first package}}: {{question}}\n{{second package}}: {{question}}\netc..." although the format may vary.

Your job is to respond with a properly formatted markdown code snippet of JSON where the keys are the package names and the values are the corresponding questions. For example, if you are given:

"hapiclient: What is HAPI?\npysat: How does pysat work?" then you must return:

```json
{{
    "hapiclient": "What is HAPI?",
    "pysat": "How does pysat work?"
}}
```

Here's the poorly formatted string: 
""")]
        chat_list.append(HumanMessage(content=prompter_response))
        for _ in range(2):
            try:
                _out = model(chat_list).content
                if "```json" not in _out:
                    raise RuntimeError(
                        f"Got invalid return object. Expected markdown code snippet with JSON object, but got:\n{_out}"
                    )
                json_str = _out.split("```json")[1].strip().strip("```").strip()
                repo_questions = json.loads(json_str)
                if self.repo_questions_are_valid(repo_questions):
                    return repo_questions
                else:
                    raise ValueError("Could not parse prompter's response to Dict: " + prompter_response)
            except (ValueError, RuntimeError):
                continue  # Retry
        raise ValueError("Could not parse output to Dict after 2 attempts. Last output: " + _out)

    def repo_questions_are_valid(self, repo_questions):
        if set(repo_questions.keys()) != set(self.repos_to_prompt):  # Check if the keys match
            return False
        for value in repo_questions.values():
            if not isinstance(value, str):  # Check if the values are strings
                return False
        return True
