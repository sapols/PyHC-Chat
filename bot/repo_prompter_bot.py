# repo_prompter_bot.py
import json
from config import model_name
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

    def formulate_repo_questions(self, chat_history, prompt):
        # TODO: catch error "This model's maximum context length is 4097 tokens..." (see: https://github.com/search?q=%22This+model%27s+maximum+context+length+is%22&type=code)
        convo = self.chat_list + chat_history
        convo.append(HumanMessage(content=prompt))
        package_questions = self.parse_output(self.chat(convo).content)
        return package_questions

    def parse_output(self, output):
        model = ChatOpenAI(model_name=model_name, temperature=0)
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
        chat_list.append(HumanMessage(content=output))
        json_out = model(chat_list).content
        json_obj = self.parse_json(json_out, model)
        return json_obj

    @staticmethod
    def parse_json(output, model):
        for i in range(5):
            try:
                if "```json" not in output:
                    raise RuntimeError(
                        f"Got invalid return object. Expected markdown code snippet with JSON "
                        f"object, but got:\n{output}"
                    )
                json_str = output.split("```json")[1].strip().strip("```").strip()
                json_obj = json.loads(json_str)
                return json_obj
            except ValueError:
                output = model(output)
            except RuntimeError:
                output = model(output)
        raise ValueError("Could not parse output to JSON after 5 attempts. Last output: " + output)
