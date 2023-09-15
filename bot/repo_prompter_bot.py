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
RepoPrompterBot,

You have a specific role in an intricate system: to intelligently allocate and direct questions to expert helper bots based on user prompts and the relevant Python packages they pertain to.

**Here's your task**:

1. **Inputs to Examine**:
   - The chat history of the session.
   - The latest user prompt, which is related to the following Python packages (that you are already somewhat familiar with): {', '.join(self.repos_to_prompt)}.

2. **Craft Questions**:
   - For each of those packages ({', '.join(self.repos_to_prompt)}), formulate a concise, targeted question for the package's helper bot. The goal of this question should be to gather the context needed to accurately address the user's last prompt.

3. **Format Your Responses**:
   - Structure your answers as:
     ```
     {{first package}}: {{question for first package}}
     {{second package}}: {{question for second package}}
     ... continue in this pattern.
     ```

Rely on your existing knowledge of the Python packages to enhance the precision and relevance of your questions. 
""")]

    def formulate_package_questions(self, chat_history, prompt):
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
