# default_bot.py
from config import model_name
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage


def condense_answers(prompt, package_statements):
    chat = ChatOpenAI(model_name=model_name)
    chat_list = [HumanMessage(content=f"""
I need you to condense multiple statements related to a question I asked into one cogent answer. The question I asked was:

"{prompt}"

Use the following statements as context to formulate a final answer to my question (without directly referencing the fact that your final answer was pieced together from the statements):

--
{expand_package_statements(package_statements)}
--
""")]
    return chat(chat_list).content


def expand_package_statements(package_statements):
    expanded = ""
    for i, (package, statement) in enumerate(package_statements.items(), start=1):
        expanded += f"{i}. (from {package}): `{statement}`\n\n"
    return expanded


def get_default_completion(chat_history, prompt):
    chat = ChatOpenAI(model_name=model_name)
    chat_list = chat_history
    chat_list.append(HumanMessage(content=prompt))
    return chat(chat_list).content


class DefaultBot:
    def __init__(self, chat_history):
        self.chat = ChatOpenAI(model_name=model_name)
        self.chat_list = chat_history

    def get_completion(self):
        return self.chat(self.chat_list).content
