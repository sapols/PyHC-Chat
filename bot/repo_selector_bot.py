# repo_selector_bot.py
from typing import List
from config import model_name
from bot.pyhc_bots import *
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.output_parsers import CommaSeparatedListOutputParser


def get_possible_repos():
    possible_repos = []
    # Iterate over all items in the pyhc_bots' module dictionary
    for name, obj in globals().items():
        # Check if the item is a class and a subclass of HelperBot
        if isinstance(obj, type) and issubclass(obj, HelperBot) and obj != HelperBot:
            # Exclude "pyhc" from the list
            if obj.REPO_NAME != "pyhc":
                possible_repos.append(obj.REPO_NAME)
    return possible_repos


class RepoSelectorBot:
    def __init__(self):
        self.possible_repos = get_possible_repos()
        self.chat = ChatOpenAI(model_name=model_name, temperature=0.0)
        # TODO: I'd get better performance out of RepoSelectorBot if its first response could reason through why or why not packages/pyhc are relevant, then end with the list. The next parse step would then extract the list.
        self.chat_list = [SystemMessage(content=f"""
RepoSelectorBot, you're part of a chat system guiding users on {str(len(self.possible_repos))} specific Python packages from the Python in Heliophysics Community (PyHC): {', '.join(self.possible_repos)}.

Your core responsibility is to:
1. Continually observe the dialogue between the user and the system.

2. First and foremost, if the user's query pertains directly and specifically to the overarching "Python in Heliophysics Community (PyHC)" itself—like their meetings, events, or general activities—respond with just "pyhc".

3. If the query relates to one of the {str(len(self.possible_repos))} individual packages, determine which of them are relevant. Factor in both your knowledge of these packages and the ongoing context of the conversation. Especially if the user's prompt seems to follow-up or reflect on a recent system response, consider the package or packages mentioned in that response.

4. Respond with a comma-separated list of relevant package names, "pyhc" when appropriate, or "N/A" if none apply.

Accuracy is paramount; subsequent system actions rely on your decisions.
""")]

    def determine_relevant_repos(self, chat_history, prompt) -> List[str]:
        # TODO: catch error "This model's maximum context length is 4097 tokens..." (see: https://github.com/search?q=%22This+model%27s+maximum+context+length+is%22&type=code)
        convo = self.chat_list + chat_history
        convo.append(HumanMessage(content=prompt))
        answer = self.chat(convo).content
        relevant_repos_list = self.parse_output_list(answer, self.possible_repos)
        return relevant_repos_list

    def parse_output_list(self, selector_response, possible_repos) -> List[str]:
        for i in range(5):
            try:
                comma_separated_list_parser = CommaSeparatedListOutputParser()
                format_instructions = comma_separated_list_parser.get_format_instructions()
                prompt = PromptTemplate(
                    template=f"""
The following text should be a list of comma-separated names, specifically one or more of the following: "{', '.join(possible_repos + ['pyhc'])}" (or just "N/A").
However, the text may contain (1) more than just the list of names (e.g. square brackets, quotation marks, extra words/sentences), in which case it is your job to extract the list from the text, or (2) none of the names, in which case you return "N/A". Do NOT surround your response with square brackets. Here is the text:

\"{{selector_response}}\"

{{format_instructions}} (or simply "N/A"). E.g. to be clear, "['hapiclient', 'sunpy']" would be an INCORRECT response while "hapiclient, sunpy" would be correct. 
""",
                    input_variables=["selector_response"],
                    partial_variables={"format_instructions": format_instructions}
                )
                model = OpenAI(temperature=0)
                _in = prompt.format(selector_response=selector_response)
                _out = model(_in)
                parsed_list = comma_separated_list_parser.parse(_out)  # returns a list by doing `text.strip().split(", ")`
                parsed_list = parsed_list if "N/A" not in parsed_list else ["N/A"]  # If "N/A" is an element, make it the only element
                if self.package_list_is_valid(parsed_list, possible_repos):
                    return parsed_list
                else:
                    raise ValueError("Could not parse selector's response to List after: " + selector_response)
            except ValueError:
                pass  # Retry
        raise ValueError("Could not parse selector's response to List after 5 attempts: " + selector_response)

    @staticmethod
    def package_list_is_valid(package_list, possible_packages):
        if not isinstance(package_list, list):
            return False
        if len(package_list) == 1 and package_list[0] == "N/A":
            return True
        for package in package_list:
            if not isinstance(package, str) or package not in possible_packages + ['pyhc']:
                return False
        return True
