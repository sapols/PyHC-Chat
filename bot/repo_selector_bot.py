# repo_selector_bot.py
import re
from typing import List
from config import model_name, secondary_model_name
from bot.pyhc_bots import *
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage


class RepoSelectorBot:
    def __init__(self):
        self.possible_packages = self.get_possible_packages()
        self.chat = ChatOpenAI(model_name=model_name, temperature=0.0)
        self.chat_list = [SystemMessage(content=f"""
You are RepoSelectorBot, an integral component of the PyHC-Chat system designed by the Python in Heliophysics Community (PyHC) to answer questions about PyHC and its {str(len(self.possible_packages))} core Python packages. 

PyHC-Chat is powered by OpenAI's GPT model, which inherently knows about PyHC and the core packages. However, its knowledge has a cutoff in 2021, making some of its information outdated. To compensate, PyHC-Chat leverages vector store retrieval to provide users with the most recent information from these packages and PyHC's overarching activities.

Your critical assignment is:

1. Understand the Datasets: The vector store contains datasets from the latest versions of GitHub repositories for each package and the PyHC website's source files. The dataset names are:
{self.expand_list(self.possible_packages)}
- pyhc (from the PyHC website's GitHub repo)

2. Monitor the Dialogue: Continuously monitor the dialogue between the user and the PyHC-Chat system. Factor in your intrinsic knowledge of these packages and the ongoing context of the conversation.

3. Determine Retrieval Needs:
- If a user's question pertains directly to the overarching Python in Heliophysics Community (PyHC) itself—like their meetings, events, or general activities—respond with "pyhc".
- If the user's query might benefit from the latest source code or documentation of one or more of the seven packages, decide which datasets are necessary.

4. Decide & Understand the Impacts: Your decisions are critical. Responding with dataset names triggers vector store retrieval for each dataset, which:
- Adds delay to the system's response.
- Risks breaking the seamless chat experience if retrieved info doesn't align with the user's query.
- Is essential for ensuring the user receives up-to-date information.

Provide a comma-separated list of relevant dataset names, or "N/A" if vector store retrieval isn't deemed necessary. Strive for a balance: minimize retrievals for a seamless experience but ensure accuracy and up-to-dateness when needed.
""")]

    @staticmethod
    def get_possible_packages():
        # Get the PyHC package names from `bot.pyhc_bots`
        possible_packages = []
        # Iterate over all items in the current module's dictionary
        for name, obj in globals().items():
            # Check if the item is a class and a subclass of HelperBot
            if isinstance(obj, type) and issubclass(obj, HelperBot) and obj != HelperBot:
                # Exclude "pyhc" from the list
                if obj.REPO_NAME != "pyhc":
                    possible_packages.append(obj.REPO_NAME)
        return possible_packages

    @staticmethod
    def expand_list(possible_packages):
        return "\n".join([f"- {package} (from the `{package}` GitHub repo)" for package in possible_packages])

    def determine_relevant_repos(self, chat_history, prompt) -> List[str]:
        # TODO: catch error "This model's maximum context length is 4097 tokens..." (see: https://github.com/search?q=%22This+model%27s+maximum+context+length+is%22&type=code)
        convo = self.chat_list + chat_history + [HumanMessage(content=prompt)]
        response = self.chat(convo).content
        try:
            relevant_repos_list = self.parse_output_list_without_gpt(response)
        except Exception as e:
            relevant_repos_list = self.parse_output_list_with_gpt(response)
        return relevant_repos_list

    def parse_output_list_without_gpt(self, selector_response) -> List[str]:
        # Try to parse the response without GPT first. Handles responses like:
        # "['hapiclient', "sunpy"]"
        # "SunPy,,hapiclient"
        # "sunpy, hapiclient{ }"
        # "(sunpy), [hapiclient]"
        # "N/A"
        cleaned_response = re.sub(r'[\[\]"\'()' '{}]', '', selector_response)  # Remove unwanted characters
        cleaned_response = cleaned_response.strip().lower()  # Assumes pyhc_bot REPO_NAMEs are lowercase
        if not cleaned_response:
            raise ValueError("Empty response after cleaning.")
        if cleaned_response == "n/a":
            return ["N/A"]
        # Split by comma and filter out empty strings
        parsed_list = [item.strip() for item in cleaned_response.split(",") if item.strip()]
        if self.package_list_is_valid(parsed_list):
            return parsed_list
        else:
            raise ValueError("Invalid list parsed without GPT.")

    def parse_output_list_with_gpt(self, selector_response) -> List[str]:
        # Try a couple times to parse the response with GPT (using secondary_model_name)
        model = ChatOpenAI(model_name=secondary_model_name, temperature=0)
        chat_list = [HumanMessage(content=f"""
The following text should be a list of comma-separated names, specifically one or more of the following: `{', '.join(self.possible_packages + ['pyhc'])}` (or just `N/A`).
However, the text may contain (1) more than just the list of names (e.g. square brackets, quotation marks, extra words/sentences), in which case it is your job to extract the list from the text, or (2) none of the names, in which case you return `N/A`. Do NOT surround your response with square brackets nor quotes. E.g. to be clear, `['hapiclient', 'sunpy']` would be an INCORRECT response while `hapiclient, sunpy` would be correct.

Here is the text:
"{selector_response}"

Your response should be a list of comma separated values, e.g.: `foo, bar, baz` (or simply `N/A`).
""")]
        for _ in range(2):
            try:
                _out = model(chat_list).content
                parsed_list = _out.strip().split(", ")
                parsed_list = parsed_list if "N/A" not in parsed_list else ["N/A"]  # Keep only "N/A" if it's in there
                if self.package_list_is_valid(parsed_list):
                    return parsed_list
                else:
                    # Try parsing without GPT once more
                    fallback_parsed_list = self.parse_output_list_without_gpt(str(parsed_list))
                    if self.package_list_is_valid(fallback_parsed_list):
                        return fallback_parsed_list
                    else:
                        raise ValueError("Could not parse selector's response to List: " + selector_response)
            except (ValueError, RuntimeError):
                continue  # Retry
        raise ValueError("Could not parse selector's response to List after 2 attempts: " + selector_response)

    def package_list_is_valid(self, package_list):
        if not isinstance(package_list, list):
            return False
        if len(package_list) == 1 and package_list[0] == "N/A":
            return True
        for package in package_list:
            if not isinstance(package, str) or package not in self.possible_packages + ['pyhc']:
                return False
        return True
