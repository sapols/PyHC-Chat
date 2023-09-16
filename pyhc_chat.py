# pyhc_chat.py
import os
import time
import sys
import threading
from contextlib import contextmanager
from config import WHITE, GREEN, BLUE, RED, RESET_COLOR
from bot.default_bot import condense_answers, get_default_completion
from bot.helper_bot import HelperBot
from bot.pyhc_bots import *
from bot.repo_selector_bot import RepoSelectorBot
from bot.repo_prompter_bot import RepoPrompterBot
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from tqdm import tqdm


class PyhcChat:
    def __init__(self, use_local_vector_store=True, verbose=False):
        self.use_local_vector_store = use_local_vector_store
        self.verbose = verbose
        self.bots = self.load_bots()
        self.chat_history = []

    def chat(self):
        print("\n=====================\nWELCOME TO PYHC-CHAT!\n=====================")
        time.sleep(1)  # Let bots progress bar finish
        while True:
            try:
                # Get the user's prompt
                user_prompt = input(f"\n{WHITE}Ask a question about PyHC or a core PyHC package (type 'exit()' to quit): {RESET_COLOR}")
                if user_prompt.lower() == "exit()":
                    break

                # Start the animated "Thinking..." in a separate thread
                stop_event = threading.Event()
                t = threading.Thread(target=self.animate_thinking, args=(stop_event,))
                t.start()

                # Get PyHC-Chat's response
                relevant_repos = self.get_relevant_repos(user_prompt)

                if len(relevant_repos) == 1 and relevant_repos[0] == "N/A":
                    response = self.chat_without_vector_store(user_prompt)
                elif len(relevant_repos) > 1:
                    response = self.chat_with_multiple_repos(user_prompt, relevant_repos)
                else:
                    response = self.chat_with_one_repo(user_prompt, relevant_repos[0])

                # Stop the "Thinking..." animation
                stop_event.set()
                t.join()

                # Display PyHC-Chat's response
                print(f"{GREEN}\nANSWER\n{response}{RESET_COLOR}\n")
                self.chat_history.append(HumanMessage(content=user_prompt))
                self.chat_history.append(AIMessage(content=response))
            except Exception as e:
                # Stop the "Thinking..." animation then display the error and move on
                stop_event.set()
                t.join()
                print(f"{RED}An error occurred: {e}{RESET_COLOR}")

    # -------------- Helper Functions ----------------------------------------------------------------------------------

    @staticmethod
    def animate_thinking(event):
        dots = 1
        while not event.is_set():
            dots = (dots % 3) + 1  # cycles through 1, 2, 3 dots
            sys.stdout.write('\r' + WHITE + 'Thinking' + '.' * dots + ' ' * (3 - dots) + RESET_COLOR)
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write('\r' + ' ' * 15 + '\r')  # Clear the line

    @staticmethod
    def get_pyhc_bot_classes():
        # List to store the bot class objects
        bot_class_objects = []
        # Iterate over all items in the current module's dictionary
        for name, obj in globals().items():
            # Check if the item is a class, a subclass of HelperBot, and not HelperBot itself
            if isinstance(obj, type) and issubclass(obj, HelperBot) and obj != HelperBot:
                bot_class_objects.append(obj)
        return bot_class_objects

    @staticmethod
    def get_pyhc_bot_names():
        # List to store the bot names
        bot_names = []
        # Iterate over all items in the current module's dictionary
        for name, obj in globals().items():
            # Check if the item is a class, a subclass of HelperBot, and not HelperBot itself
            if isinstance(obj, type) and issubclass(obj, HelperBot) and obj != HelperBot:
                bot_names.append(obj.REPO_NAME)
        return bot_names

    def load_bots(self):
        @contextmanager
        def suppress_stdout():
            # Hide the DeepLake print statements on startup
            with open(os.devnull, 'w') as fnull:
                old_stdout = sys.stdout
                sys.stdout = fnull
                try:
                    yield
                finally:
                    sys.stdout = old_stdout
        bot_classes = self.get_pyhc_bot_classes()
        bot_names = self.get_pyhc_bot_names()
        with suppress_stdout():
            bots = {}
            for bot_class, bot_name in tqdm(zip(bot_classes, bot_names), total=len(bot_classes), desc="Loading helper bots"):
                bots[bot_name] = bot_class(use_local_vector_store=self.use_local_vector_store)
        return bots

    def get_relevant_repos(self, user_prompt):
        relevant_repos = RepoSelectorBot().determine_relevant_repos(self.chat_history, user_prompt)
        if self.verbose:
            print(f"{BLUE}\nRELEVANT REPO(S)\n{', '.join(relevant_repos)}{RESET_COLOR}\n")
        return relevant_repos

    def chat_without_vector_store(self, user_prompt):
        # Let model answer without vector store retrieval
        return get_default_completion(self.chat_history, user_prompt)

    def chat_with_one_repo(self, user_prompt, repo):
        # Chatting with just one repo
        qa = self.bots[repo].get_qa_chain()
        result = qa({"question": user_prompt, "chat_history": self.chat_history})
        return result['answer']

    def chat_with_multiple_repos(self, user_prompt, repos):
        # Chatting with multiple repos
        package_questions = RepoPrompterBot(repos).formulate_package_questions(self.chat_history, user_prompt)
        if self.verbose:
            print(f"{BLUE}\nPACKAGE QUESTIONS")
            for package, question in package_questions.items():
                print(f"{package}: \"{question}\"\n")
            print(f"{RESET_COLOR}")
        package_answers = {}
        for package, package_question in package_questions.items():
            qa = self.bots[package].get_qa_chain()
            result = qa({"question": package_question, "chat_history": self.chat_history})  # TODO: does it need chat_history? Or should we one-shot prompt?
            package_answers[package] = result['answer']
        if self.verbose:
            print(f"{BLUE}\nPACKAGE ANSWERS")
            for package, answer in package_answers.items():
                print(f"{package}: \n\"{answer}\"\n")
            print(f"{RESET_COLOR}")
        return condense_answers(user_prompt, package_answers)


# -------------- Main Execution ----------------------------------------------------------------------------------------


if __name__ == "__main__":
    PyhcChat(use_local_vector_store=True, verbose=True).chat()
