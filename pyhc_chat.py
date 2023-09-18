# pyhc_chat.py
import argparse
import os
import sys
import time
import threading
from contextlib import contextmanager
from config import WHITE, GREEN, BLUE, RED, RESET_COLOR
from bot.default_bot import answer_with_context, let_pyhc_chat_answer
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
        self.stop_event = threading.Event()
        self.t = None

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
                self.start_waiting_animation()

                # Get PyHC-Chat's response
                relevant_repos = self.get_relevant_repos(user_prompt)

                if len(relevant_repos) == 1 and relevant_repos[0] == "N/A":
                    # No vector store retrieval
                    response = self.chat_without_vector_store(user_prompt)
                elif len(relevant_repos) > 1:
                    # Retrieve from multiple vector store datasets
                    response = self.chat_with_multiple_repos(user_prompt, relevant_repos)
                else:
                    # Retrieve from one vector store dataset
                    response = self.chat_with_one_repo(user_prompt, relevant_repos[0])

                # Stop the "Thinking..." animation
                self.stop_waiting_animation()

                # Display PyHC-Chat's response
                print(f"{GREEN}\nANSWER\n{response}{RESET_COLOR}\n")
                self.chat_history.append(HumanMessage(content=user_prompt))
                self.chat_history.append(AIMessage(content=response))
            except Exception as e:
                # Stop the "Thinking..." animation then display the error and move on
                self.stop_waiting_animation()
                print(f"{RED}An error occurred: {e}{RESET_COLOR}")
                raise e

    # -------------- Helper Functions ----------------------------------------------------------------------------------

    @staticmethod
    def animate_waiting(event, repo_name=None):
        dots = 1
        while not event.is_set():
            dots = (dots % 3) + 1  # cycles through 1, 2, 3 dots
            if repo_name:
                sys.stdout.write(
                    '\r' + WHITE + f'Searching {repo_name} contents' + '.' * dots + ' ' * (3 - dots) + RESET_COLOR)
            else:
                sys.stdout.write('\r' + WHITE + 'Thinking' + '.' * dots + ' ' * (3 - dots) + RESET_COLOR)
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write('\r' + ' ' * 50 + '\r')  # Clear the line

    def start_waiting_animation(self, repo_name=None):
        self.stop_event = threading.Event()
        self.t = threading.Thread(target=self.animate_waiting, args=(self.stop_event, repo_name))
        self.t.start()

    def stop_waiting_animation(self):
        self.stop_event.set()
        self.t.join()

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
        return let_pyhc_chat_answer(self.chat_history, user_prompt)

    def chat_with_one_repo(self, user_prompt, repo):
        # Chat with one repo using vector store retrieval
        qa = self.bots[repo].get_qa_chain()
        # Change "Thinking..." animation to "Searching {repo} contents..."
        self.stop_waiting_animation()
        self.start_waiting_animation(repo)
        # Get helper bot answer
        result = qa({"question": user_prompt, "chat_history": self.chat_history})
        # Stop animation
        self.stop_waiting_animation()
        # Start "Thinking..." animation one last time
        self.start_waiting_animation()
        context = {repo: result['answer']}
        return answer_with_context(user_prompt, context)

    def chat_with_multiple_repos(self, user_prompt, repos):
        # Chat with potentially multiple repos using vector store retrieval
        repo_questions = RepoPrompterBot(repos).formulate_repo_questions(self.chat_history, user_prompt)
        if self.verbose:
            print(f"{BLUE}\nREPO QUESTION(S)")
            for repo, question in repo_questions.items():
                print(f"{repo}: \"{question}\"\n")
            print(f"{RESET_COLOR}")
        repo_answers = {}
        for repo, repo_question in repo_questions.items():
            qa = self.bots[repo].get_qa_chain()
            # Change "Thinking..." animation to "Searching {repo} contents..."
            self.stop_waiting_animation()
            self.start_waiting_animation(repo)
            # Get helper bot answer
            result = qa({"question": repo_question, "chat_history": self.chat_history})  # TODO: does it need chat_history? Or should we one-shot prompt?
            # Stop animation
            self.stop_waiting_animation()
            # Store answer
            repo_answers[repo] = result['answer']
        if self.verbose:
            print(f"{BLUE}\nREPO ANSWER(S)")
            for repo, answer in repo_answers.items():
                print(f"{repo}: \n\"{answer}\"\n")
            print(f"{RESET_COLOR}")
        # Start "Thinking..." animation one last time
        self.start_waiting_animation()
        return answer_with_context(user_prompt, repo_answers)


# -------------- Main Execution ----------------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chat about PyHC and its core packages with PyHC-Chat.')

    parser.add_argument('-o', '--online_vector_store', action='store_true',
                        help='Flag to use an online vector store. Default is to use a local vector store.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Flag for verbose mode. Default is False.')
    # TODO: add a flag to optionally display documents retrieved from the vector store
    args = parser.parse_args()

    use_local_vector_store = not args.online_vector_store
    PyhcChat(use_local_vector_store, args.verbose).chat()
