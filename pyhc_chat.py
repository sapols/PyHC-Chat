# pyhc_chat.py
import argparse
import os
import sys
import time
import signal
import threading
from contextlib import contextmanager
from config import WHITE, GREEN, BLUE, RED, RESET_COLOR
from bot.pyhc_chat_bot import answer_with_context, let_pyhc_chat_answer
from bot.helper_bot import HelperBot
from bot.pyhc_bots import *
from bot.repo_selector_bot import RepoSelectorBot
from bot.repo_prompter_bot import RepoPrompterBot
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from tqdm import tqdm


class PyHCChat:
    def __init__(self, use_local_vector_store=True, verbose=False):
        self.use_local_vector_store = use_local_vector_store
        self.verbose = verbose
        self.bots = self.load_helper_bots()
        self.chat_history = []
        self.stop_event = threading.Event()
        self.thread = None
        signal.signal(signal.SIGINT, self.signal_handler)

    def chat(self):
        print("\n=====================\nWELCOME TO PYHC-CHAT!\n=====================")
        time.sleep(1)  # Let bots progress bar finish
        while True:
            try:
                # Get the user's prompt
                print(f"\n{WHITE}Ask a question about PyHC or a core PyHC package (type 'exit()' to quit): ", end="")
                user_prompt = input(f"{GREEN}")  # User's input will be green
                print(f"{RESET_COLOR}", end="")  # Reset color back to normal
                if user_prompt.lower() == "exit()":
                    print("\nExiting...")
                    sys.exit(0)

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
                print(f"{GREEN}\nANSWER\n{WHITE}{response}{RESET_COLOR}\n")
                self.chat_history.append(HumanMessage(content=user_prompt))
                self.chat_history.append(AIMessage(content=response))
            except Exception as e:
                # Stop the "Thinking..." animation then display the error and move on
                self.stop_waiting_animation()
                print(f"{RED}An error occurred: {e}{RESET_COLOR}")

    # -------------- Helper Functions ----------------------------------------------------------------------------------

    @staticmethod
    def animate_waiting(event, message=None):
        # Animate the dots in "Thinking..." / "Searching {repo_name} contents..." / "Writing response..."
        dots = 1
        while not event.is_set():
            dots = (dots % 3) + 1  # Cycle through 1, 2, 3 dots
            if message:
                sys.stdout.write(
                    '\r' + WHITE + message + '.' * dots + ' ' * (3 - dots) + RESET_COLOR)
            else:
                sys.stdout.write('\r' + WHITE + 'Thinking' + '.' * dots + ' ' * (3 - dots) + RESET_COLOR)
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write('\r' + ' ' * 50 + '\r')  # Clear the line

    def start_waiting_animation(self, message=None):
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.animate_waiting, args=(self.stop_event, message))
        self.thread.start()

    def stop_waiting_animation(self):
        self.stop_event.set()
        self.thread.join()

    @staticmethod
    def signal_handler(sig, frame):
        print(f"{RESET_COLOR}")  # Reset the terminal color
        print("\nExiting...")
        sys.exit(0)

    @staticmethod
    def get_pyhc_bot_info():
        # Get the PyHC package helper bots' class objects and names from `bot.pyhc_bots`
        bot_class_objects = []
        bot_names = []
        # Iterate over all items in the current module's dictionary
        for name, obj in globals().items():
            # Check if the item is a class, a subclass of HelperBot, and not HelperBot itself
            if isinstance(obj, type) and issubclass(obj, HelperBot) and obj != HelperBot:
                bot_class_objects.append(obj)
                bot_names.append(obj.REPO_NAME)
        return bot_class_objects, bot_names

    def load_helper_bots(self):
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
        # Load every PyHC package helper bot
        bot_classes, bot_names = self.get_pyhc_bot_info()
        with suppress_stdout():
            bots = {}
            for bot_class, bot_name in tqdm(zip(bot_classes, bot_names), total=len(bot_classes), desc="Loading helper bots"):
                bots[bot_name] = bot_class(use_local_vector_store=self.use_local_vector_store)
        return bots

    def get_relevant_repos(self, user_prompt):
        # Determine which vector store datasets to reach into
        relevant_repos = RepoSelectorBot().determine_relevant_repos(self.chat_history, user_prompt)
        self.stop_waiting_animation()
        if self.verbose:
            print(f"{BLUE}\nRELEVANT REPO(S)\n{', '.join(relevant_repos)}{RESET_COLOR}\n")
        return relevant_repos

    def chat_without_vector_store(self, user_prompt):
        # Let the model answer without vector store retrieval
        self.start_waiting_animation('Writing response')
        return let_pyhc_chat_answer(self.chat_history, user_prompt)

    def chat_with_one_repo(self, user_prompt, repo):
        # Chat with one repo using vector store retrieval
        qa = self.bots[repo].get_qa_chain()
        # Change "Thinking..." animation to "Searching {repo} contents..."
        self.stop_waiting_animation()
        self.start_waiting_animation(f'Searching {repo} contents')
        # Get helper bot answer
        result = qa({"question": user_prompt, "chat_history": self.chat_history})
        # Stop animation
        self.stop_waiting_animation()
        # Start "Writing response..." animation
        self.start_waiting_animation('Writing response')
        context = {repo: result['answer']}
        return answer_with_context(self.chat_history, user_prompt, context)

    def chat_with_multiple_repos(self, user_prompt, repos):
        # Chat with potentially multiple repos using vector store retrieval
        self.start_waiting_animation()
        repo_questions = RepoPrompterBot(repos).formulate_repo_questions(self.chat_history, user_prompt)
        self.stop_waiting_animation()
        if self.verbose:
            print(f"{BLUE}\nREPO QUESTION(S)")
            for repo, question in repo_questions.items():
                print(f"{repo}: \"{question}\"\n")
            print(f"{RESET_COLOR}")
        repo_answers = {}
        for repo, repo_question in repo_questions.items():
            qa = self.bots[repo].get_qa_chain()
            # Start "Searching {repo} contents..."
            self.start_waiting_animation(f'Searching {repo} contents')
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
        # Start "Writing response..." animation
        self.start_waiting_animation('Writing response')
        return answer_with_context(self.chat_history, user_prompt, repo_answers)


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
    PyHCChat(use_local_vector_store, args.verbose).chat()
