# pyhc_chat.py
import os
import time
import sys
import threading
from contextlib import contextmanager
from config import WHITE, GREEN, BLUE, RED, RESET_COLOR
from bot.default_bot import condense_answers, get_default_completion
from bot.pyhc_bots import HapiBot, KamodoBot, PlasmapyBot, PysatBot, PyspedasBot, SpacepyBot, SunpyBot, PyhcBot
from bot.repo_selector_bot import RepoSelectorBot
from bot.repo_prompter_bot import RepoPrompterBot
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from tqdm import tqdm


def initialize():
    def load_bots():
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
        bot_classes = [HapiBot, KamodoBot, PlasmapyBot, PysatBot, PyspedasBot, SpacepyBot, SunpyBot, PyhcBot]
        bot_names = ["hapiclient", "kamodo", "plasmapy", "pysat", "pyspedas", "spacepy", "sunpy", "pyhc"]
        with suppress_stdout():
            bots = {}
            for bot_class, bot_name in tqdm(zip(bot_classes, bot_names), total=len(bot_classes), desc="Loading helper bots"):
                bots[bot_name] = bot_class()
        return bots
    bots = load_bots()
    chat_history = []
    return bots, chat_history


bots, chat_history = initialize()


def main(verbose=False, local_vector_store=False):
    # TODO: implement local vector store!
    print("\n=====================\nWELCOME TO PYHC-CHAT!\n=====================")
    while True:
        try:
            # Get the user's prompt
            user_prompt = input(f"\n{WHITE}Ask a question about PyHC or a core PyHC package (type 'exit()' to quit): {RESET_COLOR}")
            if user_prompt.lower() == "exit()":
                break

            # Start the animated "Thinking..." in a separate thread
            stop_event = threading.Event()
            t = threading.Thread(target=animate_thinking, args=(stop_event,))
            t.start()

            # Get PyHC-Chat's response
            relevant_repos = get_relevant_repos(user_prompt, verbose)

            if len(relevant_repos) == 1 and relevant_repos[0] == "N/A":
                response = chat_without_vector_store(user_prompt)
            elif len(relevant_repos) > 1:
                response = chat_with_multiple_repos(user_prompt, relevant_repos, verbose)
            else:
                response = chat_with_one_repo(user_prompt, relevant_repos[0])

            # Stop the "Thinking..." animation
            stop_event.set()
            t.join()

            # Display PyHC-Chat's response
            print(f"{GREEN}\nANSWER\n{response}{RESET_COLOR}\n")
            chat_history.append(HumanMessage(content=user_prompt))
            chat_history.append(AIMessage(content=response))
        except Exception as e:
            stop_event.set()
            t.join()
            print(f"{RED}An error occurred: {e}{RESET_COLOR}")


# -------------- Helper Functions --------------------------------------------------------------------------------------


def get_relevant_repos(user_prompt, verbose):
    relevant_repos = RepoSelectorBot().determine_relevant_repos(chat_history, user_prompt)
    if verbose:
        print(f"{BLUE}\nRELEVANT REPO(S)\n{', '.join(relevant_repos)}{RESET_COLOR}\n")
    return relevant_repos


def animate_thinking(event):
    dots = 1
    while not event.is_set():
        dots = (dots % 3) + 1  # cycles through 1, 2, 3 dots
        sys.stdout.write('\r' + WHITE + 'Thinking' + '.' * dots + ' ' * (3 - dots) + RESET_COLOR)
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write('\r' + ' ' * 15 + '\r')  # Clear the line


def chat_without_vector_store(user_prompt):
    # Let model answer without vector store retrieval
    return get_default_completion(chat_history, user_prompt)


def chat_with_one_repo(user_prompt, repo):
    # Chatting with just one repo
    qa = bots[repo].get_qa_chain()
    result = qa({"question": user_prompt, "chat_history": chat_history})
    return result['answer']


def chat_with_multiple_repos(user_prompt, repos, verbose):
    # Chatting with multiple repos
    package_questions = RepoPrompterBot(repos).formulate_package_questions(chat_history, user_prompt)
    if verbose:
        print(f"{BLUE}\nPACKAGE QUESTIONS")
        for package, question in package_questions.items():
            print(f"{package}: \"{question}\"\n")
        print(f"{RESET_COLOR}")
    package_answers = {}
    for package, package_question in package_questions.items():
        qa = bots[package].get_qa_chain()
        result = qa({"question": package_question, "chat_history": chat_history})  # TODO: does it need chat_history? Or should we one-shot prompt?
        package_answers[package] = result['answer']
    if verbose:
        print(f"{BLUE}\nPACKAGE ANSWERS")
        for package, answer in package_answers.items():
            print(f"{package}: \n\"{answer}\"\n")
        print(f"{RESET_COLOR}")
    return condense_answers(user_prompt, package_answers)


# -------------- Main Execution ----------------------------------------------------------------------------------------


if __name__ == "__main__":
    local_vector_store = False
    verbose = True
    main(verbose)
