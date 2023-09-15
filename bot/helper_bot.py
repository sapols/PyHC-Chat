# helper_bot.py
from config import model_name
import deeplake
from deeplake.util.exceptions import DatasetHandlerError
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake
import subprocess
import tempfile


EMBEDDINGS = OpenAIEmbeddings(disallowed_special=())


def dataset_exists(dataset_path):
    try:
        # Attempt to load the dataset
        ds = deeplake.load(dataset_path)
        ds_existed = True  # If this line is reached, the dataset exists
    except DatasetHandlerError:
        # This exception is raised if the dataset doesn't exist
        ds_existed = False
    return ds_existed


def clone_github_repo(github_url, local_path):
    try:
        subprocess.run(['git', 'clone', github_url, local_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")
        return False


def store_vector_embeddings(dataset_path, github_url, suffixes=[".py"]):
    with tempfile.TemporaryDirectory() as root_dir:
        if clone_github_repo(github_url, root_dir):
            # Load files
            loader = GenericLoader.from_filesystem(
                root_dir,
                glob="**/*",
                suffixes=suffixes,
                parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
            )
            docs = loader.load()
            # Chunk files
            python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=2000,
                                                                           chunk_overlap=200)
            texts = python_splitter.split_documents(docs)
            # Make dataset
            db = DeepLake(dataset_path=dataset_path, embedding=EMBEDDINGS)
            db.add_documents(texts)
            return db


class HelperBot:
    REPO_NAME = None  # These will be overridden in subclasses
    REPO_URL = None
    SUFFIXES = None

    def __init__(self, package_name, github_url, suffixes=[".py"]):
        self.package_name = package_name
        self.dataset_path = f"hub://sapols/{package_name}"
        if dataset_exists(self.dataset_path):
            self.repo_ds = DeepLake(dataset_path=self.dataset_path, read_only=True, embedding=EMBEDDINGS)
        else:
            print(f"Creating a new vector store at: {self.dataset_path} ...")
            self.repo_ds = store_vector_embeddings(self.dataset_path, github_url, suffixes)
            print(f"Created a new vector store at: {self.dataset_path}")

    def get_qa_chain(self, distance_metric='cos', fetch_k=100, maximal_marginal_relevance=True, k=10):
        retriever = self.repo_ds.as_retriever()
        retriever.search_kwargs['distance_metric'] = distance_metric
        retriever.search_kwargs['fetch_k'] = fetch_k
        retriever.search_kwargs['maximal_marginal_relevance'] = maximal_marginal_relevance
        retriever.search_kwargs['k'] = k
        qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model=model_name), retriever=retriever)  # TODO: this is a little too strict; doesn't use its own knowledge when the answers aren't in the retrieved docs. E.g. we'll get "I'm sorry, but the provided context does not include information about HAPI and its data downloading process. Therefore, I can't provide a comparison between the data downloading processes of HAPI and pysat."
        return qa
