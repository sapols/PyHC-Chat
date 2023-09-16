# helper_bot.py
import os
from config import model_name, deeplake_username
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


def dataset_exists_online(dataset_name):
    try:
        # Attempt to load the dataset
        ds = deeplake.load(f"hub://{deeplake_username}/{dataset_name}")
        ds_existed = True  # If this line is reached, the dataset exists
    except DatasetHandlerError:
        # This exception is raised if the dataset doesn't exist
        ds_existed = False
    return ds_existed


def dataset_exists_locally(dataset_name):
    if os.path.isdir(f"vector_store/{dataset_name}"):
        return True
    else:
        return False


def dataset_exists(dataset_name, store_locally):
    if store_locally:
        return dataset_exists_locally(dataset_name)
    else:
        return dataset_exists_online(dataset_name)


def store_vector_embeddings(dataset_name, github_url, suffixes=[".py"], store_locally=False):
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
            if store_locally:
                db = DeepLake(dataset_path=f"vector_store/{dataset_name}", embedding=EMBEDDINGS)
            else:
                db = DeepLake(dataset_path=f"hub://{deeplake_username}/{dataset_name}", embedding=EMBEDDINGS)
            db.add_documents(texts)
            return db


def clone_github_repo(github_url, local_path):
    try:
        subprocess.run(['git', 'clone', github_url, local_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")
        return False


class HelperBot:
    def __init__(self, package_name, github_url, suffixes=[".py"], use_local_vector_store=True):
        if use_local_vector_store:
            dataset_path = f"vector_store/{package_name}"
        else:
            dataset_path = f"hub://{deeplake_username}/{package_name}"
        if not dataset_exists(package_name, use_local_vector_store):
            # store it first
            store_vector_embeddings(package_name, github_url, suffixes, use_local_vector_store)
        self.repo_ds = DeepLake(dataset_path=dataset_path, read_only=True, embedding=EMBEDDINGS)

    def get_qa_chain(self, distance_metric='cos', fetch_k=100, maximal_marginal_relevance=True, k=10):
        retriever = self.repo_ds.as_retriever()
        retriever.search_kwargs['distance_metric'] = distance_metric
        retriever.search_kwargs['fetch_k'] = fetch_k
        retriever.search_kwargs['maximal_marginal_relevance'] = maximal_marginal_relevance
        retriever.search_kwargs['k'] = k
        qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model=model_name), retriever=retriever)
        return qa
