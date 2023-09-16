# pyhc_bots.py
from .helper_bot import HelperBot


class HapiBot(HelperBot):
    REPO_NAME = "hapiclient"
    REPO_URL = "https://github.com/hapi-server/client-python.git"
    SUFFIXES = [".py", ".md"]
    def __init__(self, use_local_vector_store=True):
        super().__init__(HapiBot.REPO_NAME, HapiBot.REPO_URL, HapiBot.SUFFIXES, use_local_vector_store)


class KamodoBot(HelperBot):
    REPO_NAME = "kamodo"
    REPO_URL = "https://github.com/nasa/Kamodo.git"
    SUFFIXES = [".py", ".md"]
    def __init__(self, use_local_vector_store=True):
        super().__init__(KamodoBot.REPO_NAME, KamodoBot.REPO_URL, KamodoBot.SUFFIXES, use_local_vector_store)


class PlasmapyBot(HelperBot):
    REPO_NAME = "plasmapy"
    REPO_URL = "https://github.com/PlasmaPy/PlasmaPy.git"
    SUFFIXES = [".py", ".md"]
    def __init__(self, use_local_vector_store=True):
        super().__init__(PlasmapyBot.REPO_NAME, PlasmapyBot.REPO_URL, PlasmapyBot.SUFFIXES, use_local_vector_store)


class PysatBot(HelperBot):
    REPO_NAME = "pysat"
    REPO_URL = "https://github.com/pysat/pysat.git"
    SUFFIXES = [".py", ".rst"]  # TODO: include README.md but not other .md files, somehow?
    def __init__(self, use_local_vector_store=True):
        super().__init__(PysatBot.REPO_NAME, PysatBot.REPO_URL, PysatBot.SUFFIXES, use_local_vector_store)


class PyspedasBot(HelperBot):
    REPO_NAME = "pyspedas"
    REPO_URL = "https://github.com/spedas/pyspedas.git"
    SUFFIXES = [".py", ".md"]
    def __init__(self, use_local_vector_store=True):
        super().__init__(PyspedasBot.REPO_NAME, PyspedasBot.REPO_URL, PyspedasBot.SUFFIXES, use_local_vector_store)


class SpacepyBot(HelperBot):
    REPO_NAME = "spacepy"
    REPO_URL = "https://github.com/spacepy/spacepy.git"
    SUFFIXES = [".py", ".md"]
    def __init__(self, use_local_vector_store=True):
        super().__init__(SpacepyBot.REPO_NAME, SpacepyBot.REPO_URL, SpacepyBot.SUFFIXES, use_local_vector_store)


class SunpyBot(HelperBot):
    REPO_NAME = "sunpy"
    REPO_URL = "https://github.com/sunpy/sunpy.git"
    SUFFIXES = [".py", ".rst"]
    def __init__(self, use_local_vector_store=True):
        super().__init__(SunpyBot.REPO_NAME, SunpyBot.REPO_URL, SunpyBot.SUFFIXES, use_local_vector_store)


class PyhcBot(HelperBot):
    REPO_NAME = "pyhc"
    REPO_URL = "https://github.com/heliophysicsPy/heliophysicsPy.github.io.git"
    SUFFIXES = [".md", ".yml"]
    def __init__(self, use_local_vector_store=True):
        super().__init__(PyhcBot.REPO_NAME, PyhcBot.REPO_URL, PyhcBot.SUFFIXES, use_local_vector_store)
