# pyhc_chat_bot.py
from config import model_name
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage


def pyhc_chat_system_message():  # TODO: Programmatically get names of (and number of) core PyHC packages?
    return SystemMessage(content=f"""
You are PyHC-Chat, an AI custom-designed by the Python in Heliophysics Community (PyHC) to discuss PyHC and its seven core packages.

Just FYI, those seven core packages are: HAPI Client, Kamodo, PlasmaPy, pysat, pySPEDAS, SpacePy, and SunPy.

You are powered by OpenAI's {model_name} model, which inherently knows about PyHC and the core packages. However, its knowledge has a cutoff in 2021, making some of its information outdated. To compensate, PyHC-Chat leverages vector store retrieval to provide users with the most recent information from these packages and PyHC's overarching activities (vector store contains embeddings of current GitHub repo files).

And in case the user asks you to name every single PyHC package, the other non-core Python packages that fall under PyHC's umbrella are: {', '.join(get_other_pyhc_packages())}. That's probably good trivia for you to know.
""")


def let_pyhc_chat_answer(chat_history, prompt):
    # The main function to get PyHC-Chat's default response to a user's prompt (without context from the vector store)
    chat = ChatOpenAI(model_name=model_name)
    chat_list = [pyhc_chat_system_message()] + chat_history + [HumanMessage(content=prompt)]
    return chat(chat_list).content


def answer_with_context(chat_history, prompt, repo_statements):
    # The main function to incorporate context from the vector store into PyHC-Chat's response to a user's prompt
    chat = ChatOpenAI(model_name=model_name)
    chat_list = [pyhc_chat_system_message()] + chat_history + [HumanMessage(content=f"""
To best address the user's inquiry, use the information provided below which was retrieved from the vector store:

User's inquiry:
"{prompt}"

Relevant details from the vector store about the associated repo(s):
--
{expand_statements(repo_statements)}
--

If any statements indicate a lack of specific information, integrate that understanding into your final response without directly quoting those statements. Strive to offer an informative and seamless answer, even if some parts of the inquiry couldn't be fully addressed based on the available context.
""")]
    return chat(chat_list).content


def expand_statements(package_statements):
    expanded = ""
    for i, (package, statement) in enumerate(package_statements.items(), start=1):
        expanded += f"{i}. (from {package}): `{statement}`\n\n"
    return expanded


def get_other_pyhc_packages():
    return [
        'AFINO', 'CCSDSPy', 'dbprocessing', 'enlilviz', 'GeospaceLAB', 'OMMBV', 'pyDARN', 'sami2py', 'SkyWinder',
        'SkyWinder-Analysis', 'solarmach', 'solo-epd-loader', 'space-packet-parser', 'Speasy', 'fiasco', 'OCBpy',
        'AACGMV2', 'apexpy', 'SpiceyPy', 'NDCube', 'viresclient', 'aiapy', 'aidapy', 'geopack', 'MCALF', 'hissw',
        'sunraster', 'sunkit-image', 'sunkit-instruments', 'pyflct', 'irispy-lmsal', 'XRTpy', 'regularizePSF',
        'TomograPy', 'python-magnetosphere', 'pysatCDF', 'pyglow', 'geodata', 'fisspy', 'CDFlib', 'PyTplot',
        'lofarSun', 'PyGS', 'ACEmag', 'AstrometryAzEl', 'Auroral Electrojet', 'DASCutils',
        'Digital Meridian Spectrometer', 'GEOrinex', 'GOESutils', 'GIMAmag', 'GLOW', 'HWM-93', 'IGRF-13', 'IRI-2016',
        'IRI-90', 'LOWTRAN', 'Maidenhead', 'MGSutils', 'POLAN', 'PyGemini', 'PyMap3D', 'PyZenodo', 'ReesAurora',
        'Scanning Doppler Interferometer', 'ScienceDates', 'THEMISasi', 'WMM2020', 'WMM2015', 'MSISE-00', 'MadrigalWeb',
        'NEXRADutils'
    ]


class PyHCChatBot:
    def __init__(self, chat_history):
        self.chat = ChatOpenAI(model_name=model_name)
        self.chat_list = [pyhc_chat_system_message()] + chat_history

    def get_completion(self):
        return self.chat(self.chat_list).content
