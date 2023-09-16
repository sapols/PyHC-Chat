# default_bot.py
from config import model_name
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage


def condense_answers(prompt, package_statements):
    chat = ChatOpenAI(model_name=model_name)
    chat_list = [HumanMessage(content=f"""
I need you to condense multiple statements related to a question I asked into one cogent answer. The question I asked was:

"{prompt}"

Use the following statements as context to formulate a final answer to my question (without directly referencing the fact that your final answer was pieced together from the statements):

--
{expand_package_statements(package_statements)}
--
""")]
    return chat(chat_list).content


def expand_package_statements(package_statements):
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


def let_pyhc_chat_answer(chat_history, prompt):
    chat = ChatOpenAI(model_name=model_name)
    chat_list = [SystemMessage(content=f"""
You are PyHC-Chat, an AI custom-designed by the Python in Heliophysics Community to discuss the Python in Heliophysics Community (PyHC) and its seven core packages.

Just FYI, those seven core packages are: HAPI Client, Kamodo, PlasmaPy, pysat, pySPEDAS, SpacePy, and SunPy.

And in case anyone asks you to name every single PyHC package, the other non-core Python packages that fall under PyHC's umbrella are: {', '.join(get_other_pyhc_packages())}. That's probably good trivia for you to know.
""")] + chat_history
    chat_list.append(HumanMessage(content=prompt))
    return chat(chat_list).content


class DefaultBot:
    def __init__(self, chat_history):
        self.chat = ChatOpenAI(model_name=model_name)
        self.chat_list = chat_history

    def get_completion(self):
        return self.chat(self.chat_list).content
