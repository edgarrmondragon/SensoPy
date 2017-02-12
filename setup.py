""""""

from distutils.core import setup
import sensopy

setup(
    name = 'sensopy',
    packages = ['sensopy', 'sensopy.discrim'],
    version = sensopy.__version__,
    description = 'Python library for the design and analysis of sensory discrimination tests.',
    author = 'Edgar Ramírez',
    author_email = 'typingmonkey9201@gmail.com',
    url = 'https://github.com/EdgarRMmex/SensoPy',
    download_url = 'https://github.com/EdgarRMmex/SensoPy/tarball/0.0.1',
    keywords = ['sensory', 'discrimination', 'triangle'],
    classifiers = [],
)
