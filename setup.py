# External Includes
import os
from setuptools import find_packages, setup

# Internal Includes
from rfml import __version__ as VERSION


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def load_requirements():
    return read("requirements.txt").splitlines()


setup(
    name="rfml",
    version=VERSION,
    author="Bryse Flowers",
    author_email="brysef@vt.edu",
    description="Radio Frequency Machine Learning (RFML) in PyTorch",
    license="Modified BSD",
    keywords="RF Machine Learning ML RFML Datasets AMC Modulation Classification",
    url="https://github.com/brysef/rfml",
    packages=find_packages(exclude=["test*"]),
    long_description=(
        "Radio Frequency Machine Learning (RFML) in PyTorch:\n"
        "The concept of deep learning has revitalized machine learning research in "
        "recent years.  In particular, researchers have demonstrated the use of deep "
        "learning for a multitude of tasks in wireless communications, such as signal "
        "classification, waveform creation, and cognitive radio.  These technologies "
        "have been colloquially coined Radio Frequency Machine Learning (RFML) by the "
        "Defense Advanced Research Projects Agency (DARPA).  This library contains "
        "PyTorch implementations of common RFML applications and neural architectures."
    ),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Telecommunications Industry",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Communications :: Ham Radio",
        "Topic :: Communications :: Telephony",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=load_requirements(),
    extras_require={"tutorial": ["jupyter", "matplotlib", "seaborn"]},
)
