from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Testing the Reference Modification Planner'
LONG_DESCRIPTION = 'Code for accompanying paper on ref mod'

# Setting up
setup(
    name="ReferenceModification",
    version=VERSION,
    author="Benjamin Evans",
    author_email="19811799@sun.ac.za",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['torch==1.8.1', 'numpy==1.19.5', 'matplotlib==3.4.1', 'casadi==3.5.5',
                      'numba==0.53.1', 'scipy==1.6.3'],
    keywords=['obstacle avoidance', 'autonomous racing', 'reinforcement learning'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ]
)
