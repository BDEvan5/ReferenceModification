from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Testing the Reference Modification Planner'
LONG_DESCRIPTION = 'Code for accompanying paper on ref mod'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="ReferenceModification", 
        version=VERSION,
        author="Benjamin Evans",
        author_email="<youremail@email.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'autonomous racing'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: Linux",
        ]
)
