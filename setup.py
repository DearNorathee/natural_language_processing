from setuptools import setup, find_packages
# when install in other pc(Cortex) still get weird error, which seems to be hard to fix
# possibly "spacy==3.7.2",
        # "pydantic==1.10.12", might not work with python 3.12(NOT SURE)
setup(
    author= "Dear Norathee",
    description="<add short describtion here>",
    name="natural_language_processing",
    version="0.1.1rc2",
    packages=find_packages(),
    license="MIT",
    install_requires=[
        "xlwings",
        "os_toolkit>=0.1.1",
        "dataframe_short>=0.1.6",
        "python_wizard>=0.1.2",
        "inspect_py>=0.1.1",
        "py_string_tool>=0.1.3",
        "spacy==3.7.2",
        "pydantic==1.10.12",
        "scikit-learn",

        "numpy<2.0.0",
        "mlconjug3",
        "langdetect",
        "nltk",
        "langcodes",
        "pyttsx3"
        ],
    python_requires='>=3.10.0',
    extras_require={
        'full': ['torch>=1.0']  # Optional torch dependency
    },
    # example
    # install_requires=['pandas>=1.0',
    # 'scipy==1.1',
    # 'matplotlib>=2.2.1,<3'],
    

)