import setuptools # type: ignore
from typing import List

# Read the README file to use as the long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Define project variables
__version__ = "0.0.1"
REPO_NAME = "KidneyTumorCNNclassifier"
AUTHOR_USER_NAME = "kjamunap"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "kiranprasad00713@gmail.com"

# Helper function to read the requirements.txt file
def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Clean up the hidden 'new line' characters
        requirements = [req.replace("\n", "") for req in requirements]
            
    return requirements

# The main configuration
setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for a CNN app",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=get_requirements("requirements.txt")
)