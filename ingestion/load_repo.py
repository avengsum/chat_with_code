from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language
import os

Allowed_Extensions = {'.py', '.js', '.ts', '.html', '.css', '.md', '.txt',      '.json', '.yaml', '.yml', '.toml','.dockerfile', '.sh','.go','.cpp'}

def file_filter(file_path):
    # so it will remove junk file
     ignored_folders = ['node_modules', '.git', 'venv', '__pycache__', 'dist', 'build']

     for folder in ignored_folders:
        if folder in file_path:
            return False
    
    
     extension = os.path.splitext(file_path)[1]

     return extension.lower() in Allowed_Extensions



def get_repo():
    loader = GitLoader(
        clone_url="https://github.com/psf/requests.git",
        repo_path="./temp_repo",
        branch="main",
        file_filter=file_filter
    )

