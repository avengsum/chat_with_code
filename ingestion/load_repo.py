from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language

def get_repo():
    loader = GitLoader(
        clone_url="https://github.com/psf/requests.git",
        repo_path="./temp_repo",
        branch="main",
        file_filter=lambda file_path: file_path.endswith((".py", ".js", ".ts", "yaml"))
    )

    docs = loader.load()
    return docs
