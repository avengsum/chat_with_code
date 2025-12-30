from langchain_community.document_loaders import GitLoader

loader = GitLoader(
  clone_url="https://github.com/psf/requests.git",
  repo_path="./temp_repo",
    branch="main",
    file_filter=lambda file_path: file_path.endswith((".py", ".js", ".ts","yaml"))
)

docs = loader.load()

print(f"loaded {len(docs)} files")
