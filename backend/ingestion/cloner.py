import os
import shutil
import tempfile
from git import Repo, GitCommandError
from urllib.parse import urlparse

class RepoCloner:
    def __init__(self, base_dir: str = None):
        # Use a temp directory if no base dir specified
        self.base_dir = base_dir or tempfile.mkdtemp()

    def clone(self, github_url: str) -> str:
        """
        Clone a GitHub repo and return the local path.
        Handles cleanup of previously cloned same repo.
        """
        repo_name = self._extract_repo_name(github_url)
        local_path = os.path.join(self.base_dir, repo_name)

        # Clean up if already exists
        if os.path.exists(local_path):
            shutil.rmtree(local_path)

        print(f"Cloning {github_url} into {local_path}...")

        try:
            Repo.clone_from(github_url, local_path, depth=1)
            print(f"Successfully cloned: {repo_name}")
            return local_path

        except GitCommandError as e:
            raise ValueError(f"Failed to clone repo: {str(e)}")

    def cleanup(self, local_path: str):
        """Remove cloned repo from disk after indexing."""
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
            print(f"Cleaned up: {local_path}")

    def _extract_repo_name(self, github_url: str) -> str:
        """Extract owner/repo-name from GitHub URL."""
        parsed = urlparse(github_url)
        path = parsed.path.strip("/").replace("/", "_")
        return path.rstrip(".git")