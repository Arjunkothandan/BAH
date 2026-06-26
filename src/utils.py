import os


def get_project_root() -> str:
    """Return the absolute path of the project root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
