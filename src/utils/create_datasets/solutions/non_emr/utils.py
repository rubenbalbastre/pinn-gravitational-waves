from os import path, makedirs


def create_directory_if_does_not_exist(directory: str):
    """
    Create directory if it does not exist
    """
    if not path.exists(directory):
        makedirs(directory)
