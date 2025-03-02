import os

def get_specific_files(directory):
    """
    Returns a list of files with specific extensions in the given directory.
    """
    extensions = ['.png', '.png1']
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, filename))
    return files
