def md2term(text):
    '''Markdown to terminal (bold)'''
    import re
    # COLOR_RED = '\033[91m'
    # COLOR_GREEN = '\033[92m'
    # COLOR_YELLOW = '\033[93m'
    # COLOR_BLUE = '\033[94m'
    # COLOR_MAGENTA = '\033[95m'
    # COLOR_CYAN = '\033[96m'
    # COLOR_RESET = '\033[0m'  # Reset text formatting
    # BOLD_BEGIN = '\033[1m'
    # BOLD_END = '\033[0;0m'

    ans = re.sub(r'\*\*(.*?)\*\*', r'\033[1m\1\033[0m', text)
    return ans

def bold(text):
    return r'\033[1m' + text + r'\033[0m'

def safe_mkdir(folder):
    """Create folder if it does not exist"""
    import os
    if not os.path.exists(folder): os.makedirs(folder)
    return folder

def is_older_than(file1, file2):
    """Returns True if file1 does not exist or is older than file2."""
    import os.path as path
    return not path.isfile(file1) or path.getmtime(file1) < path.getmtime(file2)

def is_newer_than(file1, file2):
    """Returns True if file1 exists and is newer than file2."""
    import os.path as path
    return path.isfile(file1) and path.getmtime(file1) > path.getmtime(file2)

def write_to_json(x, json_file):
    """Write to json file"""
    import json
    with open(json_file, 'w') as output_file:
        json.dump(x, output_file, indent=2, ensure_ascii=False)

def read_from_json(json_file):
    """Read from json file"""
    import json
    with open(json_file) as in_file:
        a = json.load(in_file)
    return a