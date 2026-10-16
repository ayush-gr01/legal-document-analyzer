import sys, os
# Add the project root (parent directory) to sys.path so that tests can import the src package.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
