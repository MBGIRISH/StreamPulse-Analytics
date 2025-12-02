"""
Path utility functions for OTT Analytics project
Ensures all scripts use correct paths regardless of where they're run from
"""

import os

def get_project_paths():
    """Get project root and common paths"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return {
        'root': project_root,
        'data': os.path.join(project_root, 'data'),
        'outputs': os.path.join(project_root, 'outputs'),
        'charts': os.path.join(project_root, 'outputs', 'charts')
    }

def ensure_directories():
    """Create all necessary directories if they don't exist"""
    paths = get_project_paths()
    os.makedirs(paths['data'], exist_ok=True)
    os.makedirs(paths['outputs'], exist_ok=True)
    os.makedirs(paths['charts'], exist_ok=True)
    return paths

