# src/ui/__init__.py

"""
User Interface Module
Web dashboard templates and static assets
"""

import os
from pathlib import Path

# UI paths
TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

# Ensure directories exist
TEMPLATE_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
(STATIC_DIR / "css").mkdir(exist_ok=True)
(STATIC_DIR / "js").mkdir(exist_ok=True)
(STATIC_DIR / "images").mkdir(exist_ok=True)

__all__ = ['TEMPLATE_DIR', 'STATIC_DIR']