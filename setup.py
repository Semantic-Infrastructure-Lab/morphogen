"""Compatibility shim for legacy setuptools workflows.

Project metadata lives in ``pyproject.toml``. This file remains only so
commands that still expect ``setup.py`` continue to work.
"""

from setuptools import setup


setup()
