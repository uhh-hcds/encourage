"""Pytest configuration file."""

import nltk


def pytest_configure():
    """Allows plugins and conftest files to perform initial configuration.

    This hook is called for every plugin and initial conftest
    file after command line options have been parsed.
    """
    nltk.download("punkt_tab")
