"""Shared test configuration.

`addopts` deselects `api` tests by default, but a command-line `-m` replaces it, so
api tests also skip here when no key is configured.
"""

import pytest

from ae import config as ae_config


def pytest_collection_modifyitems(config, items):
    if ae_config.api_key():
        return
    skip = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set")
    for item in items:
        if "api" in item.keywords:
            item.add_marker(skip)
