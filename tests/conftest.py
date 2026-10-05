import pytest


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true",
                     help="also run the tests marked slow, which run at length")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs at length, and only with --slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return

    skip = pytest.mark.skip(reason="runs at length; pass --slow to run it")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
