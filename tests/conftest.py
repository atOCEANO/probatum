import pytest


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true",
                     help="also run the tests marked slow, which run at length")
    parser.addoption("--coverage", action="store_true",
                     help="also run the coverage cells, which simulate every interval "
                          "at length")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs at length, and only with --slow")
    config.addinivalue_line("markers", "gpu: needs a CUDA device, which the gpu stage "
                                       "hands in")
    config.addinivalue_line("markers", "coverage: a coverage cell, run only with "
                                       "--coverage")


def pytest_collection_modifyitems(config, items):
    skips = {"slow": pytest.mark.skip(reason="runs at length; pass --slow to run it"),
             "coverage": pytest.mark.skip(reason="a coverage cell; pass --coverage to "
                                                 "run it")}

    for name, skip in skips.items():
        if config.getoption(f"--{name}"):
            continue

        for item in items:
            if name in item.keywords:
                item.add_marker(skip)
