"""
This module is for all of the random utility functions that are used in the project.
"""

import json


def load_settings(conf_file):
    with open(conf_file, "r", encoding="utf8") as conf:
        return json.load(conf)
