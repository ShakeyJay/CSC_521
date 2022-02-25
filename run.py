"""
This is the main location where you will run Portfolio Monte Carlo simulations.
"""

import json

import portfolio


def get_conf(conf_file):
    with open(conf_file, "r", encoding="utf8") as conf:
        return json.load(conf)


def test_no_conf():

    port = portfolio.PortfolioSimulator(
        {"starting_balance": 1000, "withdraw_balance": 10, "duration": 5}
    )

    port.simulate_once()


def test_sim_once():
    settings = get_conf("confs/test_base.json")

    port = portfolio.PortfolioSimulator(settings)

    port.simulate_once()


def main():
    return NotImplementedError


if __name__ == "__main__":
    test_sim_once()
