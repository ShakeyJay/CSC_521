"""
This is the main location where you will run Portfolio Monte Carlo simulations.
"""

import portfolio
import utils


def test_no_conf():

    port = portfolio.PortfolioSimulator(
        {"starting_balance": 1000, "withdraw_balance": 10, "duration": 5}
    )

    port.simulate_once()


def test_sim_once():
    settings = utils.load_settings("confs/test_base.json")

    port = portfolio.PortfolioSimulator(settings)

    balance = port.simulate_once()

    # Going to switch to the logging module when I have time. TODO
    print(f"test_sim_once final balance: {balance}")


def main():
    raise NotImplementedError


if __name__ == "__main__":
    test_sim_once()
