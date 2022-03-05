"""
This is the main location where you will run Portfolio Monte Carlo simulations.
"""

import portfolio
import utils


def test_no_conf():

    port = portfolio.PortfolioSimulator(
        {
            "starting_balance": 10000,
            "duration": 5,
            "base_income": 2000,
            "base_expenses": 1500,
            "retirement_year": 3,
        }
    )

    port.simulate_once()


def test_sim_once():
    settings = utils.load_settings("confs/test_base.json")

    port = portfolio.PortfolioSimulator(settings)

    balance = port.simulate_once()

    # Going to switch to the logging module when I have time. TODO
    print(f"test_sim_once final balance: {balance}")


def test_run_simulation():
    settings = utils.load_settings("confs/test_base.json")

    port = portfolio.PortfolioSimulator(settings)

    port.run_simulation()

    # Going to switch to the logging module when I have time. TODO
    print(f"test_run_simulation final mean balance: {port.results.mean()}")


def main():
    settings = utils.load_settings("confs/real_run.json")

    port = portfolio.PortfolioSimulator(settings)

    port.run_simulation()

    # Going to switch to the logging module when I have time. TODO
    print(f"test_run_simulation final mean balance: {port.results.mean()}")


if __name__ == "__main__":
    main()
