"""
This is the main location where you will run Portfolio Monte Carlo simulations.
"""

import matplotlib.pyplot as plt
import numpy as np

import portfolio
import sim_ann
import utils

np.random.seed(5)


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
    settings = utils.load_settings("confs/median_income.json")
    # settings = utils.load_settings("confs/college_grad.json")

    options = utils.load_settings("confs/portfolios.json")

    for key, val in options.items():
        settings["portfolio"] = val

        port = portfolio.PortfolioSimulator(settings)

        port.run_simulation(sim_count=3)

        plt.plot(np.max(port.sim_data, axis=0), label=f"Max {key}")
        plt.plot(np.mean(port.sim_data, axis=0), label=f"Mean {key}")
        plt.plot(np.min(port.sim_data, axis=0), label=f"Min: {key}")

    plt.legend()
    plt.savefig("median_sim_ann.png", format="png")
    plt.show()

    # print(np.mean(port.sim_data, axis=0))


def run_sim_annealing():
    settings = utils.load_settings("confs/median_income.json")
    options = utils.load_settings("confs/portfolios.json")
    key = "safe"
    settings["portfolio"] = options[key]

    # port = portfolio.PortfolioSimulator(settings)
    # port.run_simulation(sim_count=3)

    anneal = sim_ann.SimAnneal(
        optimal_savings,
        savings_raise,
        settings,
        startTemp=0.45,
        endTemp=0.1,
        coolFactor=0.01,
    )

    final = anneal.optimize()

    print(final)


def optimal_savings(settings):
    port = portfolio.PortfolioSimulator(settings)
    port.run_simulation(sim_count=10)

    return port.results.mean()


def savings_raise(settings, temp):
    settings["saving_rate"] = temp
    print(f"""Savigns RATE: {settings["saving_rate"]}""")
    return settings


if __name__ == "__main__":
    main()
