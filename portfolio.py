"""
This module is the Portfolio simulator. It is derived from the mc_engine class.
"""

import numpy.random as rand

import mc_engine


class PortfolioSimulator(mc_engine.MonteCarloEngine):
    """
    This class is an extension of the MonteCarlo engine where we run individual
    simulations of an individuals portfolio.
    """

    def __init__(self, settings):
        super().__init__()
        self.starting_balance = settings.get("starting_balance")
        self.duration = settings.get("duration")
        self.income = settings.get("base_income")
        self.base_expenses = settings.get("base_expenses")
        self.retirement_year = settings.get("retirement_year")

        # Just for testing! We will put in the stock simulation portion to work
        # with this.
        self.mean_return = 0.06
        self.std_return = 0.08
        self.inf_rate = 0.03
        self.inf_std = 0.01

        # Raises? Maybe we can set starting age and retirement age and move this along
        # the actual possibilities. Much more likely to make more in your 30s than 20s
        # etc.
        self.mean_raise = 0.03
        self.std_raise = 0.01

    def simulate_once(self) -> float:
        """
        Runs a single portfolio simulation using instance settings.

        Currently assumes the difference between income and expenses is
        put into your portfolio.

        Returns:
            float: This is the final balance of one simulation
        """

        current_balance = self.starting_balance
        for year in range(self.duration):
            current_balance = self.calculate_balance_change(current_balance, year)

            # This is from the original but because I think we have have non retirement
            # years we probably do not want this.
            # if current_balance <= 0:
            #     return 0

        return current_balance

    def calculate_balance_change(self, c_balance: float, year: int) -> float:
        # At first we assume expenses move by inf but income does not fluctuate.
        # Assume no income in retirement.
        inf = self.get_inflation()
        ret = self.get_market_returns()
        delta = self.base_expenses * (1 + inf)

        self.income = self.get_current_income()

        if year < self.retirement_year:

            # We also assume if you have more expenses than income you must pull $'s out
            # of your portfolio to pay bills.

            # This keeps the - sign the same in the return for all possibilities.
            delta = delta - self.income

        return (c_balance * (1 + ret)) - delta

    def get_inflation(self):
        return rand.normal(self.inf_rate, self.inf_std)

    def get_market_returns(self):
        # This is where we will call the simulation data from stock_exploration

        # For now just testing with base values
        return rand.normal(self.mean_return, self.std_return)

    def get_current_income(self) -> float:
        y_raise = rand.normal(self.mean_raise, self.std_raise)

        # Assumes you would not accept less money for the same job.
        if y_raise <= 0:
            return self.income

        return self.income * (1 + y_raise)
