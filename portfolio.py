"""
This module is the Portfolio simulator. It is derived from the mc_engine class.


Improvement Ideas:
- Adding multiple income streams and durations
- Savings Rate so that you can manage your expenses.
- List of Expenses and if they will always exist
    - Morgage stuff like that.
- Instead of just a retirement balance
    - Have a breakdown of account types.
    - Real Estate
    - 401K
    - Investment Accounts
    - IRAs
- Control Income Raises and Peak Income in a better way. 
"""

import math
import numpy as np

import mc_engine
import stock_exploration


rand = np.random


class PortfolioSimulator(mc_engine.MonteCarloEngine):
    """
    This class is an extension of the MonteCarlo engine where we run individual
    simulations of an individuals portfolio.
    """

    def __init__(self, settings):
        super().__init__()
        self.settings = settings

        # Just for testing! We will put in the stock simulation portion to work
        # with this.
        self.mean_return = 0.06
        self.std_return = 0.08

        # Assumes that we have equal exposure in each of the symbols in our bag.
        # self.mean_return, self.std_return = self.create_market_data()

        # Maybe we could expand here to use real predictions for what these might be.
        self.inf_rate = 0.03
        self.inf_std = 0.01

        # Raises? Maybe we can set starting age and retirement age and move this along
        # the actual possibilities. Much more likely to make more in your 30s than 20s
        # etc.
        self.mean_raise = 0.03
        self.std_raise = 0.01

        self.sim_data = []

    def simulation_setup(self):
        self.starting_balance = self.settings.get("starting_balance")
        self.duration = self.settings.get("duration")
        self.income = self.settings.get("base_income")
        self.base_expenses = self.settings.get("base_expenses")
        self.retirement_year = self.settings.get("retirement_year")
        self.frugal_year = self.settings.get("frugal_years")
        self.frugal_saving_rate = self.settings.get("frugal_saving_rate")
        self.saving_rate = self.settings.get("saving_rate")
        self.portfolio = self.settings.get("portfolio")
        self.mean_return, self.std_return = self.create_market_data()

    def display(self, year, current_balance, expenses, delta, ret, income):
        print(
            f"""
                Year: {year} 
                Income: {income}
                Expenses: {expenses}
                Retirement Balance: {current_balance}
                Return %: {ret}
                Return $: {current_balance * ret}
            """
        )

    def simulate_once(self) -> float:
        # This is to reset the values for each run. While still using class setup.
        self.simulation_setup()
        ret_bal = np.array([])
        current_balance = self.starting_balance
        for year in range(self.duration):

            current_balance = self.calculate_balance_change(current_balance, year)
            if current_balance <= 0 and year >= self.retirement_year:
                current_balance = 0

            ret_bal = np.append(ret_bal, current_balance)

        self.sim_data.append(ret_bal)
        return current_balance

    def calculate_balance_change(self, c_balance: float, year: int) -> float:
        inf = self.get_inflation()
        ret = self.get_market_returns()

        if year <= self.frugal_year:
            delta = max(
                self.base_expenses * (1 + inf),
                self.income - (self.frugal_saving_rate * self.income),
            )
        else:
            delta = self.income - (self.income * self.saving_rate)

        expenses = delta
        self.income = self.get_current_income(bool(year < self.retirement_year))

        if year < self.retirement_year:

            # We also assume if you have more expenses than income you must pull $'s out
            # of your portfolio to pay bills.

            # This keeps the - sign the same in the return for all possibilities.
            delta = delta - self.income

            self.display(year, c_balance, expenses, delta, ret, self.income)
        else:
            self.display(year, c_balance, expenses, delta, ret, 0)

        return (c_balance * (1 + ret)) - delta

    def get_inflation(self):
        return rand.normal(self.inf_rate, self.inf_std)

    def get_market_returns(self):
        # This is where we will call the simulation data from stock_exploration

        # For now just testing with base values
        return rand.normal(self.mean_return, self.std_return)

    def get_current_income(self, working: bool) -> float:
        y_raise = rand.normal(self.mean_raise, self.std_raise)

        # Assumes you would not accept less money for the same job.
        if y_raise <= 0 or not working:
            return self.income

        return self.income * (1 + y_raise)

    def create_market_data(self) -> tuple:
        # This is what we would do if we wanted to do a new stock exploration each time?
        # Probably want to do this each run of monte carlo.
        stock_data = stock_exploration.testMultipleStock(
            self.portfolio,
            "2016-01-01",
            "2021-03-01",
            10,
            False,
        )

        mean_return = np.array([])
        std_return = np.array([])
        for stock in stock_data:
            mean_return = np.append(mean_return, stock[1])
            std_return = np.append(std_return, stock[2])

        print(self.portfolio)
        print(stock_data)
        print(mean_return.mean(), std_return.mean())
        print(math.exp(mean_return.mean()), math.exp(std_return.mean()))

        # placeholder for what the actual values will be just to get it working.
        return (math.exp(mean_return.mean())) - 1, math.exp(std_return.mean()) - 1
