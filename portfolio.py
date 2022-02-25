"""
This module is the Portfolio simulator. It is derived from the mc_engine class.
"""


import mc_engine


class PortfolioSimulator(mc_engine.MonteCarloEngine):
    """
    This class is an extension of the MonteCarlo engine where we run individual
    simulations of an individuals portfolio.
    """

    def __init__(self, settings):
        super().__init__()
        self.starting_balance = settings.get("starting_balance")
        self.withdraw_amt = settings.get("withdraw_balance")
        self.duration = settings.get("duration")

        # Just for testing! We will put in the stock simulation portion to work
        # with this.
        self.mean_return = 0.06
        self.std_return = 0.08
        self.inf_rate = 0.03
        self.inf_std = 0.01


    def simulate_once(self):
        raise mc_engine.SimulationNotCreated

    