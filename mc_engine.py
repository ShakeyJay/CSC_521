"""
This is the main MonteCarlo class location. All of our classes will be
derived from this one.

Author: Josh Hall
"""

import numpy as np
import abc


class MonteCarloEngine:
    """
    All of the Monte Carlo simulations are pretty much setup the same
    way at the top level and that is the goal of this class.
    """

    def __init__(self):
        self.results = np.array([])
        self.convergence = False

    @abc.abstractmethod
    def simulate_once(self):
        raise NotImplementedError

    def run_simulation(
        self, err_threshold: float = 0.001, sim_count: int = 10
    ) -> tuple:
        for i in range(sim_count):
            x = self.simulate_once()
            self.results = np.append(self.results, x)

            if i > 1:
                mu = self.results.mean()
                var = (np.square(self.results.sum()) / (i - 1)) - (mu ** 2)
                dmu = np.sqrt(var / i)

                # print("i = " + str(i) + ", dmu = " + str(dmu))
                if dmu < abs(mu) * err_threshold:
                    self.convergence = True

    def value_at_risk(self, risk: float = 0.05):
        if self.results.size > 0:
            var_arr = np.sort(self.results)
            idx = int(var_arr.size * risk)
            return var_arr[idx]
        raise SimulationNotRunException()


class SimulationNotRunException(Exception):
    def __init__(self):
        super().__init__("You have not run the simulation yet!")


class SimulationNotCreated(Exception):
    def __init__(self):
        super().__init__("You have setup the simulation yet.")


class MonteCarloTest(MonteCarloEngine):
    def simulate_once(self):
        return np.random.randint(1, 10, 10)


if __name__ == "__main__":
    sim = MonteCarloTest()

    print(sim.run_simulation())

    print(sim.value_at_risk())
