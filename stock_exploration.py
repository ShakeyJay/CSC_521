import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy.random as rand
import scipy.linalg as la
from math import sqrt, exp


def plotStocks(is_string, stock, df, alpha):
    """
    Function that is called from fetchStocks() when 'show' is set to
    True. This function produces the following plots for each stock:

    - Log adjusted close price
    - Log returns
    - QQ plot of log returns
    - Histogram of log returns

    As well as runs a test for normality.

    Args:
    is_string (bool): whether or not input into fetchStocks was a string
    stock (str): ticker symbol for stock
    df (pandas dataframe): dataframe containing relevant columns for stock
    alpha (float): significance level for normality test
    """

    if is_string:
        log_close = "log_adj_close"
        log_returns = "log_returns"

    else:
        log_close = "log_adj_close_{}".format(stock)
        log_returns = "log_returns_{}".format(stock)

    # Log price plot
    df[log_close].plot(title="{} Log Adjusted Close Price".format(stock))
    plt.show()

    # Log returns plot
    df[log_returns].plot(title="{} Log Returns".format(stock))
    plt.show()
    # QQ Plot
    stats.probplot(df[log_returns], plot=plt)
    plt.show()

    # Hist of log returns
    df[log_returns].hist(bins=50, density=True)
    xmin, xmax = plt.xlim()
    mu, std = stats.norm.fit(df[log_returns][1:])
    x = np.linspace(xmin, xmax, 50)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, color="black", linewidth=2)
    plt.show()

    # Tests the null hypothesis that a sample comes from a normal distribution
    result = stats.normaltest(np.log(df[log_close]))

    # Not from a normal distribution
    if result.pvalue < alpha:
        print(
            """The normal test p value was {}, meaning we reject the null hypothesis
                 that this data comes from a normal distribution.""".format(
                round(result.pvalue, 4)
            )
        )

    # From a normal distribution
    else:
        print(
            """The normal test p value was {}, meaning we fail to reject the null hypothesis
                 meaning this data comes from a normal distribution.""".format(
                round(result.pvalue, 4)
            )
        )


def fetchStocks(stocks, start, end, test_pct, alpha, show):
    """
    Function that retrieves historical data for 'stocks' from the time
    period identified with 'start' and 'end' dates. It then derives
    features of interest based on the 'Adj Close' column. It then
    creates plots of interest if the value for 'show' = True as well
    as runs a test of normality. It then drops all other unnecessary
    columns, and splits the data into a train/test set by using
    'test_pct' to identify how large the test set should be.

    Args:
    stocks (str/list): ticker symbol/s for stock/s
    start (str): start date in YYYY-MM-DD format
    end (str): end date in YYYY-MM-DD format
    test_pct (float): percent of overall data used for testing
    alpha (float): significance level for normality test
    show (bool): whether to show plots

    Returns:
    2 dataframes (train and test) either consisting of a single stock
    or multiple stock information.
    """

    if type(stocks) == str:

        # Fetch raw data for single stock
        df = yf.download(stocks, start=start, end=end, progress=False)

        # Derive features off of adj close
        df["log_adj_close"] = np.log(df["Adj Close"])
        df["log_returns"] = df["log_adj_close"] - df["log_adj_close"].shift(1)

        # Rename for consistency across sets
        df.rename(columns={"Adj Close": "adj_close"}, inplace=True)

        if show:

            plotStocks(True, stocks, df, alpha)

        # Subset to what we care about
        df = df[["adj_close", "log_adj_close", "log_returns"]]

        # Split into train/test
        train_set, test_set = np.split(df, [int((1 - test_pct) * len(df))])

        return train_set, test_set

    if type(stocks) == list:

        # Fetch raw data for multiple stocks
        df = yf.download(stocks, start=start, end=end, progress=False)

        # Instantiate placeholder
        keep_cols = []

        # Loop over each stock in input
        for i in stocks:

            # So we dont have to type this every time
            adj_close = "adj_close_{}".format(i)
            log_close = "log_adj_close_{}".format(i)
            log_returns = "log_returns_{}".format(i)

            # Keep track of these
            keep_cols.append(adj_close)
            keep_cols.append(log_close)
            keep_cols.append(log_returns)

            # Derive features off of adj close for each stock
            df[log_close] = np.log(df["Adj Close"][i])
            df[log_returns] = df[log_close] - df[log_close].shift(1)

            # This was done to make keeping columns simpler later
            df[adj_close] = df["Adj Close"][i]

            if show:

                plotStocks(False, i, df, alpha)

        # Subset to what we care about
        df = df[keep_cols]

        # Get rid of multi-index
        df.columns = ["".join(column) for column in df.columns.to_flat_index()]

        # Split into train/test
        # Test set is always the most recent data
        train_set, test_set = np.split(df, [int((1 - test_pct) * len(df))])

        return train_set, test_set


def brownianMotion(stock, df, days, trials, show):
    """
    Function that simulates Geometric Brownian Motion for a single stock
    with uncorrelated values to create'trials' paths forward looking 
    'days' ahead. Can be seen as equivalent to a simulateOnce() function 
    for a stock. The general form of geometric Brownian Motion in 1D is:

    S(t+1) = S(t)*exp((mean - 1/2(sigma)**2)*(t[n+1] - t[n]) + sigma*sqrt(t[n+1] - t[n])*Z(t+1))

    However, since we will be doing a random walk on each day, the values
    for the output of (t[n+1] - t[n]) will always be 1 and have no purpose 
    in the above equation. Thus, they will be removed from our implementation.

    Args:
    stock (str): name of stock ticker to use for lookup
    df (pandas dataframe): dataframe containing relevant columns for stock
    days (int): number of days ahead to simulate the adj close price for
    trials (int): number of simulations to run
    show (bool): whether to show plots

    Returns:
    numpy array containing price paths of shape (days, trials)
    """

    # log_returns_XXX will always be the 3rd position after we subset
    # to a specific stock
    log_returns = df.iloc[:, 2]

    # Use that to calculate mean and variance
    mew = log_returns.mean()

    ##############################################
    #   MORE SOPHISTICATED WAY TO REPLACE THIS   #
    var = log_returns.var()
    ##############################################

    # Need to subtract off 1/2 * var based on volatility drag effect
    drift = mew - (0.5 * var)

    # Compute volatility
    st_dev = log_returns.std()

    # Depends on inputs days and trials as this step generates
    # random variables for each day forecased and for the number
    # of simulations that are to be run for this stock
    # Z will have shape (days trials) which is equivalent to getting
    # a random variable from the standard normal distribution for each day
    # for each trial
    Z = np.random.normal(0, 1, (days, trials))

    # This is the daily return and the exponential part of the
    # Brownian Motion equation
    daily_returns = np.exp(drift + (st_dev * Z))

    # Instantiate placeholder
    price_paths = np.zeros_like(daily_returns)

    # Grab S(t-1) which is the last day from the training set
    # since we want to simulate moving forward in time from
    # then
    price_paths[0] = df.iloc[:, 0][-1]

    # We start at 1 since the first day (i.e. 0th position) is
    # already filled. And we loop
    for t in range(1, days):
        price_paths[t] = price_paths[t - 1] * daily_returns[t]

    if show:

        # Just show 30 of the potential 'trials' number of price
        # paths that were simulated
        plt.figure(figsize=(15, 6))
        plt.plot(pd.DataFrame(price_paths).iloc[:, 0:30])
        plt.title("30 modeled paths forward for {} stock".format(stock))
        plt.show()
        # print(price_paths)

    return price_paths


def brownianMotion_Cholesky(
    stocks, stick_df, days, trials, show, test
):
    df = []
    sigma = []
    # train, test = fetchStocks(stocks, "2019-01-01", "2021-06-12", 0.2, 0.05, False)

    # Acquire information
    for stock in stocks:
        df.append(stick_df["adj_close_{}".format(stock)].values)
        mew, sigma1 = getSimulatedVals(stick_df["adj_close_{}".format(stock)].values)
        sigma.append(sigma1)
    df1 = pd.DataFrame(df).T
    # COEF_MATRIX = np.cov(df1)  # df1.corr()
    COEF_MATRIX = df1.corr()
    #print(COEF_MATRIX.shape)

    #print(COEF_MATRIX)

    # Decomposition
    R = np.linalg.cholesky(COEF_MATRIX)

    # Loop through and multiply the matrix
    fullList = []
    for i in range(trials):
        T = days

        # Initialize the array
        stock_price_array = np.full(
            (len(stocks), T), test["adj_close_{}".format(stocks[0])].values[0]
        )
        index = 0

        # Adjust the starting prices accordingly
        for i in stock_price_array:
            i[0] = test["adj_close_{}".format(stocks[index])].values[0]
            index += 1

        # Some parameters
        volatility_array = sigma
        r = 0.001
        dt = 1.0 / T

        for t in range(1, T):
            # Generate array of random standard normal draws
            random_array = np.random.standard_normal(len(stocks))

            # Multiply R (from factorization) with random_array to obtain correlated epsilons
            epsilon_array = np.inner(random_array, R)

            # Sample price path per stock
            for n in range(len(stocks)):
                dt = 1 / T
                S = stock_price_array[n, t - 1]
                v = volatility_array[n]  # *10
                epsilon = epsilon_array[n]

                # Generate new stock price
                stock_price_array[n, t] = S * exp((r - 0.5 * v ** 2) * dt + v * epsilon)
                # stock_price_array[n,t] = S * exp((r - 0.5 * v**2) * dt + v * sqrt(dt) * epsilon)
                # daily_returns = np.exp(drift + (st_dev * Z))

        fullList.append(stock_price_array)

    # Plot simulated price paths
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        array_day_plot = [t for t in range(T)]

        for n in range(len(stocks)):
            ax.plot(
                array_day_plot,
                stock_price_array[n]
                - (
                    (test["adj_close_{}".format(stocks[n])].values[0])
                    - (test["adj_close_{}".format(stocks[0])].values[0])
                ),
                label="{}".format(stocks[n]),
            )

        plt.grid()
        plt.xlabel("Day")
        plt.ylabel("Asset price")
        plt.legend(loc="best")

        plt.show()

    # Transfer the result to fullArray to adjust the format
    fullArray = np.full((len(stocks), trials, days), 0.00)
    trialN = 0
    for trial in fullList:
        stockN = 0
        for stock in trial:
            recordN = 0
            for record in stock:
                fullArray[stockN][trialN][recordN] = fullList[trialN][stockN][recordN]
                recordN += 1
            stockN += 1
        trialN += 1

    return fullArray


def compareToTest(stock, paths, actual, confidence):
    """
    Function that takes as input the output to the brownianMotion
    function and calculates the average for each of the days simulated
    as well as the 'confidence' confidence interval values on those
    days. It then plots the simulated value for each day with its
    confidence interval against the actual value for that stock.

    Args:
    stock (str): name of stock ticker to use for lookup
    paths (numpy array): numpy array containing price paths of shape (days, trials)
    test (pandas dataframe): dataframe holding test data for stock
    confidence (float): percentage to use for the confidence interval
    """

    # Create placeholders
    means = []
    lower_conf = []
    upper_conf = []

    # Skip first row as that corresponds to last value from training
    # set and is not a generated path
    for path in paths[
        1:,
    ]:

        # Get mean for the row
        mew = np.mean(path)

        # Sort it
        path.sort()

        # Get upper and lower tails just like in bootstrap
        leftTail = int(((1.0 - (confidence)) / 2) * paths.shape[1])
        rightTail = (paths.shape[1] - 1) - leftTail

        # Append appropriate values to placeholder
        means.append(mew)
        lower_conf.append(path[leftTail])
        upper_conf.append(path[rightTail])

    # Plot predicted with confidence interval and actual value
    x = np.linspace(0, len(means), len(means))
    plt.figure(figsize=(15, 6))
    plt.plot(actual, "r", label="Actual movement")
    plt.plot(means, "b", label="Avg movement over all paths")
    plt.fill_between(x, lower_conf, upper_conf, alpha=0.2)
    plt.title("Predicted path vs actual path for {} stock".format(stock))
    plt.legend(loc="upper left")
    plt.show()


def getSimulatedVals(paths):
    """
    Function that takes as input the output from the brownianMotion
    function. The log returns for each of the rows in paths is derived
    and used to compute the mean and standard deviation of the log
    returns. These variables are returned for use in the portfolio
    management portion.

    Args:
    paths (numpy array_: contains price paths of shape (days, trials)

    Returns:
    mean and st dev of the log returns from the Brownian Motion simulation
    """

    # paths has adjusted close price values
    # We want to get the mean and stdev of the log returns
    # np.diff takes the difference between every consecutive pair of values.
    # Need axis = 0 in there to take difference from day to day
    log_returns = np.diff(np.log(paths), axis=0)

    # Get the mean
    mew = np.mean(log_returns)

    # Get the st dev
    sigma = np.std(log_returns)

    #print(log_returns.shape)

    return mew, sigma


def testSingleStock(stock, start_date, end_date, trials, show):
    """
    Single stock test case that includes fetching data, splitting
    data into train/test, showing plots, and using uncorrelated 
    Geometric Brownian Motion to simulate paths forward. The amount 
    of days to simulate is hardcoded to be equal to the number of 
    days found in the test set.

    Args:
    stock (str): name of stock ticker to use for lookup
    start_date (str): start date to bring data back from
    end_date (str): last day to bring data back from
    trials (int): number of Monte Carlo trials to run
    show (bool): whether or not ot show plots
    """

    # Fetch data
    train, test = fetchStocks(stock, start_date, end_date, 0.2, 0.05, show)

    # Grab adjusted close values from test set
    actual = test["adj_close"].values

    # Predict potential paths forward for stock equal to the size of
    # the test set
    paths = brownianMotion(stock, train, test.shape[0], trials, show)

    # Plot predicted path vs actual path for stock
    compareToTest(stock, paths, actual, 0.95)

    # Get mean and st dev of log returns for predicted paths
    mew, sigma = getSimulatedVals(paths)

    print("Mean log return over test set: {}".format(mew))
    print("Standard deviation log returns over test set: {}".format(sigma))


def testMultipleStock(stocks, start_date, end_date, trials, show):
    """
    Multiple stock test case that includes fetching data, splitting
    data into train/test, showing plots, and using correlated Geometric
    Brownian motion to simulate paths forward. The amount of days to 
    simulate is hardcoded to be equal to the number of days found in 
    the test set.

    Args:
    stock (str): name of stock ticker to use for lookup
    start_date (str): start date to bring data back from
    end_date (str): last day to bring data back from
    trials (int): number of Monte Carlo trials to run
    show (bool): whether or not ot show plots
    """

    # Fetch data
    train, test = fetchStocks(stocks, start_date, end_date, 0.2, 0.05, show)
    manyPaths = brownianMotion_Cholesky(
        stocks, train, test.shape[0], trials, False, test
    )

    # Another loop for handling results returned from brownianMotion_Cholesky().
    # Do not go to the loop blew since there are multiple stocks
    stockIndex = 0
    for paths in manyPaths:
        compareToTest(
            stocks[stockIndex],
            paths.T,
            test["adj_close_{}".format(stocks[stockIndex])].values,
            0.95,
        )

        # Get mean and st dev of log returns for predicted paths
        mew, sigma = getSimulatedVals(paths)

        print("Mean log return over test set: {}".format(mew))
        print("Standard deviation log returns over test set: {}".format(sigma))

        stockIndex += 1


if __name__ == "__main__":

    # Single stock test case
    testSingleStock("IBM", "2019-01-01", "2021-03-01", 1000, True)

    # Portfolio / list of stocks test case
    res = testMultipleStock(
        ["AAPL", "AMZN", "FB", "GOOG", "MSFT"], "2019-01-01", "2021-03-01", 1000, True
    )