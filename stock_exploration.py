import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yh.stats as stats
import math
import numpy.random as rand
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import (
    median_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
)
from prophet import Prophet


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
        test_set, train_set = np.split(df, [int(test_pct * len(df))])

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


def brownianMotion(df, days, trials, show):
    """
    Function that simulates Brownian Motion for a single stock to create
    'trials' paths forward looking 'days' ahead. Can be seen as equivalent
    to a simulateOnce() function.

    Args:
    df (pandas dataframe): dataframe containing relevant columns for stock
    days (int): number of days ahead to simulate the adj close price for
    trials (int): number of simulations to run
    show (bool): whether to show plots
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
    # a random variable from the normal distribution for each day
    # for each trial
    Z = stats.norm.ppf(np.random.rand(days, trials))

    # This is the daily return and the exponential part of the
    # Brownian Motion equation
    daily_returns = np.exp(drift + st_dev * Z)

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
        plt.show()
        # print(price_paths)


def computePricePath_binomial(S, dT, duration, sigma, riskFreeRate):
    t = 0
    times = np.array([0])
    prices = np.array([S])

    u = math.exp(sigma * math.sqrt(dT))
    d = math.exp(-sigma * math.sqrt(dT))

    p = (math.exp(riskFreeRate * dT) - d) / (u - d)

    while t < duration:
        t += dT
        if rand.uniform() < p:
            S = S * u
        else:
            S = S * d

        times = np.append(times, t)
        prices = np.append(prices, S)

    return times, prices


def plotNPaths_binomials(n, last_price, duration):
    for i in range(n):
        t, p = computePricePath_binomial(last_price, 1, duration, 0.1, 0.03)

        plt.plot(t, p)
    plt.show()


def plotResultHistogram_binomials(n, last_price, duration):
    finalPrices = []
    for i in range(n):
        t, p = computePricePath(last_price, 1, duration, 0.1, 0.03)
        finalPrices.append(p[len(p) - 1])

    # Plot a histogram of the prices
    lPrices = np.log(finalPrices)
    plt.hist(lPrices, bins=30, density=True)

    # Overlay a normal distribution density function
    xmin, xmax = plt.xlim()  # get the x-limits of the plot
    mu, std = stats.norm.fit(lPrices)  # fit a normal distribution
    x = np.linspace(xmin, xmax, 100)  # Compute a set of 100 x's across the plot
    p = stats.norm.pdf(x, mu, std)  # Compute the normal probability distribution
    plt.plot(x, p, color="black", linewidth=2)  # Plot it

    result = stats.normaltest(np.log(finalPrices))
    print(round(result.pvalue, 2))  # Cannot reject hypothesis that it comes from a
    # normal dist. So the final prices follow a
    # log-normal distribution
    print("LogNormal Distribution: mu = %f, std.dev = %f" % (mu, std))


def computePricePath_continue(S, dT, duration, sigma, riskFreeRate):
    t = 0
    times = np.array([0])

    S = math.log(S)
    prices = np.array([S])

    periodRate = riskFreeRate * dT
    periodSigma = sigma * math.sqrt(dT)
    while t < duration:
        t += dT

        S += rand.normal(periodRate, periodSigma)

        times = np.append(times, t)
        prices = np.append(prices, S)

    prices = np.exp(prices)
    return times, prices


def plotNPaths_continue(n, last_price, duration):
    for i in range(n):
        t, p = computePricePath_continue(last_price, 1, duration, 0.1, 0.03)
        plt.plot(t, p)
    plt.show()


def plotResultHistogram_continue(n, last_price, duration):
    finalPrices = []
    for i in range(n):
        t, p = computePricePath_continue(last_price, 1, duration, 0.1, 0.03)
        finalPrices.append(p[len(p) - 1])

    # Plot a histogram of the prices
    lPrices = np.log(finalPrices)
    plt.hist(lPrices, bins=30, density=True)
    # Overlay a normal distribution density function
    xmin, xmax = plt.xlim()  # get the x-limits of the plot
    mu, std = stats.norm.fit(lPrices)  # fit a normal distribution
    x = np.linspace(xmin, xmax, 100)  # Compute a set of 100 x's across the plot
    p = stats.norm.pdf(x, mu, std)  # Compute the normal probability distribution
    plt.plot(x, p, color="black", linewidth=2)  # Plot it
    plt.show()

    plt.hist(finalPrices, bins=120, density=True)
    # Overlay a normal distribution density function
    xmin, xmax = plt.xlim()  # get the x-limits of the plot
    s, loc, scale = stats.lognorm.fit(lPrices)  # fit a normal distribution
    x = np.linspace(xmin, xmax, 100)  # Compute a set of 100 x's across the plot
    p = stats.lognorm.pdf(
        x, s, loc=loc, scale=scale
    )  # Compute the normal probability distribution
    plt.plot(x, p, color="black", linewidth=2)  # Plot it
    plt.show()
    # print(p)

    result = stats.normaltest(np.log(finalPrices))
    print(round(result.pvalue, 2))  # Cannot reject hypothesis that it comes from a
    # normal dist. So the final prices follow a
    # log-normal distribution
    print("LogNormal Distribution: mu = %f, std.dev = %f" % (mu, std))


def plot_moving_average(series, window, plot_intervals=False, scale=1.96):

    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(17, 8))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bound = rolling_mean - (mae + scale * deviation)
        upper_bound = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bound, "r--", label="Upper bound / Lower bound")
        plt.plot(lower_bound, "r--")

    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="best")
    plt.grid(True)


if __name__ == "__main__":

    ### SINGLE STOCK / STRING TEST CASE ###

    # train, test = fetchStocks('IBM', '2019-01-01', '2021-06-12', 0.2, 0.05, True)

    # Check to see if columns and values are as expected
    # print(test)

    # brownianMotion(train, 50, 1000, True)

    ### PORTFOLIO / LIST TEST CASE ###

    stocks = ["IBM", "AMZN"]
    train, test = fetchStocks(stocks, "2019-01-01", "2021-06-12", 0.2, 0.05, True)

    # Check to see if columns and values are as expected
    # print(test)

    # Brownian Motion function only takes a single stock as input
    # so need to subset to a single stock first and then run
    for stock in stocks:

        # Identify columns with stock name in them
        target_cols = [col for col in train.columns if stock in col]

        # Make a copy of both sets so we can run this in a loop
        train2 = train.copy(deep=True)
        test2 = test.copy(deep=True)

        # Subset copies to just the columns for a single stock
        train2 = train2[target_cols]
        test2 = test2[target_cols]

        # Predict on training set to verify functionality
        # brownianMotion(train2, 50, 1000, True)

        c = train2.columns[0]
        plotNPaths_binomials(50, train2[c][-1], len(test2))
        plotNPaths_continue(50, train2[c][-1], len(test2))

        plotResultHistogram_continue(50, train2[c][-1], len(test2))

        plot_moving_average(train2[c], 30, plot_intervals=True)

        train3 = train2
        train3["index1"] = train3.index

        data = []
        data = train3[["index1", c]]
        data = data.reset_index(drop=True)
        data.columns = ["ds", "y"]

        m = Prophet()
        m.fit(data)
        future = m.make_future_dataframe(periods=len(test2))
        forecast = m.predict(future)
        forecast.head()

        m.plot(forecast)
        m.plot_components(forecast)
