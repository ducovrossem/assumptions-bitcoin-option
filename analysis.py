import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt

from plot_return_distribution import plot_retdist

def sim_rets(m=0, sd = 1.2, N = 10000):
    return np.random.normal(m, sd, size=(N)) / 100.

# Import Bitcoin prices (source: www.coindesk.com/price/)
df = pd.read_csv("data/coindesk-bpi-USD-ohlc_data-2010-07-01_2018-01-14.csv")
pricevec = np.array(df.Close[~pd.isnull(df.Close)])

# Compute daily return Bitcoin
returns_btc = (pricevec[1:(len(pricevec))] - pricevec[0:(len(pricevec)-1)]) / pricevec[0:(len(pricevec)-1)]

# Import S&P500 prices:
df500 = pd.read_csv("data/SP500.csv")
pricevec_sp500 = np.array(df500.SP500[df500.SP500 != '.'],dtype=float)

# Compute daily return S&P500
returns_sp500 = (pricevec_sp500[1:(len(pricevec_sp500))] - pricevec_sp500[0:(len(pricevec_sp500)-1)]) / pricevec_sp500[0:(len(pricevec_sp500)-1)]

num_historical_days = 500

# Limit to only use the most recent num_historical_days days in our analysis:
retshist = returns_btc[-num_historical_days:]
S_t = pricevec[-num_historical_days:]

retshist_sp500 = returns_sp500[-num_historical_days:]

print("Mean daily BTC return over the last {} days: {:.3f}%".format(num_historical_days, 100*np.mean(returns_btc)))

plot_retdist(retshist, retshist2=retshist_sp500, norm=True,add_legend = ['Bitcoin', 'S&P500'],min_upperlimit= 0.13, max_lowerlimit=-0.13)

# Simulate prices for figure

totallen = 700
num_days_to_sim = totallen - len(S_t)
S_0 =  S_t[-1]
xlist = range(totallen - num_days_to_sim-1,totallen)

curves_add = 5

plt.plot(S_t)
plt.xlim(0, totallen)
for iiii in range(curves_add):
    sampled_returns = np.append(0, np.random.choice(retshist, num_days_to_sim))
    sim_pt = np.cumprod(sampled_returns + 1)*S_0
    plt.plot(xlist, sim_pt, ls = '--')

# plt.axhline(y=strike_price, xmin=0, xmax=totallen, linewidth=2, color = 'k')
plt.ylabel('BTC price')
plt.xlabel('Date')
plt.title("Simulated bitcoin price paths ")
plt.show()

# MC option valuation

current_price = S_t[-1]
riskfree_rate = 1.01
strike_price = 20000
number_of_simulations = 30000
days_until_expiry = 365 # There are 365 bitcoin trading days in a year

def MC_value_simulations(returns_to_sample, S0, rf, K, T, num_simulations = 10000, option_type = 'call'):
    """
    Value option with parameters (S0, rf, K, R) using a simple Monte Carlo simulation that samples daily return
    from returns_to_sample.

    :param returns_to_sample: nparray of daily returns
    :param S0: starting price
    :param rf: riskfree rate
    :param K: strike price
    :param T: under of days until expiry
    :param num_simulations: number of price paths to simulate to generate an optino value
    :param option_type: call
    :return:
    """

    if option_type == 'call':
        option_payoff = lambda x:max(x - K, 0)
    elif option_type == 'put':
        option_payoff = lambda x:max(K - x, 0)
    else:
        raise Exception("Unknown option type {} found".format(option_type))

    vals_sum = np.zeros(num_simulations)
    starttime = time.time()

    for ii in range(num_simulations):
        sampled_return_array = np.random.choice(returns_to_sample, T)
        cumalative_return = np.prod(sampled_return_array + 1)
        S_T = S0 * cumalative_return
        option_value_T = option_payoff(S_T)
        vals_sum[ii] = option_value_T

    years = T/365.

    # Discount value
    option_value = np.mean(vals_sum)/(1+(rf-1)*years)

    print("{:4} option value: {:8} - Took {} sec".format(option_type, round(option_value, 1), round(time.time() - starttime, 1)))

    return option_value


# Run option value simulation

OV_call = MC_value_simulations(retshist, S0=current_price, rf=riskfree_rate, K=strike_price, T=days_until_expiry, num_simulations=number_of_simulations)
OV_put = MC_value_simulations(retshist, S0=current_price, rf=riskfree_rate, K=strike_price, T=days_until_expiry, num_simulations=number_of_simulations, option_type='put')


# Simple de-trending to remove the inherent upward trend in the simulations

detrended_return_history = retshist - np.mean(retshist)

plot_retdist(retshist, retshist2=detrended_return_history, norm=True,add_legend = ['historical', 'detrended'],min_upperlimit= 0.13, max_lowerlimit=-0.13)

OV_call_detrended = MC_value_simulations(detrended_return_history, S0=current_price, rf=riskfree_rate, K=strike_price, T=days_until_expiry, num_simulations=number_of_simulations)
OV_put_detrended = MC_value_simulations(detrended_return_history, S0=current_price, rf=riskfree_rate, K=strike_price, T=days_until_expiry, num_simulations=number_of_simulations, option_type='put')

# Using S&P500 returns

OV_call_sp500 = MC_value_simulations(returns_sp500, S0=current_price, rf=riskfree_rate, K=strike_price, T=days_until_expiry, num_simulations=number_of_simulations)
OV_put_sp500 = MC_value_simulations(returns_sp500, S0=current_price, rf=riskfree_rate, K=strike_price, T=days_until_expiry, num_simulations=number_of_simulations, option_type='put')

