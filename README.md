## Simple monte carlo pricing of a Bitcoin call option - how assumptions can affect pricing by factor 100

Using simulations will help us understand why placing a value on a Bitcoin option is hard. It also serves as a good example why relying purely on past historical data can be dangerous.

Valuation through monte carlo is done by simulating thousands of future Bitcoin price paths using past dynamics. Having a large amount of these simulated Bitcoin price paths allows us to get an idea of the option’s value.

Simulation is based on historical distribution of daily returns of bitcoin. 

### Sample output:
```
call (strike= 20000, expiration days=365) option value:   3072.1 - took 0.6 sec
put  (strike= 20000, expiration days=365) option value:   8812.2 - took 0.6 sec
```

Vanilla call and put option prices with strike price (*K*) 20000, days until expiry (*T*) 365, risk free rate (*rf*) 1% and current price of $14,199

| Return Distribution        | Call           | Put  |
| ------------- |:-------------:| -----:|
| historical      | $189,080 | $34 |
| historical de-trended      |  $3,072 |   $8,812 |
| S&P500 historical | $209  |    $2,975 |

Note: The call option value using raw historical data is nonsensical (option worth more than buying btc outright). Using raw historical data has its return distribution skewed - as reflected in Bitcoin’s meteoric price increase.

