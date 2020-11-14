# region Strike Vector for Plotting
"""### Strike Vector for Plotting"""

strike = np.linspace(K - 30, K + 30, it)
# endregion

# region Plot Data
"""### Plot Data"""

average_geom = monte_carlo_geometric(it, N, K, S0, r, sigma)[0]
payoff_call = monte_carlo_geometric(it, N, K, S0, r, sigma)[1]
payoff_put = monte_carlo_geometric(it, N, K, S0, r, sigma)[2]
price = monte_carlo_geometric(it, N, K, S0, r, sigma)[3]
# endregion

# region Geometric Average Price
"""### Geometric Average Price"""

'''Geometric Average Price vs. Spot Price'''
avg_vs_spot = plt.figure()
plt.title('Geometric Average vs. Spot Price', color='black', fontsize=title_size)
plt.xlabel('Spot Price ($)', color='black', fontsize=xlabel_size)
plt.ylabel('Geometric Average ($)', color='black', fontsize=ylabel_size)
plt.plot(average_geom, price)
plt.show()

'''Geometric Average Price'''
g_avg = plt.figure()
plt.title('Geometric Average', color='black', fontsize=title_size)
plt.xlabel('Iterations', color='black', fontsize=xlabel_size)
plt.ylabel('Geometric Average ($)', color='black', fontsize=ylabel_size)
plt.plot(average_geom)
plt.show()
# endregion

# region Geometric Average Call Option
"""### Geometric Average Call Option"""

'''Call payoff vs. Spot Price'''
callpayoff_vs_spot = plt.figure()
plt.title('Call Option Payoff vs. Spot Price', color='black', fontsize=title_size)
plt.xlabel('Spot Price ($)', color='black', fontsize=xlabel_size)
plt.ylabel('Payoff ($)', color='black', fontsize=ylabel_size)
plt.plot(payoff_call, price)
plt.show()

'''Call payoff'''
callpayoff = plt.figure()
plt.title('Call Option Payoff', color='black', fontsize=title_size)
plt.xlabel('Iterations', color='black', fontsize=xlabel_size)
plt.ylabel('Payoff ($)', color='black', fontsize=ylabel_size)
plt.plot(payoff_call)
plt.show()

'''Call vs. Strike'''
callpayoff_vs_strike = plt.figure()
plt.title('Call Option Payoff vs. Strike', color='black', fontsize=title_size)
plt.xlabel('Strike ($)', color='black', fontsize=xlabel_size)
plt.ylabel('Payoff ($)', color='black', fontsize=ylabel_size)
plt.plot(payoff_call, np.abs(strike))
plt.show()
# endregion

# region Geometric Average Put Option
"""### Geometric Average Put Option"""

'''Put payoff vs. Spot Price'''
putpayoff__vs_spot = plt.figure()
plt.title('Put Option Payoff vs. Spot Price', color='black', fontsize=title_size)
plt.xlabel('Spot Price ($)', color='black', fontsize=xlabel_size)
plt.ylabel('Payoff ($)', color='black', fontsize=ylabel_size)
plt.plot(payoff_put, price)
plt.show()

'''Put payoff'''
putpayoff = plt.figure()
plt.title('Put Option Payoff', color='black', fontsize=title_size)
plt.xlabel('Iterations', color='black', fontsize=xlabel_size)
plt.ylabel('Payoff ($)', color='black', fontsize=ylabel_size)
plt.plot(payoff_put)
plt.show()

'''Put vs. Strike'''
putpayoff_vs_strike = plt.figure()
plt.title('Put Option Payoff vs. Strike', color='black', fontsize=title_size)
plt.xlabel('Strike ($)', color='black', fontsize=xlabel_size)
plt.ylabel('Payoff ($)', color='black', fontsize=ylabel_size)
plt.plot(payoff_put, strike)
plt.show()
# endregion

# region Pricer Benchmarking: Random Data Generation
"""### Pricer Benchmarking: Random Data Generation"""

'''Generate Random Si'''
Dt = monte_carlo_gbm(N, r, S0, sigma)[1]
S1 = monte_carlo_gbm(N, np.random.random(), np.random.uniform(np.random.random() * 10, 500.0), np.random.random())[0]
S2 = monte_carlo_gbm(N, np.random.random(), np.random.uniform(np.random.random() * 10, 500.0), np.random.random())[0]
S3 = monte_carlo_gbm(N, np.random.random(), np.random.uniform(np.random.random() * 10, 500.0), np.random.random())[0]
S4 = monte_carlo_gbm(N, np.random.random(), np.random.uniform(np.random.random() * 10, 500.0), np.random.random())[0]
S5 = monte_carlo_gbm(N, np.random.random(), np.random.uniform(np.random.random() * 10, 500.0), np.random.random())[0]
S6 = monte_carlo_gbm(N, np.random.random(), np.random.uniform(np.random.random() * 10, 500.0), np.random.random())[0]
S7 = monte_carlo_gbm(N, np.random.random(), np.random.uniform(np.random.random() * 10, 500.0), np.random.random())[0]
S8 = monte_carlo_gbm(N, np.random.random(), np.random.uniform(np.random.random() * 10, 500.0), np.random.random())[0]
S9 = monte_carlo_gbm(N, np.random.random(), np.random.uniform(np.random.random() * 10, 500.0), np.random.random())[0]
S10 = monte_carlo_gbm(N, np.random.random(), np.random.uniform(np.random.random() * 10, 500.0), np.random.random())[0]

'''Random Si plots'''
market_data_random = plt.figure()
plt.plot(Dt, S1, label='Stock 1')
plt.plot(Dt, S2, label='Stock 2')
plt.plot(Dt, S3, label='Stock 3')
plt.plot(Dt, S4, label='Stock 4')
plt.plot(Dt, S5, label='Stock 5')
plt.plot(Dt, S6, label='Stock 6')
plt.plot(Dt, S7, label='Stock 7')
plt.plot(Dt, S8, label='Stock 8')
plt.plot(Dt, S9, label='Stock 9')
plt.plot(Dt, S10, label='Stock 10')

plt.title('Randomly generated paths for 10 stocks', color='black', fontsize=title_size)
plt.xlabel('Dt', color='black', fontsize=xlabel_size)
plt.ylabel('Price ($)', color='black', fontsize=ylabel_size)
plt.legend(("Stock 1",
            'Stock 2',
            'Stock 3',
            'Stock 4',
            'Stock 5',
            'Stock 6',
            'Stock 7',
            'Stock 8',
            'Stock 9',
            'Stock 10'
            ),
           fontsize=legend_size
           )
plt.show()
# endregion

# region Pricer Benchmarking: Market Data Extraction
"""### Pricer Benchmarking: Market Data Extraction
After setting the desired parameters, run the cell using the execution button in the top left corner of the cell (Cell #22):
"""
# endregion

# region Historical Data Parameters
# @title Historical Data Parameters:

# Start date of historical data
start_date = "2000-01-01"  # @param {type:"date"}
# End date of historical data
end_date = "2020-12-31"  # @param {type:"date"}
# Type of historical data
data_type = "Adj Close"  # @param ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
# endregion

# region Random assortment of stocks
# Random assortment of stocks:
market_data_real_random = plt.figure()
msft = pdr.get_data_yahoo("MSFT", start=start_date, end=end_date)[data_type]
plt.plot(msft, label='Microsoft (MSFT)')

aapl = pdr.get_data_yahoo("AAPL", start=start_date, end=end_date)[data_type]
plt.plot(aapl, label='Apple (AAPL)')

gs = pdr.get_data_yahoo("GS", start=start_date, end=end_date)[data_type]
plt.plot(gs, label='Goldman Sachs (GS)')

nflx = pdr.get_data_yahoo("NFLX", start=start_date, end=end_date)[data_type]
plt.plot(nflx, label='Netflix (NFLX)')

fb = pdr.get_data_yahoo("FB", start=start_date, end=end_date)[data_type]
plt.plot(fb, label='Facebook (FB)')

f = pdr.get_data_yahoo("F", start=start_date, end=end_date)[data_type]
plt.plot(f, label='Ford (F)')

twtr = pdr.get_data_yahoo("TWTR", start=start_date, end=end_date)[data_type]
plt.plot(twtr, label='Twitter (TWTR)')

ibm = pdr.get_data_yahoo("IBM", start=start_date, end=end_date)[data_type]
plt.plot(ibm, label='IBM (IBM)')

baba = pdr.get_data_yahoo("BABA", start=start_date, end=end_date)[data_type]
plt.plot(baba, label='Alibaba (BABA)')

ba = pdr.get_data_yahoo("BA", start=start_date, end=end_date)[data_type]
plt.plot(ba, label='Boeing (BA)')

cme = pdr.get_data_yahoo("CME", start=start_date, end=end_date)[data_type]
plt.plot(cme, label='Chicago Mercantile Exchange (CME)')

# Citi never quite recovered from the downfall (staggering losses around 2008 and no subsequent rebound)
c = pdr.get_data_yahoo("C", start=start_date, end=end_date)[data_type]
plt.plot(c, label='Citigroup (C)')

ttl1 = "Historical"
ttl2 = data_type
ttl3 = "of selected Securities"
title = " ".join((ttl1, ttl2, ttl3))
plt.title(title, color='black', fontsize=title_size)
plt.xlabel('Date', color='black', fontsize=xlabel_size)

if data_type == "Volume":
    ylab = data_type + " (#)"
else:
    ylab = data_type + " ($)"

plt.ylabel(ylab, color='black', fontsize=ylabel_size)
plt.legend()
plt.show()
# endregion

# region Dow 30 stocks
# Dow 30 stocks:
market_data_real_dow30 = plt.figure()
mmm = pdr.get_data_yahoo("MMM", start=start_date, end=end_date)[data_type]
plt.plot(mmm, label='3M (MMM)')

axp = pdr.get_data_yahoo("AXP", start=start_date, end=end_date)[data_type]
plt.plot(axp, label='American Express (AXP)')

baba = pdr.get_data_yahoo("BABA", start=start_date, end=end_date)[data_type]
plt.plot(baba, label='Alibaba (BABA)')

aapl = pdr.get_data_yahoo("AAPL", start=start_date, end=end_date)[data_type]
plt.plot(aapl, label='Apple (AAPL)')

ba = pdr.get_data_yahoo("BA", start=start_date, end=end_date)[data_type]
plt.plot(ba, label='Boeing (BA)')

cat = pdr.get_data_yahoo("CAT", start=start_date, end=end_date)[data_type]
plt.plot(cat, label='Caterpillar (CAT)')

cvx = pdr.get_data_yahoo("CVX", start=start_date, end=end_date)[data_type]
plt.plot(cvx, label='Chevron (CVX)')

csco = pdr.get_data_yahoo("CSCO", start=start_date, end=end_date)[data_type]
plt.plot(csco, label='Cisco (CSCO)')

ko = pdr.get_data_yahoo("KO", start=start_date, end=end_date)[data_type]
plt.plot(ko, label='Coca Cola (KO)')

dis = pdr.get_data_yahoo("DIS", start=start_date, end=end_date)[data_type]
plt.plot(dis, label='The Walt Disney Company (DIS)')

dd = pdr.get_data_yahoo("DD", start=start_date, end=end_date)[data_type]
plt.plot(dd, label='DowDuPont (DD)')

xom = pdr.get_data_yahoo("XOM", start=start_date, end=end_date)[data_type]
plt.plot(xom, label='ExxonMobil (XOM)')

ge = pdr.get_data_yahoo("GE", start=start_date, end=end_date)[data_type]
plt.plot(ge, label='General Electric (GE)')

gs = pdr.get_data_yahoo("GS", start=start_date, end=end_date)[data_type]
plt.plot(gs, label='Goldman Sachs (GS)')

hd = pdr.get_data_yahoo("HD", start=start_date, end=end_date)[data_type]
plt.plot(hd, label='The Home Depot (HD)')

ibm = pdr.get_data_yahoo("IBM", start=start_date, end=end_date)[data_type]
plt.plot(ibm, label='IBM (IBM)')

intc = pdr.get_data_yahoo("INTC", start=start_date, end=end_date)[data_type]
plt.plot(intc, label='Intel (INTC)')

jnj = pdr.get_data_yahoo("JNJ", start=start_date, end=end_date)[data_type]
plt.plot(jnj, label='Johnson & Johnson (JNJ)')

jpm = pdr.get_data_yahoo("JPM", start=start_date, end=end_date)[data_type]
plt.plot(jpm, label='JPMorgan Chase (JPM)')

mcd = pdr.get_data_yahoo("MCD", start=start_date, end=end_date)[data_type]
plt.plot(mcd, label='McDonalds (MCD)')

mrk = pdr.get_data_yahoo("MRK", start=start_date, end=end_date)[data_type]
plt.plot(mrk, label='Merck (MRK)')

msft = pdr.get_data_yahoo("MSFT", start=start_date, end=end_date)[data_type]
plt.plot(msft, label='Microsoft (MSFT)')

nke = pdr.get_data_yahoo("NKE", start=start_date, end=end_date)[data_type]
plt.plot(nke, label='Nike (NKE)')

pfe = pdr.get_data_yahoo("PFE", start=start_date, end=end_date)[data_type]
plt.plot(pfe, label='Pfizer (PFE)')

pg = pdr.get_data_yahoo("PG", start=start_date, end=end_date)[data_type]
plt.plot(pg, label='Procter & Gamble (PG)')

trv = pdr.get_data_yahoo("TRV", start=start_date, end=end_date)[data_type]
plt.plot(trv, label='Travelers Companies, Inc (TRV)')

utx = pdr.get_data_yahoo("UTX", start=start_date, end=end_date)[data_type]
plt.plot(utx, label='United Technologies (UTX)')

unh = pdr.get_data_yahoo("UNH", start=start_date, end=end_date)[data_type]
plt.plot(unh, label='United Health (UNH)')

vz = pdr.get_data_yahoo("VZ", start=start_date, end=end_date)[data_type]
plt.plot(vz, label='Verizon (VZ)')

v = pdr.get_data_yahoo("V", start=start_date, end=end_date)[data_type]
plt.plot(v, label='Visa (V)')

wmt = pdr.get_data_yahoo("WMT", start=start_date, end=end_date)[data_type]
plt.plot(wmt, label='Wal-Mart (WMT)')

ttl1 = "Historical"
ttl2 = data_type
ttl3 = "of the Dow 30"
title = " ".join((ttl1, ttl2, ttl3))
plt.title(title, color='black', fontsize=title_size)
plt.xlabel('Date', color='black', fontsize=xlabel_size)

if data_type == "Volume":
    ylab = data_type + " (#)"
else:
    ylab = data_type + " ($)"

plt.ylabel(ylab, color='black', fontsize=ylabel_size)
plt.legend()
plt.show()
# endregion


# region Jokes
market_data_real_jokes = plt.figure()

# Citi never quite recovered from the downfall (staggering losses around 2008 and no subsequent rebound)
#c = pdr.get_data_yahoo("C", start=start_date, end=end_date)[data_type]
#plt.plot(c, label='Citigroup (C)')

# Kalispera
#lehkq = pdr.get_data_yahoo("LEHKQ", start=start_date, end=end_date)[data_type]
#plt.plot(lehkq, label='Lehman Brothers Holdings (LEHKQ)')

ddog = pdr.get_data_yahoo("DDOG", start=start_date, end=end_date)[data_type]
plt.plot(ddog, label='Datadog (DDOG)')

ttl1 = "Historical"
ttl2 = data_type
ttl3 = "of JOKES"
title = " ".join((ttl1, ttl2, ttl3))
plt.title(title, color='black', fontsize=title_size)
plt.xlabel('Date', color='black', fontsize=xlabel_size)

if data_type == "Volume":
    ylab = data_type + " (#)"
else:
    ylab = data_type + " ($)"

plt.ylabel(ylab, color='black', fontsize=ylabel_size)
plt.legend()
plt.show()
# endregion

# region Quandl Data

start = "2016-01-01"
end = "2016-12-31"

df = quandl.get("WIKI/AMZN", start_date = start, end_date = end)

adj_close = df['Adj. Close']
time = np.linspace(1, len(adj_close), len(adj_close))

plt.plot(time, adj_close)

def daily_return(adj_close):
    returns = []
    for i in xrange(0, len(adj_close)-1):
        today = adj_close[i+1]
        yesterday = adj_close[i]
        daily_return = (today - yesterday)/today
        returns.append(daily_return)
    return returns

returns = daily_return(adj_close)

mu = np.mean(returns)*252.
sig = np.std(returns)*np.sqrt(252.)

print mu, sig
# endregion