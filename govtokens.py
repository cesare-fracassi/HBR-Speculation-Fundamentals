"""
@author: Marco di Maggio, Cesare Fracassi, and Sam Shoar
------------------------------------------------------------------------------
Evaluation of crypto governance tokens using traditional finance models
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt



#%% Compute Betas

# From https://defillama.com/fees
df = pd.read_csv('data/CF.csv')

# create a list with the token names
tokens = df['token']

# create an empty list where we insert results  
betas = pd.DataFrame({'token' : [], 'beta' : []})

# download 10 year treasury note ETF price and use adjusted close prices
tn = yf.download(tickers = "IEF", start = "2021-08-31",
                     end = "2023-08-31", interval = "1d")
tn.rename(columns = {'Adj Close':'tn_close'}, inplace = True)
tn = tn['tn_close']


# download S&P (^GSPC) from y finance and use adjusted close prices
sp = yf.download (tickers = "^GSPC", start = "2021-08-31",
                     end = "2023-08-31", interval = "1d")
sp.rename(columns = {'Adj Close':'sp_close'}, inplace = True)
sp = sp['sp_close']

# Get data from SNOWFLAKE -- exchange.historic_market_data
tk_data = pd.read_csv('data/token_data.csv')

# create time index and match to eastern time zone
tk_data_times = pd.to_datetime(tk_data['START'], utc = True)
tk_data['date'] = pd.Index(tk_data_times).tz_convert('US/Eastern')

# this because the candle shows the start time of the period
tk_data_date = tk_data.loc[tk_data['date'].dt.hour == 15]

# remove the start time, only keep date
tk_data_date['date_day']= tk_data_date['date'].dt.date

# set day as index
tk_data_date.set_index('date_day', inplace = True)

# remove unwanted columns
tk_data_date = tk_data_date[['CLOSE', 'BASE_CURRENCY']]
    
sp = sp.tz_localize(None)

# merge datasets by index and remove index
data_merge = pd.merge(sp, tk_data_date, left_index=True, right_index=True,
                how = 'left')

tn = tn.tz_localize(None)
data_merge = data_merge.tz_localize(None)

data_merge = pd.merge(tn, data_merge, left_index=True, right_index=True, 
                      how = 'left')
data_merge.reset_index(inplace = True)

# rename columns
data_merge.rename(columns = {'index' : 'date', 'CLOSE':'token_close',
                       'BASE_CURRENCY' : 'token'}, inplace = True)

# reorder columns and set date and token name as indicies
data_merge.sort_values(by = ['token', 'date'], inplace = True)
data_merge.set_index(['date', 'token'], inplace = True)

# calculate daily percentage change  
data_final = data_merge.groupby(by = ['token']).pct_change(
    fill_method='ffill')

# rename columns
data_final.columns = ['tn_ret', 'sp_ret', 'token_ret']


# drop NAs
data_final = data_final[['sp_ret', 'token_ret']].dropna()
data_final.reset_index(inplace = True)

for tk in tokens:
    y = np.array(data_final.loc[data_final['token'] == tk]['token_ret'])
    x = np.array(data_final.loc[data_final['token'] == tk]['sp_ret']).reshape(-1, 1)

    # Debug: Print variables to check values
    print(tk)
    print(len(y))

    if len(y) > 0:  # Check if there is data for this token
        # Run regression and show results
        reg = LinearRegression().fit(x, y)

        # Add betas to the DataFrame
        temp = pd.DataFrame([[tk, reg.coef_[0]]], columns=['token', 'beta'])
        betas = betas.append(temp)
        print('Done!')
    else:
        print('No data for token:', tk)

# Debug: Print the resulting 'betas' DataFrame
print(betas)


# add beta values to dataframe
df = pd.merge(df,betas, on = 'token', how = 'left')


#%% Compute DCF VALUE

# assumptions:  risk free rate 3%, market risk premium 5%, growth rate 8%
rf_rt = 0.0409 # this is the RF rate of 10 year tresury as of 8/31/2023
mr_prem = 0.045
growth = 0.09
op_margin = 0.8


# earnings = operating margin * revenues                                                                                 
df['Earnings'] = op_margin * df['Revenues']



# r = risk free rate + (beta values * market premium)
df['r'] = rf_rt + (df['beta']*mr_prem)

# gordon dividend discount model = earnings / (disc_rt - (beta + mr_prem))  
df['GDDM'] = df['Treasury'] + df['Earnings'
                ] * (1+ growth) / (df['r'] - growth)
                 
# ratio to Market Cap
df['ratio'] = df['Market Cap'] / df['GDDM']

# pe ratio = market cap / earnings                                                                                 
df['PE'] = df['Market Cap'] / (df['Earnings'])

# forward ratio = forward market cap / GDDM model results
df['ratio_FD'] = df['Market Cap FD'] / df['GDDM']

# forward PE ratio = forward market cap / earnings                                                                                
df['PE_FD'] = df['Market Cap FD'] / (df['Earnings'])

# save as CSV
df.to_csv('results/govtokens.csv', index=False)




#%%
# COMPARE SP600IT with Tokens



# Download SP600IT market cap
#df2 = pd.read_csv('data/Data for Governance Token Article - SP600 IT.csv')

# create list with ticker names as seen on y finance
#tickers = df2['Symbol']

# Iterate through tickers and retrieve market caps
#market_cap_df = pd.DataFrame(columns=['Symbol', 'Market Cap'])

# Iterate through tickers and retrieve market caps
#for ticker in tickers:
#    ticker_data = yf.Ticker(ticker)
#    market_cap = ticker_data.info.get('marketCap')
#    market_cap_df = market_cap_df.append({'Symbol': ticker, 'Market Cap': market_cap}, ignore_index=True)

#df2 = pd.merge(df2, market_cap_df, on='Symbol', how='left')

# Save df2 as sp600it.csv
#df2.to_csv('results/sp600it.csv', index=False)

# THEN ADD MANUALLY THE REVENUE TTM AND NET INCOME TTM

tk = pd.read_csv('results/govtokens.csv')
sp = pd.read_csv('data/SP600IT.csv')


# Create the scatterplot
plt.figure(figsize=(15, 9))
plt.scatter(sp['Market Cap'], sp['TTM PE'], label='S&P 600 IT', alpha=0.5)  # Scatterplot for 'sp'
plt.scatter(tk['Market Cap FD'], tk['PE_FD'], label='Governance tokens', alpha=0.5)  # Scatterplot for 'tk'

# Set labels and title
plt.xlabel('Market Cap')
plt.ylabel('PE')
plt.yscale("log")
plt.title('P/E Ratio of Crytpo-Tokens Vs SP600IT')

# Add a legend
plt.legend()

# Save the plot
plt.grid(True)
plt.savefig('results/scatterplot.png')
plt.show()
