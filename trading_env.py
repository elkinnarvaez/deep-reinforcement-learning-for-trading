"""
The MIT License (MIT)

Copyright (c) 2016 Tito Ingargiola
Copyright (c) 2019 Stefan Jansen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging
import tempfile

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from sklearn.preprocessing import scale
import talib

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info('%s logger started.', __name__)


class DataSource:
    """
    Data source for TradingEnvironment

    Loads & preprocesses daily price & volume data
    Provides data for each new episode.
    Stocks with longest history:

    ticker  # obs
    KO      14155    --> Coca-Cola Co
    GE      14155    --> General Electric Company
    BA      14155    --> Boeing Co
    CAT     14155    --> Caterpillar Inc.
    DIS     14155    --> Walt Disney Co

    """

    def __init__(self, trading_days=252, tickers=['AAPL'], normalize=True, start_date='1995-01-01', end_date='2018-03-20'):
        self.tickers = tickers
        self.trading_days = trading_days
        self.normalize = normalize
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.load_data()
        self.min_values = self.data.min()
        self.max_values = self.data.max()
        self.step = 0
        self.offset = None

    def load_data(self):
        ticker_dfs = []
        for ticker in self.tickers:
            log.info('loading data for {}...'.format(ticker))
            idx = pd.IndexSlice
            with pd.HDFStore('data/assets.h5') as store:
                ticker_df = (store['quandl/wiki/prices']
                             .loc[idx[:, ticker],
                                  ['adj_close', 'adj_volume', 'adj_low', 'adj_high']]
                             .dropna()
                             .sort_index())
            ticker_df.columns = ['close', 'volume', 'low', 'high']
            log.info('got data for {}...'.format(ticker))

            log.info('preprocessing data for {}...'.format(ticker))
            ticker_df = self.preprocess_data(ticker_df, ticker)
            log.info('finished preprocessing for {}...'.format(ticker))

            ticker_dfs.append(ticker_df.copy())
        df = pd.concat(ticker_dfs)
        return df

    def preprocess_data(self, ticker_df, ticker):
        """calculate returns and percentiles, then removes missing values"""

        items = list(map(lambda date: (date, ticker), pd.date_range(
            start=self.start_date, end=self.end_date)))
        data = ticker_df.filter(items=items, axis=0)

        data['returns'] = data.close.pct_change()
        data['ret_2'] = data.close.pct_change(2)
        data['ret_5'] = data.close.pct_change(5)
        data['ret_10'] = data.close.pct_change(10)
        data['ret_21'] = data.close.pct_change(21)
        data['rsi'] = talib.STOCHRSI(data.close)[1]
        data['macd'] = talib.MACD(data.close)[1]
        data['atr'] = talib.ATR(data.high, data.low, data.close)
        slowk, slowd = talib.STOCH(data.high, data.low, data.close)
        data['stoch'] = slowd - slowk
        data['atr'] = talib.ATR(data.high, data.low, data.close)
        data['ultosc'] = talib.ULTOSC(data.high, data.low, data.close)

        up, mid, low = talib.BBANDS(data.close)
        data['bbp'] = (data.close - low) / (up - low)
        data['obv'] = talib.OBV(data.close, data.volume)
        data['adx'] = talib.ADX(data.high, data.low, data.close)

        data = (data.replace((np.inf, -np.inf), np.nan)
                .drop(['high', 'low', 'close', 'volume'], axis=1)
                .dropna())

        r = data.returns.copy()
        if self.normalize:
            data = pd.DataFrame(scale(data),
                                columns=data.columns,
                                index=data.index)
        features = data.columns.drop('returns')
        data['returns'] = r  # don't scale returns
        data = data.loc[:, ['returns'] + list(features)]
        return data

    def reset(self):
        """Provides starting index for time series and resets step"""
        high = min(len(self.data.loc[(slice(None), ticker), :].index)
                   for ticker in self.tickers) - self.trading_days
        self.offset = np.random.randint(low=0, high=high)
        self.step = 0

    def take_step(self):
        """Returns data for current trading day and done signal"""
        observations = {}
        for ticker in self.tickers:
            observations[ticker] = self.data.loc[(
                slice(None), ticker), :].iloc[self.offset + self.step].values
        self.step += 1
        done = self.step > self.trading_days
        return observations, done


class TradingSimulator:
    """ Implements core trading simulator for single-instrument univ """

    def __init__(self, steps, trading_cost_bps, time_cost_bps, tickers=['AAPL']):
        # invariant for object life
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.steps = steps
        self.tickers = tickers

        # change every step
        self.step = 0
        self.navs = {ticker: np.ones(self.steps) for ticker in self.tickers}
        self.market_navs = {ticker: np.ones(
            self.steps) for ticker in self.tickers}
        self.strategy_returns = {ticker: np.zeros(
            self.steps) for ticker in self.tickers}
        self.market_returns = {ticker: np.zeros(
            self.steps) for ticker in self.tickers}
        self.actions = {ticker: np.zeros(self.steps)
                        for ticker in self.tickers}
        self.positions = {ticker: np.zeros(self.steps)
                          for ticker in self.tickers}
        self.costs = {ticker: np.zeros(self.steps) for ticker in self.tickers}
        self.trades = {ticker: np.zeros(self.steps) for ticker in self.tickers}

    def reset(self):
        self.step = 0
        for ticker in self.tickers:
            self.navs[ticker].fill(1)
            self.market_navs[ticker].fill(1)
            self.strategy_returns[ticker].fill(0)
            self.market_returns[ticker].fill(0)
            self.actions[ticker].fill(0)
            self.positions[ticker].fill(0)
            self.costs[ticker].fill(0)
            self.trades[ticker].fill(0)

    def take_step(self, actions, market_returns):
        """ Calculates NAVs, trading costs and reward
            based on an action and latest market return
            and returns the reward and a summary of the day's activity. """
        ticker_rewards = {}
        ticker_navs = {}
        ticker_costs = {}
        for ticker in self.tickers:
            start_position = self.positions[ticker][max(0, self.step - 1)]
            start_nav = self.navs[ticker][max(0, self.step - 1)]
            start_market_nav = self.market_navs[ticker][max(0, self.step - 1)]
            self.market_returns[ticker][self.step] = market_returns[ticker]
            self.actions[ticker][self.step] = actions[ticker]

            end_position = actions[ticker] - 1  # short, neutral, long
            n_trades = end_position - start_position
            self.positions[ticker][self.step] = end_position
            self.trades[ticker][self.step] = n_trades

            # roughly value based since starting NAV = 1
            trade_costs = abs(n_trades) * self.trading_cost_bps
            time_cost = 0 if n_trades else self.time_cost_bps
            self.costs[ticker][self.step] = trade_costs + time_cost
            reward = start_position * \
                market_returns[ticker] - \
                self.costs[ticker][max(0, self.step-1)]
            self.strategy_returns[ticker][self.step] = reward

            if self.step != 0:
                self.navs[ticker][self.step] = start_nav * \
                    (1 + self.strategy_returns[ticker][self.step])
                self.market_navs[ticker][self.step] = start_market_nav * \
                    (1 + self.market_returns[ticker][self.step])
            ticker_rewards[ticker] = reward
            ticker_navs[ticker] = self.navs[ticker][self.step]
            ticker_costs[ticker] = self.costs[ticker][self.step]

        info = {'rewards': ticker_rewards,
                'navs': ticker_navs,
                'costs': ticker_costs}

        self.step += 1
        return ticker_rewards, info

    def result(self):
        """returns current state as pd.DataFrame """
        return pd.DataFrame({'action': self.actions,
                             'nav': self.navs,
                             'market_nav': self.market_navs,
                             'market_return': self.market_returns,
                             'strategy_return': self.strategy_returns,
                             'position': self.positions,
                             'cost': self.costs,
                             'trade': self.trades})


class TradingEnvironment(gym.Env):
    """A simple trading environment for reinforcement learning.

    Provides daily observations for a stock price series
    An episode is defined as a sequence of 252 trading days with random start
    Each day is a 'step' that allows the agent to choose one of three actions:
    - 0: SHORT
    - 1: HOLD
    - 2: LONG

    Trading has an optional cost (default: 10bps) of the change in position value.
    Going from short to long implies two trades.
    Not trading also incurs a default time cost of 1bps per step.

    An episode begins with a starting Net Asset Value (NAV) of 1 unit of cash.
    If the NAV drops to 0, the episode ends with a loss.
    If the NAV hits 2.0, the agent wins.

    The trading simulator tracks a buy-and-hold strategy as benchmark.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 trading_days=252,
                 trading_cost_bps=1e-3,
                 time_cost_bps=1e-4,
                 tickers=['AAPL'],
                 start_date='1995-01-01',
                 end_date='2018-03-20'):
        self.trading_days = trading_days
        self.trading_cost_bps = trading_cost_bps
        self.tickers = tickers
        self.time_cost_bps = time_cost_bps
        self.start_date = start_date
        self.end_date = end_date
        self.data_source = DataSource(trading_days=self.trading_days,
                                      tickers=self.tickers,
                                      start_date=self.start_date,
                                      end_date=self.end_date)
        self.simulator = TradingSimulator(steps=self.trading_days,
                                          trading_cost_bps=self.trading_cost_bps,
                                          time_cost_bps=self.time_cost_bps,
                                          tickers=self.tickers)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.data_source.min_values,
                                            self.data_source.max_values)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):
        """Returns state observation, reward, done and info"""
        observations, done = self.data_source.take_step()
        rewards, info = self.simulator.take_step(actions=actions,
                                                 market_returns={ticker: observations[ticker][0] for ticker in self.tickers})
        return observations, rewards, done, info

    def reset(self):
        """Resets DataSource and TradingSimulator; returns first observation"""
        self.data_source.reset()
        self.simulator.reset()
        return self.data_source.take_step()[0]

    # TODO
    def render(self, mode='human'):
        """Not implemented"""
        pass
