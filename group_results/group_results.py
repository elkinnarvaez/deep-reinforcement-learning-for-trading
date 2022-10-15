import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import json
from pathlib import Path

experiments = ["experiment1", "experiment2"]

episodes_length = None
tickers = None
max_episodes = None

rolling_pct = 0.01


def graph_results(per_ticker_results, experiment):
    for ticker in tickers:
        results = per_ticker_results[ticker]
        # Evaluate results
        fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharey=True)

        df1 = (results[['Agent', 'Market']]
               .sub(1)
               .rolling(100)
               .mean())
        df1.plot(ax=axes[0],
                 title='Annual Returns (Moving Average) - {0}'.format(ticker),
                 lw=1)

        df2 = results['Strategy Wins (%)'].div(
            100).rolling(50).mean()
        df2.plot(ax=axes[1],
                 title='Agent Outperformance (%, Moving Average) - {0}'.format(ticker))

        for ax in axes:
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
        axes[1].axhline(.5, ls='--', c='k', lw=1)

        sns.despine()
        fig.tight_layout()
        if not Path(experiment).exists():
            Path(experiment).mkdir(parents=True)
        fig.savefig('{0}/performance_{1}'.format(experiment, ticker), dpi=300)


def main():
    global tickers, episodes_length, max_episodes
    for experiment in experiments:
        f = open("../experiments/{0}.json".format(experiment))
        experiment_metadata = json.load(f)
        f.close()

        tickers = experiment_metadata["tickers"]
        episodes_length = experiment_metadata["metadata"]["episodes_length"]

        training_result_paths = {ticker: [] for ticker in tickers}
        for d in sorted(os.listdir("../results/{0}".format(experiment)), key=lambda dir: int(dir[8:])):
            for ticker in tickers:
                training_result_paths[ticker].append(
                    "../results/{0}/{1}/results_{2}.csv".format(experiment, d, ticker))

        max_episodes = max(
            list(map(lambda dir: int(dir[8:]), os.listdir("../results/{0}".format(experiment)))))*episodes_length

        episodes = [i for i in range(1, max_episodes + 1)]

        per_ticker_results = {}
        for ticker in tickers:
            dfs = []
            for p in training_result_paths[ticker]:
                dfs.append(pd.read_csv(p))
            df = pd.concat(dfs)
            df['Episode'] = episodes
            df = df.set_index('Episode')
            per_ticker_results[ticker] = df.copy()
            per_ticker_results[ticker]['Strategy Wins (%)'] = (
                per_ticker_results[ticker].Difference > 0).rolling(100).sum()

        graph_results(per_ticker_results, experiment)


if __name__ == '__main__':
    main()
