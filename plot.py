import sys
import json
import os
import numpy as np
from dateutil.parser import parse as date_parse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import LabelEncoder


def get_data(in_path):
    data = []
    f_in = open(in_path, 'r')
    for line in f_in:
        raw = json.loads(line)
        if raw['sentiment_score_openai'] != 'ERROR':
            data.append([date_parse(raw['datetime']), float(raw['sentiment_score_openai']), raw['source']])
            # 'cap' outliers
            if data[-1][1] < -1.0:
                data[-1][1] = -1.0
            elif data[-1][1] > 1.0:
                data[-1][1] = 1.0
    f_in.close()
    return data


def plot(fnames):
    """
    Format: python plot.py filename
    """
    if not os.path.exists('./out-plot/'):
        os.makedirs('./out-plot/')

    data = []
    for fname in fnames:
        data += get_data('./out-sentiment/' + fname)
    data = np.array(data)
    data = data[np.argsort(data[:, 0])]  # sort on datetime

    le = LabelEncoder()
    colours = le.fit_transform(data[:, 2])
    # line below will throw a warning, known bug in LabelEncoder (github.com/scikit-learn/scikit-learn/issues/10449)
    sources = le.inverse_transform(np.arange(len(fnames)))

    fig, axs = plt.subplots(2, 3, figsize=(10, 15), tight_layout=True)
    # fig.suptitle(fnames)  # overlaps with plots for some reason
    # fig.autofmt_xdate()
    plt.xticks(rotation=30)

    # scatter plot of date range (X) vs sentiment (Y), coloured based on source (Z)
    axs[0, 0].scatter(data[:, 0], data[:, 1], c=colours)
    # TODO rolling average 'line'
    axs[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    axs[0, 0].set_xlabel('Date Range')
    axs[0, 0].set_ylabel('Sentiment Score')
    for tick in axs[0, 0].get_xticklabels():
        tick.set_rotation(30)

    # histogram of date range (X) vs no. of articles (Y)
    axs[1, 0].hist(data[:, 0], bins=10)
    axs[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    axs[1, 0].set_xlabel('Date Range')
    axs[1, 0].set_ylabel('No. of articles')
    for tick in axs[1, 0].get_xticklabels():
        tick.set_rotation(30)

    # 2d histogram of no. of articles (Z), with each source (Y) and date range (X) separate
    _, _, _, img = axs[0, 1].hist2d(list(map(mdates.date2num, data[:, 0])), colours, bins=(10, len(fnames)))
    axs[0, 1].set_xlabel('Date Range')
    axs[0, 1].set_ylabel('Source')
    axs[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    axs[0, 1].set_yticks(np.arange(0, len(fnames)))
    axs[0, 1].set_yticklabels(sources)
    for tick in axs[0, 1].get_xticklabels():
        tick.set_rotation(30)
    # From https://stackoverflow.com/questions/32462881/add-colorbar-to-existing-axis
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax=cax)

    # 2d histogram of no. of articles (Z), with each source (Y) and sentiment range (X) separate
    _, _, _, img = axs[1, 1].hist2d(list(map(float, data[:, 1])), colours, bins=(10, len(fnames)))
    axs[1, 1].set_xlabel('Sentiment Score')
    axs[1, 1].set_ylabel('Source')
    axs[1, 1].set_yticks(np.arange(0, len(fnames)))
    axs[1, 1].set_yticklabels(sources)
    # From https://stackoverflow.com/questions/32462881/add-colorbar-to-existing-axis
    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax=cax)

    # 2d histogram of no. of articles (Z), with each sentiment range (Y) and date range (X) separate
    _, _, _, img = axs[0, 2].hist2d(list(map(mdates.date2num, data[:, 0])), list(map(float, data[:, 1])), bins=(10, 10))
    axs[0, 2].set_xlabel('Date Range')
    axs[0, 2].set_ylabel('Sentiment Score')
    axs[0, 2].xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    for tick in axs[0, 2].get_xticklabels():
        tick.set_rotation(30)
    # From https://stackoverflow.com/questions/32462881/add-colorbar-to-existing-axis
    divider = make_axes_locatable(axs[0, 2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax=cax)

    # plt.savefig('./out-plot/' + datetime.now().strftime('%Y%m%d%H%M%S') + '.png')
    plt.savefig('./out-plot/' + ','.join(fnames) + '.png')
    plt.close()


if __name__ == '__main__':
    plot(str(sys.argv[1]).split(','))
