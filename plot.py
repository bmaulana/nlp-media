import sys
import json
import os
import numpy as np
from dateutil.parser import parse as date_parse
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import LabelEncoder


def get_data(in_path):
    data = []
    f_in = open(in_path, 'r')
    for line in f_in:
        raw = json.loads(line)
        # Change to sentiment_score_openai (or whatever scorer) when on 'regular' out_sentiment
        if raw['sentiment_score'] != 'ERROR' and raw['sentiment_score'] != 0.0:
            data.append([date_parse(raw['datetime']), float(raw['sentiment_score']), raw['source']])
            # 'cap' outliers
            if data[-1][1] < -1.0:
                data[-1][1] = -1.0
            elif data[-1][1] > 1.0:
                data[-1][1] = 1.0
            if data[-1][0] < datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc):
                data[-1][0] = datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)
    f_in.close()
    return data


def plot(keyword, folder='./out-sentiment-vader/'):
    """
    Format: python plot.py filename
    """
    if not os.path.exists('./out-plot/'):
        os.makedirs('./out-plot/')

    data = []
    for filename in os.listdir(folder):
        if keyword in str(filename):
            data += get_data(folder + filename)
    data = np.array(data)
    data = data[np.argsort(data[:, 0])]  # sort on datetime

    le = LabelEncoder()
    colours = le.fit_transform(data[:, 2])
    sources = le.classes_
    # 2000 to 2019
    years = list(map(mdates.date2num,
                     np.array([datetime.datetime(i+2000, 1, 1, tzinfo=datetime.timezone.utc) for i in range(20)])))

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), tight_layout=True)
    # fig.suptitle(keyword)  # overlaps with plots for some reason
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

    # 2d histogram of no. of articles (Z), with each sentiment range (Y) and date range (X) separate
    _, _, _, img = axs[1, 0].hist2d(list(map(mdates.date2num, data[:, 0])), list(map(float, data[:, 1])),
                                    bins=(years, np.arange(-1.0, 1.1, 0.1)))
    axs[1, 0].set_xlabel('Date Range')
    axs[1, 0].set_ylabel('Sentiment Score')
    axs[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    for tick in axs[1, 0].get_xticklabels():
        tick.set_rotation(30)
    # From https://stackoverflow.com/questions/32462881/add-colorbar-to-existing-axis
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax=cax)

    # 2d histogram of no. of articles (Z), with each source (Y) and date range (X) separate
    _, _, _, img = axs[0, 1].hist2d(list(map(mdates.date2num, data[:, 0])), colours, bins=(years, len(sources)))
    axs[0, 1].set_xlabel('Date Range')
    axs[0, 1].set_ylabel('Source')
    axs[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axs[0, 1].set_yticks(np.arange(0, len(sources)))
    axs[0, 1].set_yticklabels(sources)
    for tick in axs[0, 1].get_xticklabels():
        tick.set_rotation(30)
    # From https://stackoverflow.com/questions/32462881/add-colorbar-to-existing-axis
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax=cax)

    # 2d histogram of no. of articles (Z), with each source (Y) and sentiment range (X) separate
    _, _, _, img = axs[1, 1].hist2d(colours, list(map(float, data[:, 1])),
                                    bins=(len(sources), np.arange(-1.0, 1.1, 0.1)))
    axs[1, 1].set_ylabel('Sentiment Score')
    axs[1, 1].set_xlabel('Source')
    axs[1, 1].set_xticks(np.arange(0, len(sources)))
    axs[1, 1].set_xticklabels(sources)
    # From https://stackoverflow.com/questions/32462881/add-colorbar-to-existing-axis
    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax=cax)

    # histogram of date range (X) vs no. of articles (Y)
    axs[0, 2].hist(data[:, 0], bins=years)
    axs[0, 2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axs[0, 2].set_xlabel('Date Range')
    axs[0, 2].set_ylabel('No. of articles')
    for tick in axs[0, 2].get_xticklabels():
        tick.set_rotation(30)

    # TODO: assume normal distribution. Get mean, variance, s.d., iqr of sentiment scores for each year and each source.
    # atm, the plots don't tell us much. You can tell that it's normally distributed, but need to know means and trends

    # plt.savefig('./out-plot/' + datetime.now().strftime('%Y%m%d%H%M%S') + '.png')
    plt.savefig('./out-plot/' + keyword + '.png')
    plt.close()


if __name__ == '__main__':
    plot(sys.argv[1])
