import sys
import json
import os
import numpy as np
from dateutil.parser import parse as date_parse
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import LabelEncoder


def get_data(in_path):
    data = []
    f_in = open(in_path, 'r')
    for line in f_in:
        raw = json.loads(line)
        if raw['sentiment_score_openai'] != 'ERROR':
            data.append([date_parse(raw['datetime']), float(raw['sentiment_score_openai']), raw['source']])
    f_in.close()
    return data


def plot(fnames):
    """
    Format: python plot.py filename
    """
    # TODO enable developing 'combined' plots from several different JSON files & colouring based on source;
    if not os.path.exists('./out-plot/'):
        os.makedirs('./out-plot/')

    data = []
    for fname in fnames:
        data += get_data('./out-sentiment/' + fname)
    data = np.array(data)
    data = data[np.argsort(data[:, 0])]  # sort on datetime

    # scatterplot of date range (X) vs sentiment (Y), coloured based on source (Z)
    le = LabelEncoder()
    colours = le.fit_transform(data[:, 2])
    plt.scatter(data[:, 0], data[:, 1], c=colours)
    # TODO rolling average 'line'
    plt.title(fnames)
    plt.xlabel('Date Range')
    plt.ylabel('Sentiment Score')
    plt.gcf().autofmt_xdate()
    # plt.savefig('./out-plot/' + datetime.now().strftime('%Y%m%d%H%M%S') + '.scatter.png')
    plt.savefig('./out-plot/' + ','.join(fnames) + '.scatter.png')
    plt.close()

    # histogram of date range (X) vs no. of articles
    plt.hist(data[:, 0], bins=10)
    plt.title(fnames)
    plt.xlabel('Date Range')
    plt.ylabel('No. of articles')
    plt.gcf().autofmt_xdate()
    plt.savefig('./out-plot/' + ','.join(fnames) + '.histogram.png')
    plt.close()

    # 2d histogram, with each source separate TODO fix labels
    plt.hist2d(list(map(mdates.date2num, data[:, 0])), colours, bins=(10, len(fnames)))
    plt.title(fnames)
    plt.xlabel('Date Range')
    plt.ylabel('Source')
    plt.gcf().autofmt_xdate()
    plt.savefig('./out-plot/' + ','.join(fnames) + '.histogram2d.png')
    plt.close()

    # TODO plot also 2d histograms of date-sentiment and source-sentiment, present all 4 histograms in one figure


if __name__ == '__main__':
    plot(str(sys.argv[1]).split(','))
