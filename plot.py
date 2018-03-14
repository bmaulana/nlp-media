import sys
import json
import os
import time
import numpy as np
from dateutil.parser import parse as date_parse
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot(fnames):
    """
    Format: python plot.py filename
    """
    # TODO enable developing 'combined' plots from several different JSON files & colouring based on source;
    if not os.path.exists('./out-plot/'):
        os.makedirs('./out-plot/')
    in_path = './out-sentiment/' + fnames[0]
    # out_path = './out-plot/' + str(time.time()) + TODO extension

    data = []
    f_in = open(in_path, 'r')
    for line in f_in:
        raw = json.loads(line)
        if raw['sentiment_score_openai'] != 'ERROR':
            data.append([date_parse(raw['datetime']), float(raw['sentiment_score_openai']), raw['source']])
    f_in.close()
    data = np.array(data)
    data = data[np.argsort(data[:, 0])]  # sort on datetime

    # TODO plot histogram of date range (X) vs sentiment (Y), coloured based on source (Z)
    plt.scatter(data[:, 0], data[:, 1])
    plt.title(fnames[0])
    plt.xlabel('Date Range')
    plt.ylabel('Sentiment Score')
    plt.show()

    # f_out = open(out_path, 'w')
    # TODO save plot
    # f_out.close()


if __name__ == '__main__':
    plot([sys.argv[1]])
