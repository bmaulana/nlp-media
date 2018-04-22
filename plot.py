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
from collections import defaultdict
from scipy.stats import mannwhitneyu


class Reader:

    def __init__(self):
        self.id = 0

    def get_data(self, in_path):
        data = []
        keywords = {}
        f_in = open(in_path, 'r')
        for line in f_in:
            raw = json.loads(line)
            if raw['sentiment_score'] != 'ERROR' and raw['sentiment_score'] != 0.0:
                try:
                    data.append([date_parse(raw['datetime']), float(raw['sentiment_score']), raw['source'], self.id])
                    keywords[self.id] = raw['keywords_used']
                    self.id += 1
                except ValueError:  # invalid date or sentiment score
                    continue
                # 'cap' outliers
                if data[-1][1] < -1.0:
                    data[-1][1] = -1.0
                elif data[-1][1] > 1.0:
                    data[-1][1] = 1.0
                if data[-1][0] < datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc):
                    data[-1][0] = datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)
        f_in.close()
        return data, keywords


def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')


# Only stems plurals and tenses. To stem keyword usage trends while not removing distinction of e.g. 'ill' vs 'illness'.
def basic_stemmer(word):
    word = word.split(' ')
    if len(word) > 1:
        ret = ''
        for w in word:
            ret += basic_stemmer(w) + ' '
        return ret[:-1]
    word = word[0]
    if word.endswith('ings'):
        word = word[:-4]
    elif word.endswith('ied'):
        word = word[:-3] + 'y'
    elif word.endswith('ies'):
        word = word[:-3] + 'y'
    elif word.endswith('ing'):
        word = word[:-3]
    elif word.endswith('eds'):
        word = word[:-3]
    elif word.endswith('ed'):
        word = word[:-2]
    elif word.endswith('es'):
        word = word[:-2]
    elif word.endswith('e'):  # to match (removed) -ed and -es forms, words missing 'e' at the end would be obvious anw
        word = word[:-1]
    elif word.endswith('s'):
        word = word[:-1]
    if word[-2] == word[-1]:  # double consonants before -ing or -ed (e.g. map -> mapped/mapping)
        word = word[:-1]
    return word


def plot(keyword, in_folder='./out-sentiment-vader/', out_folder='./out-plot-vader/'):
    """
    Format: python plot.py keyword
    """
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    reader = Reader()
    data, keywords = [], {}
    for filename in os.listdir(in_folder):
        if keyword in str(filename):
            d, k = reader.get_data(in_folder + filename)
            data += d
            for i, v in k.items():
                keywords[i] = v
    data = np.array(data)
    data = data[np.argsort(data[:, 0])]  # sort on datetime

    moving_avg_window = data.shape[0] // 10
    moving_avg_window = min(500, max(50, moving_avg_window))  # 10% of dataset size, capped to 50-500

    le = LabelEncoder()
    colours = le.fit_transform(data[:, 2])
    sources = le.classes_
    # 2000 to 2019
    years = list(map(mdates.date2num,
                     np.array([datetime.datetime(i+2000, 1, 1, tzinfo=datetime.timezone.utc) for i in range(20)])))

    fig, axs = plt.subplots(3, 3, figsize=(15, 15), tight_layout=True)
    # fig.suptitle(keyword)  # overlaps with plots for some reason
    # fig.autofmt_xdate()
    plt.xticks(rotation=30)

    # scatter plot of date range (X) vs sentiment (Y), coloured based on source (Z)
    axs[0, 0].scatter(data[:, 0], data[:, 1], c=colours)
    rolling_avg = moving_average(data[:, 1], moving_avg_window)
    axs[0, 0].plot(data[:, 0][moving_avg_window - 1:], rolling_avg, 'r')
    axs[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
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
    axs[0, 2].xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    axs[0, 2].set_xlabel('Date Range')
    axs[0, 2].set_ylabel('No. of articles')
    for tick in axs[0, 2].get_xticklabels():
        tick.set_rotation(30)

    # histogram of sentiment score (X) vs no. of articles (Y)
    # axs[1, 2].hist(list(map(float, data[:, 1])), bins=np.arange(-1.0, 1.1, 0.1))
    # axs[1, 2].set_xlabel('Sentiment Score')
    # axs[1, 2].set_ylabel('No. of articles')

    # plot line graphs for annual mean (or rolling average mean) for each source
    rolling_avg = moving_average(data[:, 1], moving_avg_window)
    axs[1, 2].plot(data[:, 0][moving_avg_window - 1:], rolling_avg, 'r', label='all')
    for src in sources:
        src_data = data[data[:, 2] == src]
        if src_data.shape[0] < moving_avg_window:
            continue
        src_avg = moving_average(src_data[:, 1], moving_avg_window)
        axs[1, 2].plot(src_data[:, 0][moving_avg_window - 1:], src_avg, label=src)
    axs[1, 2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axs[1, 2].set_xlabel('Date Range')
    axs[1, 2].set_ylabel('Sentiment Score')
    axs[1, 2].legend()

    # Violin plot of source (X) and sentiment range (Y)
    sources_data = []
    for source in sources:
        sources_data.append(np.array(data[data[:, 2] == source][:, 1], dtype=np.float32))

    axs[2, 0].violinplot(sources_data, [i+1 for i in range(len(sources))])
    axs[2, 0].boxplot(sources_data, [i for i in range(len(sources))], '')
    axs[2, 0].set_ylabel('Sentiment Score')
    axs[2, 0].set_xlabel('Source')
    axs[2, 0].set_xticks(np.arange(1, len(sources) + 1))
    axs[2, 0].set_xticklabels(sources)

    # Violin plot of date range (X) and sentiment range (Y)
    years = np.array([datetime.datetime(i + 2000, 1, 1, tzinfo=datetime.timezone.utc) for i in range(0, 22, 2)])
    years_data = [[] for i in range(len(sources))]
    for i in range(years.shape[0] - 1):
        for j in range(len(sources)):
            to_append = np.array([a[1] for a in data[data[:, 2] == sources[j]] if years[i] <= a[0] < years[i + 1]],
                                 dtype=np.float32)
            years_data[j].append(to_append if to_append.shape[0] > 0 else np.zeros(1))

    for i in range(len(years_data)):
        axs[2, 1].violinplot(years_data[i][:5], [i for i in range(2000, 2010, 2)], showmeans=True, widths=0.8)
        axs[2, 2].violinplot(years_data[i][5:], [i for i in range(2010, 2020, 2)], showmeans=True, widths=0.8)

    axs[2, 1].set_xlabel('Date Range')
    axs[2, 1].set_ylabel('Sentiment Score')
    for tick in axs[2, 1].get_xticklabels():
        tick.set_rotation(30)

    axs[2, 2].set_xlabel('Date Range')
    axs[2, 2].set_ylabel('Sentiment Score')
    for tick in axs[2, 2].get_xticklabels():
        tick.set_rotation(30)

    # TODO Violin plot of sentiment range (Y) and date range (X) + source (colour)

    # plt.savefig('./out-plot/' + datetime.now().strftime('%Y%m%d%H%M%S') + '.png')
    plt.savefig(out_folder + keyword + '.png')
    plt.close()

    # Print some additional info to a .txt file
    f_out = open(out_folder + keyword + '.txt', 'w')

    # Mann-Whitney U Test
    #
    # Note: A small p-value (typically ≤ 0.05) indicates strong evidence against the null hypothesis,
    # so you reject the null hypothesis. A large p-value (> 0.05) indicates weak evidence against the null
    # hypothesis, so you fail to reject the null hypothesis.
    #
    # Null hypothesis: it is equally likely that a randomly selected value from one sample will be less than or greater
    # than a randomly selected value from a second sample. (thus if not rejected, no significant difference)
    #
    # (Wikipedia)
    for i in range(len(sources)):
        for j in range(len(sources)):
            if i != j:
                f_out.write(sources[i] + '>' + sources[j] + ': ' +
                            str(mannwhitneyu(sources_data[i], sources_data[j], alternative='greater')) + '\n')

    years = np.array([datetime.datetime(i + 2000, 1, 1, tzinfo=datetime.timezone.utc) for i in range(20)])
    for i in range(len(years)-1):
        data_in_year = np.array([a for a in data if years[i] <= a[0] < years[i+1]])
        if len(data_in_year) == 0:
            continue
        f_out.write(str(years[i].year) + ':\n')
        for src1 in sources:
            for src2 in sources:
                if src1 != src2 and len(data_in_year[data_in_year[:, 2] == src1]) > 20 \
                        and len(data_in_year[data_in_year[:, 2] == src2]) > 20:
                    f_out.write('\t' + src1 + '>' + src2 + ': ' +
                                str(mannwhitneyu(data_in_year[data_in_year[:, 2] == src1][:, 1],
                                                 data_in_year[data_in_year[:, 2] == src2][:, 1], alternative='greater'))
                                + '\n')

    f_out.write('\n')

    print('Topic', keyword)
    print(data.shape[0], 'total articles')

    # Means, s.d., no. of articles for whole data set
    f_out.write(str(data.shape[0]) + ' total articles with mean sentiment ' +
                str(np.average(np.array(data[:, 1], dtype=np.float32))) +
                ' and std. dev. ' + str(np.std(data[:, 1], dtype=np.float32)) + '\n\n')

    # Means, s.d., and no. of articles for each year and each source
    years = np.array([datetime.datetime(i+2000, 1, 1, tzinfo=datetime.timezone.utc) for i in range(20)])
    for i in range(len(years)-1):
        data_in_year = np.array([a[1] for a in data if years[i] <= a[0] < years[i+1]], dtype=np.float32)
        # print(years[i].year, ':', data_in_year.shape[0], 'articles with mean sentiment', np.average(data_in_year))
        f_out.write(str(years[i].year) + ': ' + str(data_in_year.shape[0]) + ' articles with mean sentiment ' +
                    str(np.average(data_in_year)) + ' and std. dev. ' + str(np.std(data_in_year)) + '\n')

    for source in sources:
        data_in_source = np.array(data[data[:, 2] == source][:, 1], dtype=np.float32)
        # print(source, ':', data_in_source.shape[0], 'articles with mean sentiment', np.average(data_in_source))
        f_out.write(source + ': ' + str(data_in_source.shape[0]) + ' articles with mean sentiment ' +
                    str(np.average(data_in_source)) + ' and std. dev. ' + str(np.std(data_in_source)) + '\n')
        print(data_in_source.shape[0], 'articles from', source)

    f_out.write('\n')

    # Print no. of articles, mean, s.d., for each (year, source) combination.
    for source in sources:
        for i in range(len(years) - 1):
            subset = np.array([a for a in data if years[i] <= a[0] < years[i+1]])
            if subset.shape[0] == 0:
                continue
            subset = subset[subset[:, 2] == source]
            if subset.shape[0] == 0:
                continue

            sentiment_subset = np.array(subset[:, 1], dtype=np.float32)
            f_out.write(str(source) + ' ' + str(years[i].year) + ': ' + str(sentiment_subset.shape[0]) +
                        ' articles with mean sentiment ' + str(np.average(sentiment_subset)) +
                        ' and std. dev. ' + str(np.std(sentiment_subset)) + '\n')

    f_out.write('\n')

    # Print Keyword usage for each year and each source
    for i in range(len(years)-1):
        data_in_year = [a[3] for a in data if years[i] <= a[0] < years[i+1]]
        words_in_year = defaultdict(int)
        for j in data_in_year:
            for k, v in keywords[j].items():
                words_in_year[basic_stemmer(k)] += v
        f_out.write('Keywords used in ' + str(years[i].year) + ':\n')
        for word, occurrence in words_in_year.items():
            f_out.write('\t' + word + ': ' + str(occurrence) + ' occurrences (' +
                        str(occurrence / len(data_in_year)) + ' per article)\n')

    for source in sources:
        data_in_source = list(data[data[:, 2] == source][:, 3])
        words_in_source = defaultdict(int)
        for j in data_in_source:
            for k, v in keywords[j].items():
                words_in_source[basic_stemmer(k)] += v
        f_out.write('Keywords used in ' + source + ':\n')
        for word, occurrence in words_in_source.items():
            f_out.write('\t' + word + ': ' + str(occurrence) + ' occurrences (' +
                        str(occurrence / len(data_in_source)) + ' per article)\n')

    f_out.write('\n')

    # Print keyword usage for each (year, source) combination.
    keyword_usage = {}
    keywords_stemmed = []
    for source in sources:
        for i in range(len(years) - 1):
            subset = np.array([a for a in data if years[i] <= a[0] < years[i+1]])
            if subset.shape[0] == 0:
                continue
            subset = subset[subset[:, 2] == source]
            if subset.shape[0] == 0:
                continue

            f_out.write(str(source) + ' ' + str(years[i].year) + ':\n')

            keywords_subset = subset[:, 3]
            words_in_subset = defaultdict(int)
            for j in keywords_subset:
                for k, v in keywords[j].items():
                    words_in_subset[basic_stemmer(k)] += v
            for word, occurrence in words_in_subset.items():
                per_article = occurrence / len(keywords_subset)
                f_out.write('\t' + word + ': ' + str(occurrence) + ' occurrences (' +
                            str(per_article) + ' per article)\n')

                if word not in keywords_stemmed:
                    keywords_stemmed.append(word)
                if len(keywords_subset) > 10:  # only plot keyword occurrence if sample size for that year > 10
                    if (word, str(source)) in keyword_usage:
                        keyword_usage[(word, str(source))][0].append(years[i].year)
                        keyword_usage[(word, str(source))][1].append(per_article)
                    else:
                        keyword_usage[(word, str(source))] = ([years[i].year], [per_article])

    f_out.close()

    # occurrences per article for each word (y) over time (x), all/DE/DM/Guardian (4 plots)
    fig, axs = plt.subplots(len(keywords_stemmed), 1, figsize=(10, len(keywords_stemmed) * 5), tight_layout=True)
    for i in range(len(keywords_stemmed)):
        word = keywords_stemmed[i]
        ax = axs[i]
        for src in sources:
            if (word, src) in keyword_usage:
                x, y = keyword_usage[(word, src)]
                ax.plot(x, y, label=src)
        ax.set_title(word)
        ax.legend()
    plt.savefig(out_folder + keyword + '2.png')
    plt.close()


if __name__ == '__main__':
    plot(sys.argv[1])
