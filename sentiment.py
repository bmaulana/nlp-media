import sys
import json
import os
import time
from vaderSentiment import vaderSentiment
import xiaohan_sentiment
from senti_classifier import senti_classifier
from openai_encoder import Model

openai_time = time.time()
openai_model = Model()
print('\nLoading OpenAI sentiment model took', time.time() - openai_time, 'seconds\n')


def sentiment(fname):
    """
    Format: python sentiment.py filename
    """
    start_time = time.time()

    if not os.path.exists('./out-sentiment/'):
        os.makedirs('./out-sentiment/')
    in_path = './out-parse/' + fname
    out_path = './out-sentiment/' + fname

    f_in = open(in_path, 'r')
    count, total = 0, sum(1 for line in f_in)
    f_in.close()

    f_in = open(in_path, 'r')
    f_out = open(out_path, 'w')
    for line in f_in:
        article_time = time.time()
        to_write = json.loads(line)

        # TODO test relevance score to see which performs better (sample n high/low-relevance articles from each)
        to_write['relevance_score_sents'] = to_write['num_relevant_sentences'] / to_write['num_sentences']
        to_write['relevance_score_keyword_rank'] = 1 / to_write['keyword_rank']  # Assume Zipf distribution with s ~= 1

        # TODO improve sentiment score - add more scores/libraries/models, test
        # TODO normalise all y-values to 0 to 1 or -1 to 1 for easier comparison
        sents = to_write['relevant_sentences']
        sents_array = []  # OpenAI runs faster with batched sentences
        for sent in sents:
            # VADER: rule-based / lexical (https://github.com/cjhutto/vaderSentiment)
            vader_analyser = vaderSentiment.SentimentIntensityAnalyzer()
            sents[sent]['sentiment_score_vader'] = vader_analyser.polarity_scores(sent)['compound']  # y = -1 to 1

            # XiaoHan: Convolutional Neural Network, trained on Twitter (github.com/xiaohan2012/twitter-sent-dnn)
            sents[sent]['sentiment_score_xiaohan'] = (xiaohan_sentiment.sentiment_score(sent) - 0.5) * 2  # y = 0 to 1

            # K. Cobain: MaxEnt / Naive Bayes, trained on SentiWordNet (github.com/kevincobain2000/sentiment_classifier)
            # Note: slow (~2 seconds per sentence)
            # time_kcobain = time.time()
            pos_score, neg_score = senti_classifier.polarity_scores([sent])
            sents[sent]['sentiment_score_kcobain'] = pos_score - neg_score  # y = normal distribution mean=0, unbounded
            # print(time.time() - time_kcobain, 'seconds to analyse a sentence using Kevin Cobain')

            sents_array.append(sent)  # OpenAI runs faster with batched sentences

        # OpenAI: mLSTM-based, trained on IMDB reviews (github.com/openai/generating-reviews-discovering-sentiment)
        # Note: VERY slow (~7 seconds per sentence, faster with larger batches). May be faster with tensorflow-gpu.
        try:
            openai_sentiment = openai_model.transform(sents_array)[:, 2388]
            for i in range(len(sents_array)):
                sents[sents_array[i]]['sentiment_score_openai'] = float(openai_sentiment[i])
                # y = normal distribution with mean=0, unbounded
        except ValueError:  # tensorflow bugs
            for i in range(len(sents_array)):
                sents[sents_array[i]]['sentiment_score_openai'] = 'ERROR'

        del to_write['matches']  # for testing (make output file easier to read)

        # 'summarise' sentiment score of an article via weighted average of each sentence
        # TODO measure relevance score of each sentence & use it as weight, instead of keyword_count?
        sentiment_score_labels = ['vader', 'xiaohan', 'kcobain', 'openai']
        for label in sentiment_score_labels:
            full_label = 'sentiment_score_' + label
            weighted_avg, keyword_max_count = 0.0, 0
            for sent in sents:
                if sents[sent][full_label] != 'ERROR':
                    weighted_avg += sents[sent][full_label] * sents[sent]['keyword_count']
                    keyword_max_count += 1
            try:
                to_write[full_label] = weighted_avg / keyword_max_count
            except ZeroDivisionError:
                # TODO these articles should be filtered out by filter.py... need to test why this isn't the case
                to_write[full_label] = 'ERROR'

        f_out.write(json.dumps(to_write))
        f_out.write('\n')

        count += 1
        print(count, "/", total, "articles analysed (last article =", time.time() - article_time, "seconds)")

    f_in.close()
    f_out.close()

    full_time = time.time() - start_time
    print('Sentiment analyses took', int(full_time // 60), 'minutes', full_time % 60, 'seconds')


if __name__ == '__main__':
    sentiment(sys.argv[1])
