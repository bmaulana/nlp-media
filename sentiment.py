import sys
import json
import os
import time
from vaderSentiment import vaderSentiment
import xiaohan_sentiment
from senti_classifier import senti_classifier
from openai_encoder import Model
from stanfordcorenlp import StanfordCoreNLP
from textblob import TextBlob
from textblob.en.sentiments import NaiveBayesAnalyzer, PatternAnalyzer
# TODO for eval - split into two files? (so no need to load the sentiment models to run eval_sentiment)
import random
import numpy as np
import matplotlib.pyplot as plt

openai_time = time.time()
openai_model = Model()
print('\nLoading OpenAI sentiment model took', time.time() - openai_time, 'seconds\n')

vader_analyser = vaderSentiment.SentimentIntensityAnalyzer()
stanford_nlp = StanfordCoreNLP('http://localhost', 9000)
nba = NaiveBayesAnalyzer()
pa = PatternAnalyzer()


def sentiment(fname, test_time=False):
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
    if not test_time:
        f_out = open(out_path, 'w')
    else:
        f_out = open('./out-sentiment/times.csv', 'a')

    for line in f_in:
        article_time = time.time()
        scorer_times = []

        to_write = json.loads(line)
        if to_write['num_relevant_sentences'] == 0:
            count += 1
            print(count, "/", total, "articles analysed (last article skipped)")
            continue

        # TODO test relevance scores to see which performs better (sample n articles from each)
        to_write['relevance_score_sents'] = to_write['num_relevant_sentences'] / to_write['num_sentences']
        to_write['relevance_score_keyword_rank'] = 1 / to_write['keyword_rank']  # Assume Zipf distribution with s ~= 1

        # TODO test sentiment scores to see which performs better (sample n articles from each)
        sents = to_write['relevant_sentences']

        batched_sents = []  # OpenAI runs faster with batched sentences
        # VADER: rule-based / lexical (https://github.com/cjhutto/vaderSentiment)
        score_time = time.time()
        for sent in sents:
            sents[sent]['sentiment_score_vader'] = vader_analyser.polarity_scores(sent)['compound']  # y = -1 to 1
        # print(time.time() - score_time, 'seconds to analyse article using VADER')
        scorer_times.append(time.time() - score_time)

        # XiaoHan: Convolutional Neural Network, trained on Twitter (github.com/xiaohan2012/twitter-sent-dnn)
        score_time = time.time()
        for sent in sents:
            sents[sent]['sentiment_score_xiaohan'] = (xiaohan_sentiment.sentiment_score(sent) - 0.5) * 2  # y = 0 to 1
        # print(time.time() - score_time, 'seconds to analyse article using XiaoHan')
        scorer_times.append(time.time() - score_time)

        # K. Cobain: MaxEnt / Naive Bayes, trained on SentiWordNet (github.com/kevincobain2000/sentiment_classifier)
        # Note: slow (~2 seconds per sentence)
        score_time = time.time()
        for sent in sents:
            pos_score, neg_score = senti_classifier.polarity_scores([sent])
            sents[sent]['sentiment_score_kcobain'] = pos_score - neg_score  # y = normal distribution mean=0, unbounded
        # print(time.time() - score_time, 'seconds to analyse article using Kevin Cobain')
        scorer_times.append(time.time() - score_time)

        # TODO if after evaluation, this library is chosen (likely), optimise performance via batching everything
        # TODO (may be easier to do in a separate function)
        # OpenAI: mLSTM-based, trained on Amazon reviews (github.com/openai/generating-reviews-discovering-sentiment)
        # Note: VERY slow (~7 seconds per sentence, faster with larger batches). May be faster with tensorflow-gpu.
        for sent in sents:
            batched_sents.append(sent)  # OpenAI runs faster with batched sentences
        score_time = time.time()
        try:
            openai_sentiment = openai_model.transform(batched_sents)[:, 2388]
            for i in range(len(batched_sents)):
                sents[batched_sents[i]]['sentiment_score_openai'] = float(openai_sentiment[i])
                # y = normal distribution with mean=0, unbounded
        except ValueError:  # tensorflow bugs
            for i in range(len(batched_sents)):
                sents[batched_sents[i]]['sentiment_score_openai'] = 'ERROR'
        scorer_times.append(time.time() - score_time)

        # StanfordNLP (https://nlp.stanford.edu/sentiment/). Classifier instead of regressor (y = [1,2,3,4,5])
        # Can be batched, but may not split sentences consistently at the same points as SpaCy (and runs quick anw)
        score_time = time.time()
        for sent in sents:
            props = {'annotators': 'sentiment', 'outputFormat': 'json', 'timeout': 15000}
            res = json.loads(stanford_nlp.annotate(sent, properties=props))
            sents[sent]['sentiment_score_stanford'] = (float(res['sentences'][0]['sentimentValue']) - 2.0) / 2.0
        # print(time.time() - score_time, 'seconds to analyse article using StanfordNLP')
        scorer_times.append(time.time() - score_time)

        # TextBlob. Uses Naive Bayes analyzer trained on the movie review corpus (textblob.readthedocs.io/en/dev/)
        # Can be batched, but may not split sentences consistently at the same points as SpaCy (and runs quick anw)
        score_time = time.time()
        for sent in sents:
            res = nba.analyze(sent)
            sents[sent]['sentiment_score_textblob_bayes'] = res[1] - res[2]
        # print(time.time() - score_time, 'seconds to analyse article using TextBlob (Naive Bayes)')
        scorer_times.append(time.time() - score_time)

        # TextBlob. Uses pattern analyser (https://textblob.readthedocs.io/en/dev/)
        # Can be batched, but may not split sentences consistently at the same points as SpaCy (and runs quick anw)
        score_time = time.time()
        for sent in sents:
            sents[sent]['sentiment_score_textblob'] = pa.analyze(sent).polarity
        # print(time.time() - score_time, 'seconds to analyse article using TextBlob (Pattern Analyser)')
        scorer_times.append(time.time() - score_time)

        # 'summarise' sentiment score of an article via weighted average of each sentence
        # TODO measure relevance score of each sentence & use it as weight for sentiment score? instead of keyword_count
        sentiment_score_labels = ['vader', 'xiaohan', 'kcobain', 'openai', 'stanford', 'textblob', 'textblob_bayes']
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
                to_write[full_label] = 'ERROR'

        # Used to only analyse one article per topic/source to test performance of each scorer, comment out otherwise
        if test_time:
            f_out.write('\n')
            f_out.write(",".join([str(t) for t in scorer_times]))
            break
        else:
            f_out.write(json.dumps(to_write))
            f_out.write('\n')

        count += 1
        print(count, "/", total, "articles analysed (last article =", time.time() - article_time, "seconds)")

    f_in.close()
    f_out.close()

    full_time = time.time() - start_time
    print('Sentiment analyses took', int(full_time // 60), 'minutes', full_time % 60, 'seconds')


# TODO after evaluation of which scorer, write function that only calculates one sentiment score to improve performance
def sentiment2(fname):
    pass


def eval_sentiment(num_samples=5):
    # for each label, print the mean score of sentences I manually class positive, neutral, or negative
    # issue: bias in my classification. Mention in report, further work could involve having multiple reviewers
    labels = ['vader', 'xiaohan', 'kcobain', 'openai', 'stanford', 'textblob', 'textblob_bayes']
    mean_scores = np.array([[0.0, 0.0, 0.0]] * len(labels))
    all_scores = [([], [], []) for i in range(len(labels))]
    answers = []  # to save answers so results are reproducible
    num_sents = 0

    # From each file in ./out-sentiment/, show a sample of num_sents sentences to manually classify
    for filename in os.listdir('./out-sentiment'):
        f_in = open('./out-sentiment/' + filename, 'r')
        sents = {}
        for line in f_in:
            raw = json.loads(line)
            for sent, scores in raw['relevant_sentences'].items():
                if scores['sentiment_score_openai'] != 'ERROR':  # if one score is 'ERROR', all is 'ERROR'
                    sents[sent] = scores
        f_in.close()

        if len(sents) > num_samples:
            sampled = random.Random(0).sample(list(sents.items()), k=num_samples)
        else:
            sampled = list(sents.items())

        for sent, scores in sampled:
            res = input(sent + '\n')
            print(scores)
            print()
            if res == '+':
                for i in range(len(labels)):
                    full_label = 'sentiment_score_' + labels[i]
                    mean_scores[i, 0] += float(scores[full_label])
                    all_scores[i][0].append(float(scores[full_label]))
                    answers.append('+')
            elif res == '-':
                for i in range(len(labels)):
                    full_label = 'sentiment_score_' + labels[i]
                    mean_scores[i, 2] += float(scores[full_label])
                    all_scores[i][2].append(float(scores[full_label]))
                    answers.append('-')
            else:
                for i in range(len(labels)):
                    full_label = 'sentiment_score_' + labels[i]
                    mean_scores[i, 1] += float(scores[full_label])
                    all_scores[i][1].append(float(scores[full_label]))
                    answers.append('n')
            num_sents += 1

    # Print mean scores for each classification, for each sentiment scorer
    mean_scores /= num_sents
    for i in range(len(labels)):
        print(labels[i], '\tPositive:', mean_scores[i][0], '\tNeutral:', mean_scores[i][1],
              '\tNegative:', mean_scores[i][2])

    # Plot histogram for each classification, for each sentiment scorer
    fig, axs = plt.subplots(2, (len(labels) + 1) // 2, figsize=(5 * ((len(labels) + 1) // 2), 10), tight_layout=True)
    for i in range(len(labels)):
        ax = axs[i % 2, i // 2]
        ax.hist(all_scores[i][0], bins=np.arange(-1.0, 1.1, 0.1), alpha=0.5, label='pos')
        ax.hist(all_scores[i][2], bins=np.arange(-1.0, 1.1, 0.1), alpha=0.5, label='neg')
        ax.hist(all_scores[i][1], bins=np.arange(-1.0, 1.1, 0.1), alpha=0.5, label='ntr')
        ax.set_title(labels[i])
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('No. of articles')
        ax.legend()
    plt.savefig('./out-plot/sentiment_eval.png')
    plt.close()

    print(answers)


if __name__ == '__main__':
    if sys.argv[1] == 'eval':
        if len(sys.argv) > 2:
            eval_sentiment(int(sys.argv[2]))
        eval_sentiment()
    else:
        sentiment(sys.argv[1])
