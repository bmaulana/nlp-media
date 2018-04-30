import sys
import json
import os
from tqdm import tqdm
from vaderSentiment import vaderSentiment

vader_analyser = vaderSentiment.SentimentIntensityAnalyzer()


def sentiment_vader(fname):
    if not os.path.exists('./out-sentiment-vader/'):
        os.makedirs('./out-sentiment-vader/')
    in_path = './out-parse/' + fname
    out_path = './out-sentiment-vader/' + fname

    f_in = open(in_path, 'r')
    f_out = open(out_path, 'w')

    for line in tqdm(f_in):
        to_write = json.loads(line)
        if to_write['num_relevant_sentences'] == 0:
            continue

        # TODO test relevance scores to see which performs better (sample n articles from each)
        to_write['relevance_score_sents'] = to_write['num_relevant_sentences'] / to_write['num_sentences']
        to_write['relevance_score_keyword_rank'] = 1 / to_write['keyword_rank']  # Assume Zipf distribution with s ~= 1

        sents = to_write['relevant_sentences']

        # VADER: rule-based / lexical (https://github.com/cjhutto/vaderSentiment)
        for sent in sents:
            sents[sent]['sentiment_score'] = vader_analyser.polarity_scores(sent)['compound']  # y = -1 to 1

        # 'summarise' sentiment score of an article via weighted average of each sentence
        # TODO measure relevance score of each sentence & use it as weight for sentiment score? instead of keyword_count
        weighted_avg, keyword_total_count = 0.0, 0
        for sent in sents:
            if sents[sent]['sentiment_score'] != 'ERROR':
                weighted_avg += sents[sent]['sentiment_score'] * sents[sent]['keyword_count']
                keyword_total_count += sents[sent]['keyword_count']
        try:
            to_write['sentiment_score'] = weighted_avg / keyword_total_count
        except ZeroDivisionError:
            to_write['sentiment_score'] = 'ERROR'

        # Used to only analyse one article per topic/source to test performance of each scorer, comment out otherwise
        f_out.write(json.dumps(to_write))
        f_out.write('\n')

    f_in.close()
    f_out.close()


if __name__ == '__main__':
    sentiment_vader(sys.argv[1])
