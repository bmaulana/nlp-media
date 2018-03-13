import sys
import json
import os
from vaderSentiment import vaderSentiment
import xiaohan_sentiment
from senti_classifier import senti_classifier


def sentiment(fname):
    """
    Format: python sentiment.py filename [keywords(comma-delimited)]
    """

    if not os.path.exists('./out-sentiment/'):
        os.makedirs('./out-sentiment/')
    in_path = './out-parse/' + fname
    out_path = './out-sentiment/' + fname

    f_in = open(in_path, 'r')
    f_out = open(out_path, 'w')

    for line in f_in:
        to_write = json.loads(line)

        # TODO improve/test relevance score
        to_write['relevance_score_sents'] = to_write['num_relevant_sentences'] / to_write['num_sentences']
        to_write['relevance_score_keyword_rank'] = 1 / to_write['keyword_rank']  # Assume Zipf distribution with s ~= 1
        # test all and see which performs better (sample n high/low-relevance articles from each)

        # TODO improve sentiment score - add more scores/libraries/models, test
        sents = to_write['relevant_sentences']
        for sent in sents:
            # TODO normalise all y-values to 0 to 1 or -1 to 1 for easier comparison
            # VADER: rule-based / lexical (https://github.com/cjhutto/vaderSentiment)
            vader_analyser = vaderSentiment.SentimentIntensityAnalyzer()
            sents[sent]['sentiment_score_vader'] = vader_analyser.polarity_scores(sent)['compound']  # y = -1 to 1

            # XiaoHan: Convolutional Neural Network, trained on Twitter (github.com/xiaohan2012/twitter-sent-dnn)
            sents[sent]['sentiment_score_xiaohan'] = xiaohan_sentiment.sentiment_score(sent)  # y = 0 to 1

            # Kevin Cobain: Maximum Entropy, Naive Bayes, SentiWordNet (github.com/kevincobain2000/sentiment_classifier)
            # Note: REALLY slow (few minutes per article).
            pos_score, neg_score = senti_classifier.polarity_scores([sent])
            sents[sent]['sentiment_score_kcobain'] = pos_score - neg_score

            # OpenAI: mLSTM-based, trained on IMDB reviews (github.com/openai/generating-reviews-discovering-sentiment)
            # TODO need TensorFlow installed

        del to_write['matches']  # for testing (make output file easier to read)

        f_out.write(json.dumps(to_write))
        f_out.write('\n')

    f_in.close()
    f_out.close()


if __name__ == '__main__':
    sentiment(sys.argv[1])
