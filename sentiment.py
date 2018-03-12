import sys
import json
import os


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
        to_write['relevance_score_keyword_rank_stopwords'] = 1 / to_write['keyword_rank_include_stop_words']
        # test all and see which performs better (sample n high/low-relevance articles from each)

        # TODO improve sentiment score
        sents = to_write['relevant_sentences']
        for sent in sents:
            sents[sent]['sentiment_score'] = 0.0  # TODO

        del to_write['matches']  # for testing (reduce output file size so easier to read)

        f_out.write(json.dumps(to_write))
        f_out.write('\n')

    f_in.close()
    f_out.close()


if __name__ == '__main__':
    sentiment(sys.argv[1])
