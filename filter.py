import json
import sys
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer  # TODO test which is better between this and cyhunspell
# from sklearn.feature_extraction.text import TfidfTransformer

KEYWORD_TOKEN = 'KEYWORDTOKEN'  # something that shouldn't naturally occur in a document


def filter(fname, keywords=None):
    """
    Format: python filter.py filename(without extension) [keywords(comma-delimited)]
    """

    # TODO add filename to parameter (easier to use with crawler in a script)
    if not os.path.exists('./out-filtered/'):
        os.makedirs('./out-filtered/')
    in_path = './out/' + fname + '.json'
    out_path = './out-filtered/' + fname + '.json'
    topic = fname.split('-', 1)[1]

    f_in = open(in_path, 'r')
    in_lines, out_lines = 0, 0

    # TODO read keywords from file: first line is list of valid keywords for the topic.
    # TODO does not work with >1-gram keywords (e.g. "Down's Syndrome")

    stemmer = SnowballStemmer('english')
    if keywords is None:
        keywords = [topic]
    keywords = set([stemmer.stem(word) for word in keywords])
    print(keywords)

    data, corpus = [], []  # TODO will this fit in memory with millions of articles?
    for line in f_in:
        js = json.loads(line)
        data.append(js)

        text = list(js.values())[0]['text'].split()
        for i in range(len(text)):
            word = stemmer.stem(text[i])
            if word in keywords:
                word = KEYWORD_TOKEN  # change all words that match any keyword to one token for CountVectoriser
            text[i] = word
        text = ' '.join(text)
        corpus.append(text)

        in_lines += 1
    f_in.close()
    print(corpus[0])  # to test stemmer

    vectoriser = CountVectorizer(stop_words='english')
    matrix = vectoriser.fit_transform(corpus)
    # print(matrix[:2])

    '''
    transformer = TfidfTransformer(smooth_idf=False)
    tf_idf = transformer.fit_transform(matrix)
    # print(tf_idf[:2])
    '''

    analyse = vectoriser.build_analyzer()
    keyword_index = vectoriser.vocabulary_.get(analyse(KEYWORD_TOKEN)[0])
    # keyword_array = matrix[:, keyword_index].toarray().reshape([1, len(corpus)])[0]
    # print(sorted(keyword_array)[::-1])
    # print(sum(keyword_array) / len(keyword_array))

    relevant = []
    f_out = open(out_path, 'w')
    for i in range(len(corpus)):
        vectors = matrix[i].toarray()
        keyword_vector = vectors[0][keyword_index]
        vectors.sort()
        vectors = np.fliplr(vectors)

        _, rank = np.where(vectors <= keyword_vector)
        rank = rank[0] + 1  # occurrence rank of the keyword(s), relative to other non-stop words in the article

        if rank <= 20:  # rank threshold
            relevant.append((True, rank))
            data[i]['keyword_count'] = int(keyword_vector)
            data[i]['keyword_rank'] = int(rank)
            f_out.write(json.dumps(data[i]))
            f_out.write('\n')
            out_lines += 1
        else:
            relevant.append((False, rank))
    f_out.close()

    print(relevant[:10])  # Used to sample articles to determine rank threshold.
    print('Input size =', in_lines, 'Output size =', out_lines)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        keys = sys.argv[2].split(',')
        filter(sys.argv[1], keys)
    else:
        filter(sys.argv[1])
