import json
import sys
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer  # TODO test which is better between this and cyhunspell
# from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm

KEYWORD_TOKEN = 'KEYWORDTOKEN'  # something that shouldn't naturally occur in a document


def filter(fname, keywords=None, threshold=20):
    """
    Format: python filter.py filename [keywords(comma-delimited)]
    """

    if not os.path.exists('./out-filtered/'):
        os.makedirs('./out-filtered/')
    in_path = './out/' + fname
    out_path = './out-filtered/' + fname
    topic = fname.replace('-', '.').split('.')[1]

    f_in = open(in_path, 'r')
    in_lines, out_lines = 0, 0

    stemmer = SnowballStemmer('english')
    if keywords is None:
        keywords = [topic]
    keywords = set([' '.join(list(map(stemmer.stem, word.split()))) for word in keywords])
    print(keywords)

    print('Reading input from ./out/')
    data, corpus = [], []
    for line in tqdm(f_in):
        js = json.loads(line)
        data.append(js)

        # Stem text & replace up to 3-gram keywords with KEYWORD_TOKEN
        text = list(js.values())[0]['title'].split() + list(js.values())[0]['text'].split()
        for i in range(len(text)):
            word = stemmer.stem(text[i])
            if i >= 2 and text[i-2] + ' ' + text[i-1] + ' ' + word in keywords:
                text[i-2] = KEYWORD_TOKEN
                text[i-1] = ''
                word = ''
            if i >= 1 and text[i-1] + ' ' + word in keywords:
                text[i-1] = KEYWORD_TOKEN
                word = ''
            if word in keywords:
                word = KEYWORD_TOKEN  # change all words that match any keyword to one token for CountVectoriser
            text[i] = word
        text = ' '.join(text)

        corpus.append(text)
        in_lines += 1
    f_in.close()
    # print(corpus[0])  # to test stemmer

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

    print('Writing output to ./out-filtered/')
    # relevant = []
    f_out = open(out_path, 'w')
    for i in tqdm(range(len(corpus))):
        vectors = matrix[i].toarray()
        keyword_vector = vectors[0][keyword_index]
        vectors.sort()
        vectors = np.fliplr(vectors)

        _, rank = np.where(vectors <= keyword_vector)
        rank = rank[0] + 1  # occurrence rank of the keyword(s), relative to other non-stop words in the article

        if rank <= threshold:  # rank threshold
            # relevant.append((True, rank))

            # Save info on rank of keyword
            data[i]['keyword_count'] = int(keyword_vector)
            data[i]['keyword_rank'] = int(rank)
            data[i]['num_tokens'] = int(np.sum(vectors))

            # Output to file
            f_out.write(json.dumps(data[i]))
            f_out.write('\n')
            out_lines += 1

        # else:
            # relevant.append((False, rank))
    f_out.close()

    # print(relevant[:10])  # Used to sample articles to determine rank threshold.
    print('Input size =', in_lines, 'Output size =', out_lines)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        keys = sys.argv[2].split(',')
        filter(sys.argv[1], keys)
    else:
        filter(sys.argv[1])
