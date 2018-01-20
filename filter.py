import json
import sys
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def main():
    """
    Format: python filter.py filename(without .json)
    """

    if not os.path.exists('./out-filtered/'):
        os.makedirs('./out-filtered/')
    in_path = './out/' + sys.argv[1] + '.json'
    out_path = './out-filtered/' + sys.argv[1] + '.json'
    topic = sys.argv[1].split('-', 1)[1]

    f_in = open(in_path, 'r')
    in_lines, out_lines = 0, 0

    data, corpus = [], []  # TODO will this fit in memory with millions of articles?
    for line in f_in:
        js = json.loads(line)
        data.append(js)
        corpus.append(list(js.values())[0]['text'])
        in_lines += 1
    f_in.close()

    vectoriser = CountVectorizer(stop_words='english')
    matrix = vectoriser.fit_transform(corpus)
    # print(matrix[:2])

    # TODO tf-idf causes the rank-determining code to break occasionally (maybe float equality?)
    '''
    transformer = TfidfTransformer(smooth_idf=False)
    tf_idf = transformer.fit_transform(matrix)
    # print(tf_idf[:2])
    '''

    # TODO does not work with >1-gram keywords (e.g. "Down's Syndrome")
    # TODO enable using multiple keywords per topic (e.g. 'Autism' and 'Autistic', 'Dyslexia' and 'Dyslexic')
    analyse = vectoriser.build_analyzer()
    keyword_index = matrix[:, vectoriser.vocabulary_.get(analyse(topic)[0])]\
        .toarray().reshape([1, len(corpus)])[0]
    # print(sorted(keyword_index)[::-1])
    # print(sum(keyword_index) / len(keyword_index))

    relevant = []
    f_out = open(out_path, 'w')
    for i in range(len(corpus)):
        keyword_vector = keyword_index[i]

        vectors = matrix[i].toarray()
        vectors.sort()
        vectors = np.fliplr(vectors)

        _, rank = np.where(vectors == keyword_vector)
        rank = rank[0] + 1

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
    # TODO currently filters too many relevant articles. Need to enable multiple keywords.


if __name__ == '__main__':
    main()
