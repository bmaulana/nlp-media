import json
import sys
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer


def main(fname):
    """
    Format: python filter.py filename(without extension)
    """

    if not os.path.exists('./out-filtered/'):
        os.makedirs('./out-filtered/')
    in_path = './out/' + fname + '.json'
    out_path = './out-filtered/' + fname + '.json'
    topic = fname.split('-', 1)[1]

    f_in = open(in_path, 'r')
    in_lines, out_lines = 0, 0

    data, corpus = [], []  # TODO will this fit in memory with millions of articles?
    # TODO first line is list of valid keywords for the topic
    for line in f_in:
        js = json.loads(line)
        data.append(js)
        corpus.append(list(js.values())[0]['text'])
        in_lines += 1
    f_in.close()

    vectoriser = CountVectorizer(stop_words='english')
    matrix = vectoriser.fit_transform(corpus)
    # print(matrix[:2])

    '''
    transformer = TfidfTransformer(smooth_idf=False)
    tf_idf = transformer.fit_transform(matrix)
    # print(tf_idf[:2])
    '''

    # TODO does not work with >1-gram keywords (e.g. "Down's Syndrome")
    # TODO enable using multiple keywords per topic (e.g. 'Autism' and 'Autistic', 'Dyslexia' and 'Dyslexic')
    analyse = vectoriser.build_analyzer()
    keyword_index = vectoriser.vocabulary_.get(analyse(topic)[0])  # TODO with multiple keywords, save a list
    # keyword_array = matrix[:, keyword_index].toarray().reshape([1, len(corpus)])[0]
    # print(sorted(keyword_array)[::-1])
    # print(sum(keyword_array) / len(keyword_array))

    relevant = []
    f_out = open(out_path, 'w')
    for i in range(len(corpus)):
        vectors = matrix[i].toarray()
        keyword_vector = vectors[0][keyword_index]  # TODO with multiple keywords, sum them
        vectors.sort()
        vectors = np.fliplr(vectors)

        _, rank = np.where(vectors <= keyword_vector)
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
    # TODO currently filters too many relevant articles. Need to enable multiple keywords per topic.


if __name__ == '__main__':
    main(sys.argv[1])
