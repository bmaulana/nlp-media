import os
from collections import defaultdict
from tqdm import tqdm


def howmany(in_folder='./out/', out_folder='./out-eval/'):
    """
    Simple script to find out how many articles I have in my dataset
    """
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    total = 0
    per_source = defaultdict(int)
    per_topic = defaultdict(int)
    per_st = defaultdict(int)

    for filename in tqdm(os.listdir(in_folder)):
        st = filename.split('.')[0].split('-')
        source, topic = st[0], st[1]

        f_in = open(in_folder + filename, 'r')
        for line in f_in:
            total += 1
            per_topic[topic] += 1
            per_source[source] += 1
            per_st[(source, topic)] += 1
        f_in.close()

    # TODO output to file

    print(per_st)
    print(per_topic)
    print(per_source)
    print(total)


if __name__ == '__main__':
    howmany()
