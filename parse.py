import sys
import json
import spacy
import time
from spacy.matcher import Matcher

start_time = time.time()
nlp = spacy.load('en_core_web_lg')
print('Loading SpaCy model took', time.time() - start_time, 'seconds')


def parse(fname, keywords=None):
    """
    Format: python parse.py filename(without .json)
    """

    path = './out-filtered/' + fname + '.json'

    topic = fname.split('-', 1)[1]

    if keywords is None:
        keywords = [topic]

    pattern = []
    seen = set()
    for keyword in keywords:
        stem = nlp(keyword)[0].lemma_
        if stem not in seen:
            pattern.append([{'LEMMA': stem}])
            seen.add(stem)
    print(pattern)

    matcher = Matcher(nlp.vocab)
    matcher.add(0, None, *pattern)

    f_in = open(path, 'r')
    for line in f_in:
        js = json.loads(line)
        print(list(js.keys())[0])
        text = list(js.values())[0]['text']
        doc = nlp(text)
        matches = matcher(doc)

        for match_id, start, end in matches:
            token = doc[start]
            print(token.text, start)
            # TODO instead of just using direct parent/child, use their parent/child as well,
            # but decrease weight of sentiment depending on how 'far' the word is
            if token.dep_ != 'ROOT':
                print('Parent', token.head.text, token.head.sentiment)
            for child in token.children:
                print('Child', child.text, child.sentiment)
                # TODO SpaCy's sentiment analyser isn't good enough.
                # e.g. words with obvious sentiment (e.g. issues, disorder) has sentiment of 0.0
                # Implement some other sentiment data set and pass it the token's text instead.
            print()

            # TODO get whole sentence containing token, and feed sentence to some sentiment analyser like Microsoft's

        # TODO print output to file, and structure it better (e.g. use a feature->value dictionary)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        keys = sys.argv[2].split(',')
        parse(sys.argv[1], keys)
    else:
        parse(sys.argv[1])
