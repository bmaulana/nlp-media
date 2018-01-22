import sys
import json
import spacy
import time
from spacy.matcher import Matcher

start_time = time.time()
nlp = spacy.load('en_core_web_lg')
matcher = Matcher(nlp.vocab)
print('Loading SpaCy model took', time.time() - start_time, 'seconds')


def main(fname):
    """
    Format: python parse.py filename(without .json)
    """

    path = './out-filtered/' + fname + '.json'
    topic = fname.split('-', 1)[1]
    pattern = [[{'LOWER': topic.lower()}]]  # TODO one for each keyword in the topic
    matcher.add(0, None, *pattern)

    f_in = open(path, 'r')
    for line in f_in:
        js = json.loads(line)
        text = list(js.values())[0]['text']
        doc = nlp(text)
        matches = matcher(doc)

        for match_id, start, end in matches:
            token = doc[start]
            print(token.text, start)
            if token.dep_ != 'ROOT':
                print('Parent', token.head.text, token.head.sentiment)
            for child in token.children:
                print('Child', child.text, child.sentiment)
                # TODO SpaCy's sentiment analyser isn't good enough.
                # e.g. words with obvious sentiment (e.g. issues, disorder) has sentiment of 0.0
                # Implement some other sentiment analyser and pass it the token's text instead.
            print()
        print()


if __name__ == '__main__':
    main(sys.argv[1])
