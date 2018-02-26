import sys
import json
import spacy
import time
import os
from spacy.matcher import Matcher

start_time = time.time()
nlp = spacy.load('en_core_web_lg')
print('Loading SpaCy model took', time.time() - start_time, 'seconds')


def parse(fname, keywords=None):
    """
    Format: python parse.py filename [keywords(comma-delimited)]
    """

    if not os.path.exists('./out-parse/'):
        os.makedirs('./out-parse/')
    in_path = './out-filtered/' + fname
    out_path = './out-parse/' + fname
    topic = fname.replace('-', '.').split('.')[1]

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

    f_in = open(in_path, 'r')
    f_out = open(out_path, 'w')
    for line in f_in:
        to_write = {}

        js = json.loads(line)
        to_write['url'] = list(js.keys())[0]

        text = list(js.values())[0]['text']
        title = list(js.values())[0]['title']
        doc = nlp(title + '. ' + text)
        matches = matcher(doc)

        # to_write['text'] = text
        to_write['title'] = title
        to_write['datetime'] = list(js.values())[0]['datetime']
        to_write['source'] = list(js.values())[0]['source']

        to_write['keyword_rank'] = js['keyword_rank']
        to_write['keyword_count'] = js['keyword_count']  # could also use len(matches) for consistency (spacy over nltk)
        to_write['num_tokens'] = js['num_tokens']  # total no. of tokens, excluding stop words; could also use spacy

        to_write['num_sentences'] = len([0 for sent in doc.sents])  # but this looks like a better relevance metric anw

        match_vectors = []
        sents = set()
        for match_id, start, end in matches:
            token = doc[start]
            this_match = {'text': token.text, 'start': start}

            span = doc[start: end]  # matched span
            sent = span.sent  # sentence containing matched span
            if sent.text not in sents:
                this_match['Sentence'] = sent.text  # TODO do some other sentiment analysis on this sentence
                sents.add(sent.text)
            else:
                this_match['Sentence'] = 'seen'

            # TODO instead of just using direct parent/child, use their parent/child as well,
            # but decrease weight of sentiment depending on how 'far' the word is
            if token.dep_ != 'ROOT':
                this_match['Parent'] = {'text': token.head.text, 'sentiment': token.head.sentiment}

            child_vectors = []
            for child in token.children:
                child_vectors.append({'text': child.text, 'sentiment': child.sentiment})
                # TODO SpaCy's sentiment analyser isn't good enough.
                # e.g. words with obvious sentiment (e.g. issues, disorder) has sentiment of 0.0
                # Implement some other sentiment data set and pass it the token's text instead.
            this_match['Children'] = child_vectors

            match_vectors.append(this_match)

        to_write['relevant_sentences'] = len(sents)
        to_write['matches'] = match_vectors
        f_out.write(json.dumps(to_write))
        f_out.write('\n')

    f_in.close()
    f_out.close()


if __name__ == '__main__':
    if len(sys.argv) > 2:
        keys = sys.argv[2].split(',')
        parse(sys.argv[1], keys)
    else:
        parse(sys.argv[1])
