import sys
import json
import spacy
import time
import os
import re
from spacy.matcher import Matcher
from tqdm import tqdm

# Usage: python parse.py filename [keywords(comma-delimited)]

start_time = time.time()
nlp = spacy.load('en_core_web_lg')
print('Loading SpaCy model took', time.time() - start_time, 'seconds\n')


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
    keywords.append('suffers from')
    # for keyword in [i for i in keywords]:
    #     keywords.append('person with ' + keyword)

    pattern = []
    seen = set()
    for keyword in keywords:
        stem = [nlp(word)[0].lemma_ for word in re.split('[ -]', keyword)]
        if ' '.join(stem) not in seen:
            p = []
            for word in stem:
                p.append({'LEMMA': word})
            seen.add(' '.join(stem))
            pattern.append(p)
    print(pattern)

    matcher = Matcher(nlp.vocab)
    matcher.add(0, None, *pattern)

    f_in = open(in_path, 'r')
    f_out = open(out_path, 'w')
    for line in tqdm(f_in):
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

        # match_vectors = []
        sents = {}
        keywords_used = {}
        for match_id, start, end in matches:
            # token = doc[start]
            # this_match = {'text': token.text, 'start': start}

            span = doc[start: end]  # matched span
            if span.text.lower() in keywords_used:
                keywords_used[span.text.lower()] += 1
            else:
                keywords_used[span.text.lower()] = 1

            sent = span.sent  # sentence containing matched span
            if sent.text not in sents:
                # this_match['Sentence'] = sent.text
                sents[sent.text] = {'keyword_count': 1}
            else:
                # this_match['Sentence'] = 'seen'
                sents[sent.text]['keyword_count'] += 1

            '''
            # instead of just using direct parent/child, get all words that refer/modify the keyword somehow?
            # Not used in current implementation of sentiment.py, thus low priority.
            if token.dep_ != 'ROOT':
                this_match['Parent'] = {'text': token.head.text, 'sentiment': token.head.sentiment}
            child_vectors = []
            for child in token.children:
                child_vectors.append({'text': child.text, 'sentiment': child.sentiment})
            this_match['Children'] = child_vectors
            '''

            # match_vectors.append(this_match)

        to_write['num_relevant_sentences'] = len(sents)
        to_write['relevant_sentences'] = sents
        to_write['keywords_used'] = keywords_used

        if len(sents) == 0:
            continue

        # to_write['matches'] = match_vectors

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
