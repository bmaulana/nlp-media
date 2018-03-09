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
    # TODO does not work with >1-gram keywords (e.g. "Down's Syndrome")
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
        sents = {}
        for match_id, start, end in matches:
            token = doc[start]
            this_match = {'text': token.text, 'start': start}

            span = doc[start: end]  # matched span
            sent = span.sent  # sentence containing matched span
            if sent.text not in sents:
                # this_match['Sentence'] = sent.text
                sents[sent.text] = 1
            else:
                # this_match['Sentence'] = 'seen'
                sents[sent.text] += 1

            # TODO instead of just using direct parent/child, get all words that refer/modify the keyword somehow?
            if token.dep_ != 'ROOT':
                this_match['Parent'] = {'text': token.head.text, 'sentiment': token.head.sentiment}
            child_vectors = []
            for child in token.children:
                child_vectors.append({'text': child.text, 'sentiment': child.sentiment})
            this_match['Children'] = child_vectors

            match_vectors.append(this_match)

        to_write['num_relevant_sentences'] = len(sents)
        to_write['relevant_sentences'] = sents
        to_write['matches'] = match_vectors

        # TODO for each sentence:
        # TODO add 'sentiment score' from ~5 different sentiment models (e.g. NLTK) to compare - pass as parameter?
        # TODO add 'relevance score' (use my model from Data Mining CW? - ~early April)
        # TODO calculate sentiment and relevance score of article by weighted average (sentiment) / sum (relevance)

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
