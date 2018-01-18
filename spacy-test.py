import spacy
import time

start = time.time()
nlp = spacy.load('en_core_web_lg')

print('Loading SpaCy model took', time.time() - start, 'seconds')

doc = nlp(u'Autonomous cars shift insurance liability toward manufacturers')
spacy.displacy.serve(doc, style='dep')
# now open 'localhost:5000' in your web browser

'''
TODO parse all documents from output file,
print all sentences with keyword,
parse these sentences (POS tag, head, child(ren) for each token; noun chunks)
find (nested) children of the keyword (or if multiple words, root of key noun phrase)
print all entities (see if common disability names are recognised as entities)

Can use Matcher to find specific tokens (i.e. the keyword0 in a document
e.g. Matcher [{'LOWER': 'hello'}, {'IS_PUNCT': True}, {'LOWER': 'world'}]
matches '[hello in any case] [punctuation] [world in any case]' in that sequence
returns a list of (match_id, start, end) tuples;
One matcher can match multiple patterns (e.g. 'Alzhemier's or Dementia')
'''