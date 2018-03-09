import oop_scraper
from crawler import crawler
from filter import filter
from parse import parse

# topic: [keywords]  TODO add more
TOPICS = {'Dyslexia': ['Dyslexia, Dyslexic'],
          'Autism': ['Autism', 'Autistic', 'Asperger\'s', 'ASD'],
          'Dementia': ['Dementia', 'Alzheimer\'s']}

SOURCES = [oop_scraper.DailyExpressScraper(),
           oop_scraper.DailyMailScraper(),
           oop_scraper.GuardianScraper("7f2c7c42-2600-4292-a417-1b8efc5271a6")]

NUM_DOCS = -1


def pipeline(topic, keywords, source):
    fname = crawler(topic, source, NUM_DOCS, keywords)
    filter(fname, keywords)
    parse(fname, keywords)
    # TODO graph


'''
for src in SOURCES:
    for tpc, words in TOPICS:
        pipeline(tpc, words, src)
'''

pipeline('Dyslexia', ['Dyslexia', 'Dyslexic'], SOURCES[2])
