import time
import oop_scraper
from crawler import crawler
from filter import filter
from parse import parse
from sentiment import sentiment

# topic: [keywords]  TODO add more
TOPICS = {'Dyslexia': ['Dyslexia, Dyslexic'],
          'Autism': ['Autism', 'Autistic', 'Asperger\'s', 'ASD'],
          'Dementia': ['Dementia', 'Alzheimer\'s']}

SOURCES = [oop_scraper.DailyExpressScraper(),
           oop_scraper.DailyMailScraper(),
           oop_scraper.GuardianScraper("7f2c7c42-2600-4292-a417-1b8efc5271a6")]

NUM_DOCS = -1


def pipeline(topic, keywords, source):
    start_time = time.time()

    fname = crawler(topic, source, NUM_DOCS, keywords)
    # fname = 'Guardian-Dyslexia.json'

    filter(fname, keywords)
    parse(fname, keywords)
    sentiment(fname)
    # TODO graph

    total_time = time.time() - start_time
    print('\nPipeline took', int(total_time // 60), 'minutes', total_time % 60, 'seconds\n')


'''
for src in SOURCES:
    for tpc, words in TOPICS:
        pipeline(tpc, words, src)
'''

pipeline('Dyscalculia', ['Dyscalculia'], SOURCES[2])
