import time
import oop_scraper
from crawler import crawler
from filter import filter
from parse import parse
from sentiment import sentiment
from plot import plot

# topic: [keywords]  TODO add more
TOPICS = {'Dyslexia': ['Dyslexia', 'Dyslexic'],
          'Autism': ['Autism', 'Autistic', 'Asperger\'s', 'ASD'],
          'Dementia': ['Dementia', 'Alzheimer\'s'],
          'Dyscalculia': ['Dyscalculia']
          }

# TODO add more
SOURCES = [oop_scraper.DailyExpressScraper(),
           oop_scraper.DailyMailScraper(),
           oop_scraper.GuardianScraper("7f2c7c42-2600-4292-a417-1b8efc5271a6")]

NUM_DOCS = 200  # -1 to get all documents

FILTER_THRESHOLD = 20  # higher = weaker filter


def pipeline(topic, keywords, source):
    start_time = time.time()

    fname = crawler(topic, source, NUM_DOCS, keywords)
    # fname = 'Guardian-Dyslexia.json'
    print('\n')

    filter(fname, keywords, FILTER_THRESHOLD)
    parse(fname, keywords)
    print('\n')
    sentiment(fname)

    total_time = time.time() - start_time
    print('\nPipeline took', int(total_time // 60), 'minutes', total_time % 60, 'seconds\n')
    return fname


for tpc, words in TOPICS.items():
    fnames = []
    for src in SOURCES:
        fnames.append(pipeline(tpc, words, src))
    plot(fnames)

# pipeline('Dyslexia', ['Dyslexia', 'Dyslexic'], SOURCES[0])
