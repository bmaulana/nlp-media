import time
import oop_scraper
from crawler import crawler
from filter import filter
from parse import parse
from sentiment import sentiment
from plot import plot

# topic: [keywords]
TOPICS = {'Dyslexia': ['Dyslexia', 'Dyslexic'],
          'Autism': ['Autism', 'Autistic', 'Asperger\'s', 'ASD'],
          'Dementia': ['Dementia', 'Alzheimer\'s'],
          }

TOPICS2 = {'cerebral palsy': ['cerebral palsy', 'spastic'],
           'deaf': ['deaf', 'hearing impaired', 'hard of hearing', 'hearing loss'],  # impairment and impaired
           'blind': ['blind', 'visual impairments', 'partially sighted'],  # visual and visually has the same stem
           'epilepsy': ['epilepsy', 'epileptic', 'seizure'],
           'mute': ['mute', 'cannot speak', 'difficulty speaking', 'synthetic speech', 'non-vocal', 'non-verbal'],
           # 'speech impairment': ['speech impairment', 'stutter', 'speech disability', 'speech disorder',
           #                      'communication disability', 'difficulty speaking', 'language impairment',
           #                      'language disorder', 'language disability'],
           'mental illness': ['mental illness', 'mental health', 'psychiatric', 'emotional disorder',
                              'developmental disability', 'retardation', 'developmental delay', 'brain injured',
                              'brain injury', 'brain damaged', 'learning disability', 'slow learner', 'mental issue'],
           'paralysis': ['paraplegic', 'quadriplegic', 'spinal cord', 'paraplegia', 'paralysed', 'paralyzed',
                         'paralysis', 'crippled', 'leg braces', 'wheelchair'],  # Not sure what should be the query
           'dyslexia': ['dyslexia', 'dyslexic'],
           'autism': ['autism', 'autistic', 'asperger\'s', 'ASD']
           }

# TODO add more
SOURCES = [oop_scraper.DailyExpressScraper(),
           oop_scraper.DailyMailScraper(),
           oop_scraper.GuardianScraper("7f2c7c42-2600-4292-a417-1b8efc5271a6")]

NUM_DOCS = 50  # -1 to get all documents

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


for tpc, words in TOPICS2.items():
    fnames = []
    for src in SOURCES:
        fnames.append(pipeline(tpc, words, src))
    plot(fnames)

# pipeline('Dyslexia', ['Dyslexia', 'Dyslexic'], SOURCES[0])
