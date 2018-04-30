import time
import oop_scraper
from crawler import crawler
from filter import filter
from parse import parse
# from sentiment import sentiment, sentiment_vader, sentiment_openai
from sentiment_barebones import sentiment_vader
from plot import plot

# topic: [keywords]
TOPICS = {'disabled': ['disabled', 'disability', 'handicapped', 'cripple', 'invalid', 'accessible', 'ablism',
                       'ableism', 'differently abled'],
          'cerebral palsy': ['cerebral palsy', 'spastic'],
          'deaf': ['deaf', 'deafness', 'hearing impaired', 'hard of hearing', 'hearing loss'],
          'blind': ['blind', 'blindness', 'blindism', 'visual impairment', 'partially sighted', 'vision loss'],
          'epilepsy': ['epilepsy', 'epileptic', 'seizure'],
          'mute': ['mute', 'muteness', 'mutism', 'cannot speak', 'difficulty speaking', 'synthetic speech',
                   'non-vocal', 'non-verbal'],
          'speech impairment': ['speech impairment', 'stutter', 'speech disability', 'speech disorder',
                                'communication disability', 'difficulty speaking', 'language impairment',
                                'language disorder', 'language disability', 'speech impediment', 'stammer'],
          'mental illness': ['mental illness', 'mental health', 'mental disability', 'mental disorder', 'mental issue',
                             'brain injured', 'brain injury', 'brain damaged', 'psychological', 'psychiatric',
                             'emotional disorder', 'behavioural disorder', 'intellectual disability',
                             'mentally ill', 'mentally disabled', 'mentally handicapped'],
          'developmental delay': ['developmental delay', 'developmental disability', 'developmental disorder',
                                  'learning disability', 'slow learner', 'retardation', 'intellectual disability'],
          'paralysis': ['paraplegic', 'quadriplegic', 'spinal cord', 'paraplegia', 'quadriplegia', 'paralysed',
                        'paralyzed', 'paralysis', 'crippled', 'leg braces', 'wheelchair'],
          'dyslexia': ['dyslexia', 'dyslexic'],
          'autism': ['autism', 'autistic', 'asperger\'s', 'ASD'],
          }

# Only keywords that don't have other meanings (combinations of common words are also problematic)
QUERIES = {'cerebral palsy': ['cerebral palsy', 'spastic'],
           'disabled': ['disabled', 'disability', 'ablism', 'ableism', 'differently abled'],
           'deaf': ['deaf', 'deafness', 'hearing impairment', 'hard of hearing', 'hearing impaired'],
           'blind': ['blind', 'blindness', 'visual impairment', 'partially sighted', 'visually impaired'],
           'epilepsy': ['epilepsy', 'epileptic'],
           'mute': ['mute', 'muteness', 'mutism', 'non-verbal'],
           'speech impairment': ['speech impairment', 'stutter', 'speech disorder', 'speech impediment'],
           'mental illness': ['mental illness', 'mental health', 'mental disorder', 'mental disability',
                              'mentally ill', 'mentally disabled', 'mentally handicapped'],
           'developmental delay': ['developmental delay', 'developmental disability', 'developmental disorder',
                                   'learning disability'],
           'paralysis': ['paraplegic', 'quadriplegic', 'paraplegia', 'quadriplegia', 'paralysis'],
           'dyslexia': ['dyslexia', 'dyslexic'],
           'autism': ['autism', 'autistic', 'asperger\'s', 'ASD']
           }


# TODO add more
SOURCES = [oop_scraper.GuardianScraper("7f2c7c42-2600-4292-a417-1b8efc5271a6"),
           oop_scraper.DailyMailScraper(),
           oop_scraper.DailyExpressScraper()
           ]

NUM_DOCS = -1  # -1 to get all documents

FILTER_THRESHOLD = 20  # higher = weaker filter


def pipeline(topic, keywords, source):
    start_time = time.time()

    fname = crawler(topic, source, NUM_DOCS, QUERIES[topic])
    # fname = source.get_fname(topic)
    print('\n Filtering:')
    filter(fname, keywords, FILTER_THRESHOLD, sample=0)
    print('\n Parsing:')
    parse(fname, keywords)
    print('\n Scoring sentiment:')
    sentiment_vader(fname)
    # sentiment_openai(fname)
    # sentiment(fname, max_articles=25)

    total_time = time.time() - start_time
    print('\nPipeline took', int(total_time // 60), 'minutes', total_time % 60, 'seconds\n')


for tpc, words in TOPICS.items():
    for src in SOURCES:
        pipeline(tpc, words, src)
    print('Plotting:')
    try:
        plot(tpc, in_folder='./out-sentiment-vader/', out_folder='./out-plot-vader/')
        # plot(tpc, in_folder='./out-sentiment-openai/', out_folder='./out-plot-openai/')
    except Exception as err:
        print('PLOT ERROR')
        print(err)
    print()
