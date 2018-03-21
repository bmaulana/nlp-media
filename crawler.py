import json
import sys
import os
import oop_scraper as scr
import time
from tqdm import tqdm


def crawler(query, scraper, num_articles=-1, keywords=None):
    """
    Format: python crawler.py query source [max. articles]
    """
    if keywords is None:
        keywords = [query]

    if not os.path.exists('./out/'):
        os.makedirs('./out/')
    fname = './out/' + scraper.get_fname(query)
    print(fname)

    saved_links = set()
    if os.path.isfile(fname):
        f = open(fname, 'rb')
        for line in f:
            saved_links.add(list(json.loads(line).keys())[0])

    start_time = time.time()
    page_links = set()
    for keyword in keywords:
        page_links = page_links.union(set(scraper.search_phrase(keyword, num_articles)))
    print("Number of articles found", len(page_links))
    print(time.time() - start_time, "seconds spent finding articles")

    start_time = time.time()
    g = open(fname, 'a')
    count, skip_count = 0, 0
    for link in tqdm(page_links):
        print(count, "/", len(page_links), "articles saved", end="\r")
        count += 1
        if link in saved_links:
            skip_count += 1
            continue

        res = scraper.get_article_text(link)
        if res is None:
            continue
        res = {link: res}

        g.write(json.dumps(res))
        g.write('\n')
        g.flush()

    print(count, "/", len(page_links), "articles parsed")
    print(skip_count, "articles already saved")
    time_spent = time.time()-start_time
    print(int(time_spent // 60), 'minutes', time_spent % 60, "seconds spent parsing articles")
    g.close()

    return fname.split('/')[-1]


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit('Format: python crawler.py query source [max. articles] [keywords]\n')

    source = sys.argv[2]
    if source == 'DE':
        scraper_inst = scr.DailyExpressScraper()
    elif source == 'DM':
        scraper_inst = scr.DailyMailScraper()
    elif source == 'Guardian':
        scraper_inst = scr.GuardianScraper("7f2c7c42-2600-4292-a417-1b8efc5271a6")
    else:
        sys.exit('Unknown Source')

    if len(sys.argv) == 3:
        crawler(sys.argv[1], scraper_inst)
    elif len(sys.argv) == 4:
        crawler(sys.argv[1], scraper_inst, int(sys.argv[3]))
    else:
        keys = sys.argv[4].split(',')
        crawler(sys.argv[1], scraper_inst, int(sys.argv[3]), keys)
