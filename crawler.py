import json
import sys
import os
import oop_scraper as scr
import time

guardian_api_key = "7f2c7c42-2600-4292-a417-1b8efc5271a6"


def main(query, source, num_articles, keywords=None):
    """
    Format: python crawler.py query source [max. articles]
    """
    # TODO enable using multiple keywords per topic (e.g. 'Autism' or 'Autistic')
    if keywords is None:
        keywords = [query]

    # TODO pass Scraper class constructor in parameter to main (instead of a string 'source')
    if source == 'DE':
        scraper = scr.DailyExpressScraper()
    elif source == 'DM':
        scraper = scr.DailyMailScraper()
    elif source == 'Guardian':
        scraper = scr.GuardianScraper(guardian_api_key)
    else:
        print('Unknown Source')
        return

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
    page_links = scraper.search_phrase(query, num_articles)
    print("Number of articles found", len(page_links))
    print(time.time() - start_time, "seconds spent finding articles")

    start_time = time.time()
    g = open(fname, 'a')
    # TODO Save keywords to the first line of JSON for use by other scripts.
    count, skip_count = 0, 0
    for link in page_links:
        print(count, "/", len(page_links), "articles parsed", end="\r")
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

    # TODO return filename (to use as parameter for filter)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit('Format: py crawler.py query source [max. articles]\n')
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else -1)
