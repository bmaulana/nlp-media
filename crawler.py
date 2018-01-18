import json
import sys
import os
import oop_scraper as scr
import time


def main():
    """
    Format: python crawler.py query source [max. articles]
    """

    if len(sys.argv) < 3:
        sys.exit('Format: py crawler.py query source [max. articles]\n')

    if sys.argv[2] == 'DE':
        scraper = scr.DailyExpressScraper()
    elif sys.argv[2] == 'DM':
        scraper = scr.DailyMailScraper()
    else:
        print('Unknown Source')
        return

    if not os.path.exists('./out/'):
        os.makedirs('./out/')
    fname = './out/' + scraper.get_fname(sys.argv[1])
    print(fname)
    num_articles = int(sys.argv[3]) if len(sys.argv) > 3 else -1

    saved_links = []
    if os.path.isfile(fname):
        f = open(fname, 'rb')
        for line in f:
            saved_links.append(list(json.loads(line).keys())[0])

    # TODo for multi-word search phrases, ensure these words are next to each other in article
    start_time = time.time()
    page_links = scraper.search_phrase(sys.argv[1], num_articles)
    print("Number of articles found", len(page_links))
    print(time.time() - start_time, "seconds spent finding articles")

    start_time = time.time()
    g = open(fname, 'a')
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


if __name__ == '__main__':
    main()
