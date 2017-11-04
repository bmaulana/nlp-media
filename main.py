import json
import sys
import os
import oop_scraper as scr


def main():
    if len(sys.argv) < 3:
        sys.exit('Format: main.py query source [max. articles]\n')

    if sys.argv[2] == 'DE':
        scraper = scr.DailyExpressScraper()
    elif sys.argv[2] == 'DM':
        scraper = scr.DailyMailScraper()
    else:
        print('Unknown Source')
        return
    fname = scraper.get_fname(sys.argv[1])
    print(fname)
    num_articles = int(sys.argv[3]) if len(sys.argv) > 3 else -1

    saved_links = []
    if os.path.isfile(fname):
        f = open(fname, 'rb')
        for line in f:
            saved_links.append(list(json.loads(line).keys())[0])

    page_links = scraper.search_phrase(sys.argv[1], num_articles)
    print("Number of articles found", len(page_links))

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
    g.close()


if __name__ == '__main__':
    main()
