#!/usr/local/bin/python

from bs4 import BeautifulSoup
import json
import sys
import requests
import os


def get_soup(url):
    try:
        # session = dryscrape.Session()
        # session.visit(url)
        # r = session.body()
        r = requests.get(url).text
    except:
        print('\nServer not available. Skipped %s\n' % url)
        return None
    soup = BeautifulSoup(r, 'html5lib')
    return soup


def get_links(soup):
    res = soup.select('#searchCommand > div.alpha > div > div')
    page_links = []
    for r in res:
        if r.h2 is None:
            continue
        page_links.append(r.h2.a.attrs['href'])
    return page_links


def search_phrase(phrase, num_articles):
    url = 'http://www.dailymail.co.uk/home/search.html?sel=site&searchPhrase='
    url += phrase
    print(url)
    soup = get_soup(url)

    selector = ('#searchCommand > div.alpha > div > div + div '
                '> div.sch-pagesummary.gr5ox')
    n_pages = soup.select(selector)[0].text
    n_pages = int(n_pages[n_pages.find('of')+3:].strip())

    page_links = get_links(soup)

    for i in range(1, n_pages):
        print("getting URLs from page", i)
        page_url = '{}&offset={}&sort=recent'.format(url, 50*i)
        soup = get_soup(page_url)
        page_links += get_links(soup)

        if num_articles != -1 and len(page_links) >= num_articles:
            page_links = page_links[:num_articles]
            break

    return page_links


def get_article_text(page_link):
    section = page_link[1:page_link.find('/', 1)]
    link = page_link[page_link.find('/', 1)+1:]
    url = 'http://www.dailymail.co.uk/news/'
    url += link
    soup = get_soup(url)
    if soup is None:
        return None
    res = soup.find_all('p', {'class': 'mol-para-with-font'})
    res = [x.text for x in res]
    res = ' '.join(res)
    # res = res.encode('utf-8')
    res.replace('\xc2\xa0', '')
    return {'section': section, 'text': res}


def main():
    if len(sys.argv) < 2:
        sys.exit('Need and word to search for\n')
    saved_links = []
    fname = 'DM-{}.json'.format(sys.argv[1])
    num_articles = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    print(fname)
    if os.path.isfile(fname):
        f = open(fname, 'rb')
        for line in f:
            saved_links.append(list(json.loads(line).keys())[0])
    page_links = search_phrase(sys.argv[1], num_articles)
    g = open('DM-{}.json'.format(sys.argv[1]), 'a')

    print("Number of articles found", len(page_links))
    count, skip_count = 0, 0
    for link in page_links:
        print(count, "/", len(page_links), "articles parsed", end="\r")
        count += 1
        if link in saved_links:
            skip_count += 1
            continue
        res = get_article_text(link)
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
