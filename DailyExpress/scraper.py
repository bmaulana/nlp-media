#!/usr/local/bin/python

from bs4 import BeautifulSoup
import json
import sys
import requests
import os


def get_soup(url):
    try:
        r = requests.get(url).text
    except:
        print('\nServer not available. Skipped %s\n' % url)
        return None
    soup = BeautifulSoup(r, 'lxml')
    return soup


def search_phrase(phrase, num_articles):
    url = 'http://www.express.co.uk/search?s='
    url += phrase
    print(url)
    soup = get_soup(url)

    selector = '#search_form > fieldset > p > b'
    n_results = int(soup.select(selector)[0].text)
    if num_articles != -1:
        n_results = min(num_articles, n_results)
    rem = n_results % 10
    if rem:
        n_results += 10-rem

    page_links = get_links(soup)

    for i in range(10, n_results, 10):
        page_url = '{}&o={}'.format(url, i)
        soup = get_soup(page_url)
        page_links += get_links(soup)

    return page_links


def get_links(soup):
    selector = ('#maincontainer > div.clear.clearfix > section > div > '
                'div.sections > section.block.search-results.mb0 > div > a')
    res = soup.select(selector)
    page_links = []
    for r in res:
        page_links.append(r.attrs['href'][24:])  # remove duplicate http://www.express.co.uk
    return page_links


def get_article_text(page_link):
    first_slash = page_link.find('/', 1)
    section = page_link[1:first_slash]
    subsection = page_link[first_slash+1:page_link.find('/', first_slash+1)]
    url = 'http://www.express.co.uk'
    url += page_link
    soup = get_soup(url)
    if soup is None:
        return None
    selector = ('#singleArticle > div.ctx_content.p402_premium '
                '> div > section > p')
    res = soup.select(selector)
    res = [r.text for r in res if not len(r.attrs)]
    res = ' '.join(res)
    # res = res.encode('utf-8')
    res.replace('\xe2\x80', '')
    return {'section': section, 'subsection': subsection, 'text': res}


def main():
    if len(sys.argv) < 2:
        sys.exit('Need and word to search for\n')
    saved_links = []
    fname = 'DE-{}.json'.format(sys.argv[1])
    num_articles = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    print(fname)
    if os.path.isfile(fname):
        f = open(fname, 'rb')
        for line in f:
            saved_links.append(list(json.loads(line).keys())[0])
    page_links = search_phrase(sys.argv[1], num_articles)
    g = open('DE-{}.json'.format(sys.argv[1]), 'a')

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
