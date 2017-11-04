import abc
import requests
from bs4 import BeautifulSoup


class Scraper(metaclass=abc.ABCMeta):
    def get_soup(self, url, parser=''):
        try:
            r = requests.get(url).text
        except:
            print('\nServer not available. Skipped %s\n' % url)
            return None
        soup = BeautifulSoup(r, parser)
        return soup

    @abc.abstractmethod
    def search_phrase(self, phrase, num_articles):
        pass

    @abc.abstractmethod
    def get_article_text(self, page_link):
        pass

    @abc.abstractmethod
    def get_fname(self, phrase):
        pass


class DailyExpressScraper(Scraper):
    def get_soup(self, url, **kwargs):
        return super().get_soup(url, parser='lxml')

    def get_article_text(self, page_link):
        first_slash = page_link.find('/', 1)
        section = page_link[1:first_slash]
        subsection = page_link[first_slash + 1:page_link.find('/', first_slash + 1)]

        url = 'http://www.express.co.uk'
        url += page_link
        soup = self.get_soup(url)
        if soup is None:
            return None

        selector = ('#singleArticle > div.ctx_content.p402_premium '
                    '> div > section > p')
        res = soup.select(selector)
        res = [r.text for r in res if not len(r.attrs)]
        res = ' '.join(res)
        res.replace('\xe2\x80', '')

        datetime = soup.find('meta', itemprop="datepublished")
        datetime = datetime['content'] if datetime else ''

        return {'datetime': datetime, 'section': section, 'subsection': subsection, 'text': res}

    def search_phrase(self, phrase, num_articles):
        url = 'http://www.express.co.uk/search?s='
        url += phrase
        print(url)
        soup = self.get_soup(url)

        selector = '#search_form > fieldset > p > b'
        n_results = int(soup.select(selector)[0].text)
        if num_articles != -1:
            n_results = min(num_articles, n_results)
        rem = n_results % 10
        if rem:
            n_results += 10 - rem

        page_links = self.__get_links(soup)
        print("getting URLs from page", 1, '/', n_results // 10, end='\r')
        for i in range(10, n_results, 10):
            print("getting URLs from page", i // 10 + 1, '/', n_results // 10, end='\r')
            page_url = '{}&o={}'.format(url, i)
            soup = self.get_soup(page_url)
            page_links += self.__get_links(soup)
        print()
        return page_links

    def __get_links(self, soup):
        selector = ('#maincontainer > div.clear.clearfix > section > div > '
                    'div.sections > section.block.search-results.mb0 > div > a')
        res = soup.select(selector)

        page_links = []
        for r in res:
            page_links.append(r.attrs['href'][24:])  # remove duplicate http://www.express.co.uk
        return page_links

    def get_fname(self, phrase):
        return 'DE-{}.json'.format(phrase)


class DailyMailScraper(Scraper):
    def get_soup(self, url, **kwargs):
        return super().get_soup(url, parser='html5lib')

    def get_article_text(self, page_link):
        section = page_link[1:page_link.find('/', 1)]
        link = page_link[page_link.find('/', 1) + 1:]
        url = 'http://www.dailymail.co.uk/news/'
        url += link

        soup = self.get_soup(url)
        if soup is None:
            return None

        res = soup.find('div', {'itemprop': 'articleBody'})
        res = res.find_all('p')
        res = [x.text for x in res]
        res = ' '.join(res)
        res.replace('\xc2\xa0', '')

        datetime = soup.find('meta', property='article:published_time')
        datetime = datetime['content'] if datetime else ''

        return {'datetime': datetime, 'section': section, 'text': res}

    def search_phrase(self, phrase, num_articles):
        url = 'http://www.dailymail.co.uk/home/search.html?sel=site&searchPhrase='
        url += phrase
        print(url)
        soup = self.get_soup(url)

        selector = ('#searchCommand > div.alpha > div > div + div '
                    '> div.sch-pagesummary.gr5ox')
        n_pages = soup.select(selector)[0].text
        n_pages = int(n_pages[n_pages.find('of') + 3:].strip())

        page_links = self.__get_links(soup)
        for i in range(1, n_pages):
            print("getting URLs from page", i, end='\r')
            page_url = '{}&offset={}&sort=recent'.format(url, 50 * i)
            soup = self.get_soup(page_url)
            page_links += self.__get_links(soup)

            if num_articles != -1 and len(page_links) >= num_articles:
                page_links = page_links[:num_articles]
                break
        print()
        return page_links

    def __get_links(self, soup):
        res = soup.select('#searchCommand > div.alpha > div > div')
        page_links = []
        for r in res:
            if r.h2 is None:
                continue
            page_links.append(r.h2.a.attrs['href'])
        return page_links

    def get_fname(self, phrase):
        return 'DM-{}.json'.format(phrase)
