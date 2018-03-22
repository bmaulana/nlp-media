import abc
import json
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
        # Return list of page links
        # Note: if num_articles is -1, then return all articles
        pass

    @abc.abstractmethod
    def get_article_text(self, page_link):
        # Return JSON representation of article source, text, datetime, section and title
        pass

    @abc.abstractmethod
    def get_fname(self, phrase):
        # Get output filename to store articles with this source (e.g. DM-<phrase>.json)
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

        selector = '#singleArticle > div.ctx_content.p402_premium > div > section > p'
        res = soup.select(selector)
        res = [r.text for r in res if not len(r.attrs)]
        res = ' '.join(res)
        res.replace('\xe2\x80', '')

        datetime = soup.find('meta', property="article:published_time")
        datetime = datetime['content'] if datetime else ''
        title = soup.find('meta', property="og:title")
        title = title['content'] if title else ''

        source = 'Daily Express'

        return {'source': source, 'title': title, 'datetime': datetime, 'section': section, 'subsection': subsection,
                'text': res}

    def search_phrase(self, phrase, num_articles):
        url = 'http://www.express.co.uk/search?s='
        url += phrase
        print(url)
        soup = self.get_soup(url)
        if soup is None:
            print('Server unavailable.\n')
            return []

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
            if soup is not None:
                page_links += self.__get_links(soup)
        print()
        return page_links

    def __get_links(self, soup):
        selector = ('#maincontainer > div.clear.clearfix > section > div > '
                    'div.sections > section.block.search-results.mb0 > div > a')
        res = soup.select(selector)

        page_links = []
        for r in res:
            page_links.append(r.attrs['href'])
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
        if res is None:
            return None
        res = res.find_all('p')
        res = [x.text for x in res]
        res = ' '.join(res)
        res.replace('\xc2\xa0', '')

        title = soup.find('meta', property='og:title')
        title = title['content'] if title else ''
        datetime = soup.find('meta', property='article:published_time')
        datetime = datetime['content'] if datetime else ''

        source = 'Daily Mail'

        return {'source': source, 'title': title, 'datetime': datetime, 'section': section, 'text': res}

    def search_phrase(self, phrase, num_articles):
        url = 'http://www.dailymail.co.uk/home/search.html?sel=site&searchPhrase='
        url += phrase
        print(url)
        soup = self.get_soup(url)
        if soup is None:
            print('Server unavailable.\n')
            return []

        selector = '#searchCommand > div.alpha > div > div + div > div.sch-pagesummary.gr5ox'
        n_pages = soup.select(selector)[0].text
        n_pages = int(n_pages[n_pages.find('of') + 3:].strip())

        page_links = self.__get_links(soup)
        for i in range(1, n_pages):
            print("getting URLs from page", i, end='\r')
            page_url = '{}&offset={}&sort=recent'.format(url, 50 * i)
            soup = self.get_soup(page_url)
            if soup is not None:
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


class GuardianScraper(Scraper):
    # TODO fix to get around 5000/day API calls limit (try: get body on search_phrase, map[link] = body
    # TODO and have get_article_text return map[link])

    def __init__(self, api_key):
        self.api_key = api_key
        self.articles = {}

    def get_fname(self, phrase):
        return 'Guardian-{}.json'.format(phrase)

    def search_phrase(self, phrase, num_articles):
        url = 'https://content.guardianapis.com/search?q=\"' + phrase + '\"&api-key=' + self.api_key + \
              '&show-fields=body'
        print(url)

        page_links = []
        page = 0
        while num_articles == -1 or num_articles > 0:  # TODO refactor to be tqdm-friendly
            try:
                page += 1
                if num_articles == -1:
                    param = '&page=' + str(page) + '&page-size=50'
                else:
                    param = '&page=' + str(page) + '&page-size=' + str(min(50, num_articles))
                    num_articles -= 50

                r = requests.get(url + param)
                data = json.loads(r.text)

                if data['response']['status'] == 'error' or data['response']['total'] == 0:
                    break
                data = data['response']['results']

                for i in data:
                    page_links.append(i['apiUrl'])

                    # Cache body text on /search endpoint (50 articles per API call) instead of single-item endpoint
                    # Due to rate limit of 5000 API calls / day
                    self.articles[i['apiUrl']] = {
                        'source': 'Guardian',
                        'title': i['webTitle'],
                        'datetime': i['webPublicationDate'],
                        'section': i['sectionName'],
                        'text': i['fields']['body']
                    }

            except:
                print('\nServer not available. Skipped %s\n' % url)
                break

        return page_links

    def get_article_text(self, page_link):
        if page_link in self.articles:
            ret = self.articles[page_link]

            soup = BeautifulSoup(ret['text'], 'html5lib')
            res = soup.find_all('p')
            res = [x.text for x in res]
            res = ' '.join(res)
            res.replace('[]\\', '')

            ret['text'] = res
            return ret

        # should never reach here, check whenever something is printed (will reach 5000/day API call rate limit quickly)
        url = page_link + '?api-key=' + self.api_key + '&show-fields=body'
        print(url)

        try:
            r = requests.get(url)
            data = json.loads(r.text)

            source = 'Guardian'
            title = data['response']['content']['webTitle']
            datetime = data['response']['content']['webPublicationDate']
            section = data['response']['content']['sectionName']

            soup = BeautifulSoup(data['response']['content']['fields']['body'], 'html5lib')
            res = soup.find_all('p')
            res = [x.text for x in res]
            res = ' '.join(res)
            res.replace('[]\\', '')
            return {'source': source, 'title': title, 'datetime': datetime, 'section': section, 'text': res}

        except:
            print('\nServer not available. Skipped %s\n' % url)
            return None

    def free(self):
        # Removes all saved articles. Useful when you're done with a query.
        self.articles = {}
