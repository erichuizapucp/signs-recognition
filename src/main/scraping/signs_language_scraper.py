import requests
import logging
import os

from bs4 import BeautifulSoup


class SignsLanguageScraper:
    def __init__(self, base_url, storage_location):
        self.logger = logging.getLogger(__name__)

        self.base_url = base_url
        self.storage_location = storage_location
        self.chunk_size = 8192

    def scrap(self, url):
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')

        rec_session_links = [link for link in soup.find_all('a') if link.next_sibling.name == 'span' and
                             'Z3988' in link.next_sibling['class']]

        self.logger.debug('%d recording sessions were found.', len(rec_session_links))

        for index, rec_link in enumerate(reversed(rec_session_links)):
            recording_name = rec_link.text
            recording_url = rec_link['href']

            self.logger.debug('Scraping recording session %s at %s', recording_name, recording_url)

            video_links = self.get_rec_video_links(recording_url)
            for video_link in video_links:
                video_url = video_link['href']
                video_label = video_link.next.next

                self.logger.debug('Scraping video ''%s'' at %s', video_label, video_url)

                local_video_path = self.get_video_local_path(index + 1, video_label)
                if not os.path.exists(local_video_path):
                    self.download_file(video_url, local_video_path)
                    self.logger.debug('Video stored locally at ''%s''', local_video_path)

    def get_rec_video_links(self, url):
        rec_url = self.base_url + url
        page = requests.get(rec_url)
        soup = BeautifulSoup(page.content, 'html.parser')

        return soup.find_all('a', href=lambda x: '.mp4' in str(x).lower() or '.flv' in str(x).lower())

    def download_file(self, url, local_path):
        abs_url = self.base_url + url

        if not os.path.exists(os.path.dirname(local_path)):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

        with requests.get(abs_url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)

    def get_video_local_path(self, session_id, label: str):
        local_video_path: str = \
            os.path.join(self.storage_location, str(session_id), label.lower().
                         replace(' lsp', 'lsp').replace(' ', '-').replace(',', '').replace('---', '-'))

        ext_index = local_video_path.find('.mp4')
        if ext_index < 0:
            ext_index = local_video_path.find('.flv')

        local_video_path = local_video_path[:ext_index + 4]
        return local_video_path
