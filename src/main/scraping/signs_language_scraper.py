import requests
from bs4 import BeautifulSoup


class SignsLanguageScraper:
    def __init__(self, base_url):
        self.base_url = base_url
        page = requests.get(self.base_url)
        self.soup = BeautifulSoup(page.content, 'html.parser')

    def scrap(self):
        recording_links = self.get_recording_links()
        for recording_link in recording_links:
            recording_name = recording_link.text
            recording_url = recording_link['href']

            video_links = self.get_recording_video_links(recording_url)
            for video_link in video_links:
                print()

    def get_recording_links(self):
        links = self.soup.find_all('a', lambda x: x.next_sibling['class'] == 'Z3988')
        return links

    @staticmethod
    def get_recording_video_links(link_url):
        recording_page = requests.get(link_url)
        recording_soup = BeautifulSoup(recording_page.content, 'html.parser')

        video_links = recording_soup.find_all('a', href=lambda url: '.mp4' or '.flv' in url.lower())
        return video_links
