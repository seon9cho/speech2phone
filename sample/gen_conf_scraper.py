from bs4 import BeautifulSoup
import requests
import re
import urllib3
from tqdm import tqdm

def main():
    lds_url = "https://www.lds.org"
    base_url = "https://www.lds.org/general-conference/conferences"
    language = "eng"
    interval = (2016, 2018)
    interval_list = list(range(interval[0], interval[1]+1))
    stringy = [str(s) for s in interval_list]
    reg_str = "|".join(stringy)
    lang_url = base_url + "?lang=" + language
    source = requests.get(lang_url).text
    soup = BeautifulSoup(source, "html.parser")
    gen_links = soup.findAll(name='a', href=True, class_="year-line__link")
    for gen_link in gen_links:
        pattern = re.compile(reg_str)
        if pattern.search(gen_link.text):
            print(gen_link.text)
    return
    for gen_link in gen_links:
        if not pattern.search(gen_link.text):
            continue
        if "general-conference" not in gen_link['href']:
            continue
        conf_url = lds_url + gen_link['href']
        conf_source = requests.get(conf_url).text
        conf_soup = BeautifulSoup(conf_source, "html.parser")
        talk_links = conf_soup.findAll(name='a', href=True, class_="lumen-tile__link")
        for talk in talk_links:
            talk_url = lds_url + talk['href']
            talk_source = requests.get(talk_url).text
            talk_soup = BeautifulSoup(talk_source, "html.parser")
            button = talk_soup.find(name='a', href=True, class_="button button--round button--blue", text="MP3")
            print(type(button['href']))
            mp3file = urllib3.urlopen(button['href'])
            with open("audio/test.mp3", 'wb') as output:
                output.write(mp3file.read())
            break
        break

if __name__ == '__main__':
    main()