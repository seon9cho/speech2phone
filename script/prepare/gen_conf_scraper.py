b
import re
import urllib
from tqdm import tqdm
import subprocess
import os
import argparse
from pathlib import Path

all_langs = ['apw', 'aym', 'ind', 'msa', 'bis', 'cak', 'ceb', 'ces', 'dan', 'deu', 'nav', 'cuk', 'yor', 'est', 'efi', 'eng', 'spa', 'ton', 'fat', 'hif', 'chk', 'fra', 'smo', 'grn', 'hil', 'hmn', 'hrv', 'ibo', 'ilo', 'isl', 'ita', 'kos', 'mah', 'qvi', 'gil', 'swa', 'hat', 'lav', 'lit', 'lin', 'hun', 'pon', 'mlg', 'mlt', 'mam', 'nld', 'cag', 'nor', 'pau', 'pap', 'pol', 'por', 'ept', 'kek', 'quc', 'tah', 'ron', 'tsn', 'sna', 'alb', 'slk', 'slv', 'fin', 'swe', 'tgl', 'yap', 'vie', 'tur', 'twi', 'fij', 'xho', 'quz', 'zul', 'ell', 'bul', 'kaz', 'mon', 'rus', 'srp', 'ukr', 'kat', 'hye', 'urd', 'ara', 'pes', 'amh', 'nep', 'hin', 'tam', 'tel', 'sin', 'tha', 'lao', 'mya', 'khm', 'kor', 'zho', 'jpn', 'zhs']

def valid_lang():
    base_url = "https://www.lds.org/languages?lang=eng"
    conf_url = "https://www.lds.org/general-conference/conferences"
    source = requests.get(base_url).text
    'soup = BeautifulSoup(source, "html.parser")'
    lang_grid = soup.find(name='ul', class_="lds-grid lds-grid--lg lds-grid--linklist language-list")
    lang_links = lang_grid.findAll(name='a', href=True)
    lang_list = []
    for l in lang_links:
        lang = l['data-lang']
        lang_url = conf_url + "?lang=" + lang
        conf_source = requests.get(lang_url).text
        if "An error occurred. Please try again later." in conf_source:
            print("Invalid:", lang)
        else:
            print("Valid:", lang)
            lang_list.append(lang)
    return lang_list

def scrape(language, interval, out_dir, skip=True):
    lds_url = "https://www.lds.org"
    base_url = "https://www.lds.org/general-conference/conferences"
    interval = (2015, 2018)
    interval_list = list(range(interval[0], interval[1]+1))
    stringy = [str(s) for s in interval_list]
    reg_str = "|".join(stringy)
    pattern = re.compile(reg_str)
    lang_url = base_url + "?lang=" + language
    source = requests.get(lang_url).text
    soup = BeautifulSoup(source, "html.parser")
    gen_links = soup.findAll(name='a', href=True, class_="year-line__link")
    for i, gen_link in enumerate(gen_links):
        if not pattern.search(gen_link.text):
            continue
        if "general-conference" not in gen_link['href']:
            continue
        conf_url = lds_url + gen_link['href']
        conf_source = requests.get(conf_url).text
        conf_soup = BeautifulSoup(conf_source, "html.parser")
        talk_links = conf_soup.findAll(name='a', href=True, class_="lumen-tile__link")
        loop = tqdm(total=len(talk_links), position=0, leave=False)
        loop.set_description("Session: {}/{}".format(i+1, 2*len(interval_list)))
        for talk in talk_links:
            talk_name = talk['href'].split('/')[2:]
            talk_name[-1] = talk_name[-1].split('?')[0]
            talk_name.append(language)
            talk_name = '_'.join(talk_name) + ".mp3"
            talk_url = lds_url + talk['href']
            talk_source = requests.get(talk_url).text
            talk_soup = BeautifulSoup(talk_source, "html.parser")
            button = talk_soup.find(name='a', href=True, class_="button button--round button--blue", text="MP3")
            try:
                mp3file = urllib.request.urlopen(button['href'])
                file_name = Path(out_dir + '/' + language + '/' + talk_name)
                if file_name.is_file():
                    print(str(file_name), "exists")
                    continue
                with open(str(file_name), 'wb') as output:
                    output.write(mp3file.read())
            except TypeError:
                continue
            loop.update(1)
        loop.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                      description="Scrapes General Conference audio data" \
                                  "from the site lds.org")
    parser.add_argument('-o', '--out-dir', 
                        help="Output dirctory of the audio data",
                        required=True)
    parser.add_argument('-s', '--skip', action="store_true",
                        help="Skips downloading over existing data")
    parser.add_argument('-y', '--years', nargs='+', type=int,
                        help="Years to download from",
                        required=True)
    args = parser.parse_args()
    #all_langs = valid_lang()
    
    for i,lang in enumerate(all_langs):
        print("Current language:", lang, "({}/{})".format(i, len(all_langs)))
        try:
            os.mkdir(args.out_dir+'/'+lang)
        except FileExistsError:
            print(args.out_dir+'/'+lang, "exists")
            print("Skipping...")
            continue
        if not args.skip:
            scrape(lang, args.years, args.out_dir, skip=False)
        else: scrape(lang, args.years, args.out_dir, skip=True)
    

