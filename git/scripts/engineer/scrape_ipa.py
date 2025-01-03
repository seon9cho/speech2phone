from bs4 import BeautifulSoup
import requests

root_dir = "/Users/Seong-EunCho/Desktop/BYU/2018 Fall/acmeproject"

def scrape_ipa_wiki(ipa_type):
	wiki_url = "https://en.wikipedia.org"
	url_dict = {
				"vowel": "/wiki/IPA_vowel_chart_with_audio",
				"pulmonic_consonant": "/wiki/Template:IPA_chart_pulmonic_consonants_with_audio"
	}
	base_url = wiki_url + url_dict[ipa_type]
	source = requests.get(base_url).text
	soup = BeautifulSoup(source, "html.parser")
	vowel_links = soup.findAll(name='a', href=True, title="About this sound")
	for link in vowel_links:
		file_name = link['href'].split(':')[1]
		out_dir = "{}/ipa/{}/".format(root_dir, ipa_type)
		print(file_name)
		pl = wiki_url + link['href']
		pl_source = requests.get(pl).text
		pl_soup = BeautifulSoup(pl_source, "html.parser")
		dl_link = pl_soup.find(name='a', href=True, class_="internal")
		r = requests.get("https:" + dl_link['href'])
		with open(out_dir + file_name, 'wb') as f:
			f.write(r.content)

if __name__ == '__main__':
	scrape_ipa_wiki("pulmonic_consonant")



