import requests
import urllib
from bs4 import BeautifulSoup
from utils.utils import formateLine


subscription_key = '081233ba8e524aa0a71a531281c94a9b'
assert subscription_key
search_url = "https://api.cognitive.microsoft.com/bing/v7.0/search"

def getHtml(url):
    try:
        page = urllib.request.urlopen(url)
        html_doc = page.read()
    except Exception:
        html_doc = ''
    soup = BeautifulSoup(html_doc, 'html.parser')
    return soup

def search(question):
    print(question)
    texts = []
    search_term = question
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": search_term, "textDecorations": True, "textFormat": "HTML"}
    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()
    except Exception:
        return ''
    print("search finish")
    i = 0
    for v in search_results["webPages"]["value"]:
        texts.append(formateLine(getHtml(v["url"]).get_text()))
        i += 1
        if i == 5:
            break
    return texts

# from IPython.display import HTML
#
# rows = "\n".join(["""<tr>
#                        <td><a href=\"{0}\">{1}</a></td>
#                        <td>{2}</td>
#                      </tr>""".format(v["url"],v["name"],v["snippet"]) \
#                   for v in search_results["webPages"]["value"]])
# HTML("<table>{0}</table>".format(rows))
# print(rows)




