subscription_key = '081233ba8e524aa0a71a531281c94a9b'
assert subscription_key
search_url = "https://api.cognitive.microsoft.com/bing/v7.0/search"
search_term = "Did they dry them properly?"

import requests
import urllib
from bs4 import BeautifulSoup

headers = {"Ocp-Apim-Subscription-Key": subscription_key}
params = {"q": search_term, "textDecorations": True, "textFormat": "HTML"}
response = requests.get(search_url, headers=headers, params=params)
response.raise_for_status()
search_results = response.json()


# from IPython.display import HTML
#
# rows = "\n".join(["""<tr>
#                        <td><a href=\"{0}\">{1}</a></td>
#                        <td>{2}</td>
#                      </tr>""".format(v["url"],v["name"],v["snippet"]) \
#                   for v in search_results["webPages"]["value"]])
# HTML("<table>{0}</table>".format(rows))
# print(rows)

def getHtml(url):

    page = urllib.request.urlopen(url)
    html_doc = page.read()
    soup = BeautifulSoup(html_doc, 'html.parser')
    return soup

for v in search_results["webPages"]["value"]:
    print(getHtml(v["url"]).get_text())
    break


# import http.client, urllib.request, urllib.parse, urllib.error, base64
#
# headers = {
#     # Request headers
#     'Ocp-Apim-Subscription-Key': '{081233ba8e524aa0a71a531281c94a9b}',
# }
#
# params = urllib.parse.urlencode({
#     # Request parameters
#     'q': 'bill gates',
#     'count': '1',
#     'offset': '0',
#     'mkt': 'en-us',
#     'safesearch': 'Moderate',
# })
#
# try:
#     conn = http.client.HTTPSConnection('api.cognitive.microsoft.com')
#     conn.request("GET", "/bing/v7.0/search?%s" % params, "{body}", headers)
#     response = conn.getresponse()
#     data = response.read()
#     print(data)
#     conn.close()
# except Exception as e:
#     print("[Errno {0}] {1}".format(e.errno, e.strerror))
