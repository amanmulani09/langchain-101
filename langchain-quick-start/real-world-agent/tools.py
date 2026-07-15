import urllib.error 
import urllib.request

from langchain.tools import tool

@tool('fetch_text_from_url',description="fetch the document from url")
def fetch_text_from_url(url:str) -> str:

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )

    try:
        with urllib.request.urlopen(req,timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetch failed : {e}"
    
    text = raw.decode("utf-8",errors="replace")
    return text