import requests
from bs4 import BeautifulSoup
import json
import os
import re
import time
from urllib.parse import urlparse

MAX_PAGES = 60

CDX_QUERIES = [
    (
        "https://web.archive.org/cdx/search/cdx"
        "?url=amenify.com/*"
        "&output=json&fl=timestamp,original&filter=statuscode:200"
        "&collapse=urlkey&limit=150"
    ),
]

WAYBACK = "https://web.archive.org/web/{timestamp}id_/{url}"

_YEAR_PATH = re.compile(r'/20\d{2}/')
_AMENIFY_RE = re.compile(r'\bamenify\b', re.IGNORECASE)
MIN_BRAND_MENTIONS = 4

_NOISE_ATTRS = re.compile(
    r'\bcomments?\b|\bsidebar\b|\bwidget\b|\bbreadcrumb\b|\bcookie\b|'
    r'\bnewsletter\b|\bsubscribe\b|\brelated[_-]posts?\b|\bsocial[_-]share\b|'
    r'\bauthor[_-]bio\b|\btag[_-]cloud\b|\bsearch[_-]form\b|'
    r'\bentry[_-]meta\b|\bpost[_-]meta\b|\bsite[_-]info\b|'
    r'\bpost[_-]navigation\b|\bwp[_-]caption\b|\bpage[_-]links\b',
    re.IGNORECASE,
)

_CONTENT_ID = re.compile(r'^(content|main|primary|post|entry)$', re.IGNORECASE)
_CONTENT_CLASS = re.compile(r'\b(entry|post|page)[_-]content\b', re.IGNORECASE)

SKIP_FRAGMENTS = [
    'et_core_page_resource', 'utm_source', 'utm_medium',
    'page/2', 'page/3', '.xml', '.css', '.js',
    '/api/', '/feed', '/rss', 'rest_route', 'wp-', 'trk=', 'page_source=',
    '/author/', '/tag/', '/category/',
]

SKIP_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg',
                   '.ico', '.mp4', '.zip', '.webp', '.woff', '.ttf'}

LIVE_SITEMAP_CANDIDATES = [
    "https://www.amenify.com/sitemap.xml",
    "https://www.amenify.com/sitemap_index.xml",
    "https://www.amenify.com/page-sitemap.xml",
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}


def is_useful(url):
    url_lower = url.lower()
    if any(url_lower.endswith(ext) for ext in SKIP_EXTENSIONS):
        return False
    if any(frag in url for frag in SKIP_FRAGMENTS):
        return False
    if _YEAR_PATH.search(url):
        return False
    return True


def _is_on_topic(content: str, url: str) -> bool:
    path = urlparse(url).path.lower()
    if 'amenify' in path:
        return True
    return len(_AMENIFY_RE.findall(content)) >= MIN_BRAND_MENTIONS


def extract_text(html):
    soup = BeautifulSoup(html, 'html.parser')

    for tag in soup(['script', 'style', 'nav', 'footer', 'header',
                     'head', 'noscript', 'iframe', 'form', 'aside']):
        tag.decompose()

    for tag in soup.find_all(True):
        cls = ' '.join(tag.get('class', []))
        tid = tag.get('id', '')
        if _NOISE_ATTRS.search(cls) or _NOISE_ATTRS.search(tid):
            tag.decompose()

    title = soup.find('title')
    title_text = title.get_text(strip=True) if title else ''

    content_root = (
        soup.find('main') or
        soup.find('article') or
        soup.find(id=_CONTENT_ID) or
        soup.find(class_=_CONTENT_CLASS) or
        soup.body or
        soup
    )
    body_text = content_root.get_text(separator='\n', strip=True)

    seen: set[str] = set()
    lines: list[str] = []
    for line in body_text.splitlines():
        line = line.strip()
        if len(line) < 20 or line in seen:
            continue
        seen.add(line)
        lines.append(line)

    content = '\n'.join(lines)
    if title_text:
        content = f"{title_text}\n\n{content}"
    return content


def try_live_sitemap():
    all_locs = []
    for sitemap_url in LIVE_SITEMAP_CANDIDATES:
        try:
            print(f"  Trying live sitemap: {sitemap_url}")
            resp = requests.get(sitemap_url, headers=HEADERS, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, 'html.parser')
                locs = [loc.get_text(strip=True) for loc in soup.find_all('loc')]
                if locs:
                    print(f"  Found {len(locs)} URLs in {sitemap_url}")
                    all_locs.extend(locs)
        except Exception as e:
            print(f"  Sitemap {sitemap_url} failed: {type(e).__name__}")
    return all_locs


def fetch_live(url, max_retries=2):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=True)
            if resp.status_code == 200 and 'text/html' in resp.headers.get('content-type', ''):
                return resp
            if resp.status_code in (403, 429, 503):
                return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def fetch_wayback(timestamp, url, max_retries=3):
    archive_url = WAYBACK.format(timestamp=timestamp, url=url)
    for attempt in range(max_retries):
        try:
            resp = requests.get(archive_url, headers=HEADERS, timeout=20)
            if resp.status_code == 200 and 'text/html' in resp.headers.get('content-type', ''):
                return resp
            return None
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    Retry {attempt + 1} ({wait}s) for {url}")
                time.sleep(wait)
    return None


def get_wayback_candidates():
    seen_paths: set[str] = set()
    candidates = []
    for cdx_url in CDX_QUERIES:
        print(f"  Querying Wayback CDX: ...{cdx_url[cdx_url.find('url='):cdx_url.find('&')]}")
        try:
            resp = requests.get(cdx_url, headers=HEADERS, timeout=90)
            rows = resp.json()
            count = 0
            for row in rows[1:]:
                timestamp, original = row[0], row[1]
                if not is_useful(original):
                    continue
                path = urlparse(original).path.rstrip('/')
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                candidates.append((timestamp, original))
                count += 1
            print(f"  → {count} usable URLs")
        except Exception as e:
            print(f"  CDX query failed: {e}")
    return candidates


def scrape():
    pages = []

    print("Step 1: Attempting live site scraping via sitemaps...")
    live_urls = try_live_sitemap()

    if live_urls:
        for url in live_urls:
            if len(pages) >= MAX_PAGES:
                break
            if not is_useful(url):
                continue
            resp = fetch_live(url)
            if resp:
                text = extract_text(resp.content)
                if len(text) > 200 and _is_on_topic(text, url):
                    pages.append({'url': url, 'content': text})
                    print(f"[LIVE {len(pages)}] {url} ({len(text)} chars)")
                time.sleep(0.5)
            else:
                print(f"  Live blocked: {url}")
    else:
        print("  Live sitemaps unavailable — falling back to Wayback Machine.")

    remaining_needed = MAX_PAGES - len(pages)
    if remaining_needed > 0:
        print(f"\nStep 2: Fetching {remaining_needed} more pages from Wayback Machine...")
        candidates = get_wayback_candidates()
        print(f"  {len(candidates)} candidate URLs")

        scraped_paths = {urlparse(p['url']).path.rstrip('/') for p in pages}

        for timestamp, url in candidates:
            if len(pages) >= MAX_PAGES:
                break
            path = urlparse(url).path.rstrip('/')
            if path in scraped_paths:
                continue

            resp = fetch_wayback(timestamp, url)
            if resp:
                text = extract_text(resp.content)
                if len(text) > 200 and _is_on_topic(text, url):
                    pages.append({'url': url, 'content': text})
                    print(f"[WB {len(pages)}] {url} ({len(text)} chars)")
                    scraped_paths.add(path)
            time.sleep(0.4)

    os.makedirs('data', exist_ok=True)
    with open('data/pages.json', 'w', encoding='utf-8') as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Scraped {len(pages)} pages → data/pages.json")


if __name__ == '__main__':
    scrape()
