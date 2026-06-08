"""
Tavily-backed ETF data extraction from etfdb.com.

Uses extract() for structured page content and falls back to
requests+BeautifulSoup for HTML table parsing.
"""

import os
import re
import time
import logging
from typing import Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from models import ETF, Issuer

load_dotenv()
log = logging.getLogger(__name__)

ISSUERS_URL = "https://etfdb.com/etfs/issuers/"
ISSUER_PAGE_TPL = "https://etfdb.com/etfs/issuers/{slug}/"

# Delay between requests to be polite
REQUEST_DELAY = 1.5

# ─── Tavily client (lazy init) ────────────────────────────────────────────────

_tavily_client = None

def _get_tavily():
    global _tavily_client
    if _tavily_client is None:
        from tavily import TavilyClient
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise EnvironmentError("TAVILY_API_KEY not set. Copy .env.example → .env and add your key.")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_number(text: str) -> Optional[float]:
    """Strip $, commas, % and parse to float. Returns None on failure."""
    if not text:
        return None
    cleaned = re.sub(r"[$,%\s]", "", text.strip())
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_volume(text: str) -> Optional[int]:
    val = _parse_number(text)
    return int(val) if val is not None else None


def _derive_acronym(issuer_name: str) -> str:
    """
    Produce a short acronym for the issuer.
    'Vanguard' → 'VG', 'BlackRock' → 'BLK', 'Invesco' → 'IVZ', etc.
    Falls back to first-letters of words or first 3 chars.
    """
    known = {
        "vanguard": "VG",
        "blackrock": "BLK",
        "invesco": "IVZ",
        "state street": "STT",
        "schwab": "SCHW",
        "first trust": "FT",
        "wisdomtree": "WETF",
        "jpmorgan": "JPM",
        "fidelity": "FID",
        "dimensional": "DFA",
        "ishares": "BLK",   # iShares is BlackRock
        "proshares": "PSH",
        "global x": "GX",
        "vaneck": "VEC",
        "ark": "ARK",
        "direxion": "DIR",
    }
    lower = issuer_name.lower()
    for key, acronym in known.items():
        if key in lower:
            return acronym
    # generic fallback: first letters of each word, upper-cased
    words = issuer_name.split()
    if len(words) >= 2:
        return "".join(w[0] for w in words).upper()
    return issuer_name[:3].upper()


# ─── Issuer list scraping ─────────────────────────────────────────────────────

def fetch_issuers(use_tavily: bool = True) -> list[Issuer]:
    """
    Scrape the ETF DB issuers power-rankings page and return Issuer objects.
    Falls back to direct requests if Tavily is unavailable.
    """
    log.info("Fetching issuer list from %s", ISSUERS_URL)

    html = _fetch_html(ISSUERS_URL, use_tavily=use_tavily)
    return _parse_issuers(html)


def _parse_issuers(html: str) -> list[Issuer]:
    soup = BeautifulSoup(html, "lxml")
    issuers: list[Issuer] = []

    # etfdb renders a table with id "issuer-power-rankings__fund-flow" or similar
    # Also look for any table with issuer links
    table = (
        soup.find("table", {"id": re.compile(r"issuer", re.I)})
        or soup.find("table")
    )

    if table is None:
        log.warning("No table found on issuers page – trying anchor-based fallback")
        return _parse_issuers_from_links(soup)

    rows = table.find_all("tr")
    for row in rows[1:]:  # skip header
        cols = row.find_all(["td", "th"])
        if not cols:
            continue
        # First column typically contains the issuer link
        link_tag = cols[0].find("a")
        if link_tag is None:
            continue
        name = link_tag.get_text(strip=True)
        href = link_tag.get("href", "")
        slug = _slug_from_href(href, name)
        aum = _parse_number(cols[1].get_text(strip=True)) if len(cols) > 1 else None

        issuers.append(Issuer(
            name=name,
            slug=slug,
            acronym=_derive_acronym(name),
            url=ISSUER_PAGE_TPL.format(slug=slug),
            aum_billions=aum,
        ))

    log.info("Found %d issuers via table", len(issuers))
    return issuers


def _parse_issuers_from_links(soup: BeautifulSoup) -> list[Issuer]:
    """Fallback: find all /etfs/issuers/{slug}/ links."""
    issuers = []
    seen = set()
    for a in soup.find_all("a", href=re.compile(r"/etfs/issuers/[^/]+/$")):
        href = a["href"]
        slug = href.rstrip("/").split("/")[-1]
        if slug in seen or slug in ("", "#"):
            continue
        seen.add(slug)
        name = a.get_text(strip=True) or slug.replace("-", " ").title()
        issuers.append(Issuer(
            name=name,
            slug=slug,
            acronym=_derive_acronym(name),
            url=f"https://etfdb.com{href}",
        ))
    log.info("Found %d issuers via link fallback", len(issuers))
    return issuers


def _slug_from_href(href: str, name: str) -> str:
    m = re.search(r"/etfs/issuers/([^/]+)/?", href)
    if m:
        return m.group(1)
    return name.lower().replace(" ", "-")


# ─── Per-issuer ETF scraping ──────────────────────────────────────────────────

def fetch_issuer_etfs(issuer: Issuer, use_tavily: bool = True) -> list[ETF]:
    """Fetch and parse the ETF list for a single issuer, filtering for Equity."""
    log.info("Fetching ETFs for %s → %s", issuer.name, issuer.url)
    time.sleep(REQUEST_DELAY)

    html = _fetch_html(issuer.url, use_tavily=use_tavily)
    all_etfs = _parse_etf_table(html)
    equity = [e for e in all_etfs if e.asset_class.lower() == "equity"]
    log.info("  %s: %d total ETFs, %d equity", issuer.name, len(all_etfs), len(equity))
    return equity


def _parse_etf_table(html: str) -> list[ETF]:
    soup = BeautifulSoup(html, "lxml")
    etfs: list[ETF] = []

    # etfdb uses a table with class "etf-ticker-table" or data tables
    table = (
        soup.find("table", {"id": re.compile(r"etf", re.I)})
        or soup.find("table", {"class": re.compile(r"etf", re.I)})
        or soup.find("table")
    )

    if table is None:
        log.warning("No ETF table found, attempting text-based parsing")
        return _parse_etfs_from_text(str(soup))

    headers = _extract_headers(table)
    col_idx = _map_columns(headers)

    rows = table.find_all("tr")
    for row in rows[1:]:
        cols = row.find_all(["td", "th"])
        if len(cols) < 3:
            continue

        symbol = _cell_text(cols, col_idx.get("symbol", 0))
        name = _cell_text(cols, col_idx.get("name", 1))
        asset_class = _cell_text(cols, col_idx.get("asset_class", 2))

        if not symbol or not asset_class:
            continue

        etfs.append(ETF(
            symbol=symbol.upper(),
            name=name,
            asset_class=asset_class,
            aum_millions=_parse_number(_cell_text(cols, col_idx.get("aum"))),
            fund_flow_pct=_parse_number(_cell_text(cols, col_idx.get("fund_flow"))),
            avg_volume=_parse_volume(_cell_text(cols, col_idx.get("volume"))),
            price=_parse_number(_cell_text(cols, col_idx.get("price"))),
            one_year_return_pct=_parse_number(_cell_text(cols, col_idx.get("return"))),
        ))

    return etfs


def _extract_headers(table) -> list[str]:
    thead = table.find("thead")
    if thead:
        return [th.get_text(strip=True).lower() for th in thead.find_all(["th", "td"])]
    first_row = table.find("tr")
    if first_row:
        return [th.get_text(strip=True).lower() for th in first_row.find_all(["th", "td"])]
    return []


def _map_columns(headers: list[str]) -> dict[str, int]:
    """Map semantic column names to indices."""
    mapping: dict[str, int] = {}
    patterns = {
        "symbol": re.compile(r"symbol|ticker"),
        "name": re.compile(r"name|fund"),
        "asset_class": re.compile(r"asset.?class|type"),
        "aum": re.compile(r"aum|assets"),
        "fund_flow": re.compile(r"flow|ytd"),
        "volume": re.compile(r"volume|vol"),
        "price": re.compile(r"price"),
        "return": re.compile(r"return|1.?y"),
    }
    for i, h in enumerate(headers):
        for key, pat in patterns.items():
            if key not in mapping and pat.search(h):
                mapping[key] = i
    return mapping


def _cell_text(cols, idx) -> str:
    if idx is None or idx >= len(cols):
        return ""
    return cols[idx].get_text(strip=True)


def _parse_etfs_from_text(text: str) -> list[ETF]:
    """
    Last-resort: regex extraction of ticker + name + asset class patterns
    from raw page text (handles Tavily markdown output).
    """
    # Pattern: SYMBOL  Name  Asset Class  ...
    pattern = re.compile(
        r"\b([A-Z]{2,6})\b\s+"                          # symbol
        r"([A-Za-z][A-Za-z0-9 &/\-,\.]+?)\s+"           # name
        r"(Equity|Fixed Income|Commodity|Currency|"      # asset class
        r"Real Estate|Multi-Asset|Alternative)\b",
        re.MULTILINE,
    )
    etfs = []
    seen = set()
    for m in pattern.finditer(text):
        sym = m.group(1)
        if sym in seen:
            continue
        seen.add(sym)
        etfs.append(ETF(symbol=sym, name=m.group(2).strip(), asset_class=m.group(3)))
    return etfs


# ─── Low-level fetch (Tavily extract or fallback requests) ────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch_html(url: str, use_tavily: bool = True) -> str:
    """
    Fetch page content.  Tries Tavily extract first (handles JS-rendered pages),
    then falls back to direct requests.
    """
    if use_tavily:
        try:
            client = _get_tavily()
            result = client.extract(urls=[url])
            # result is a dict with 'results' list; each item has 'raw_content'
            if result and result.get("results"):
                content = result["results"][0].get("raw_content", "")
                if content:
                    log.debug("Tavily extract succeeded for %s (%d chars)", url, len(content))
                    return content
            log.warning("Tavily returned empty content for %s, falling back", url)
        except Exception as exc:
            log.warning("Tavily extract failed (%s), falling back to requests", exc)

    # Direct HTTP fallback
    resp = requests.get(url, headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.text
