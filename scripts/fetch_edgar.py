"""Download real 10-K filings from SEC EDGAR.

For each configured ticker this resolves ticker -> CIK, finds the most recent
10-K, downloads the primary document, strips it to clean text, and writes:

    data/corpus/{TICKER}_{FY}_10K.txt    # clean text
    data/corpus/{TICKER}_{FY}_10K.json   # metadata sidecar

SEC requires a descriptive User-Agent with contact info on every request
(set EDGAR_USER_AGENT in .env). We also rate-limit politely.

    python scripts/fetch_edgar.py                 # tickers from .env (NVDA,AAPL,MSFT)
    python scripts/fetch_edgar.py NVDA TSLA       # override on the CLI
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.loaders import clean_text  # noqa: E402
from config import settings  # noqa: E402

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": settings.edgar_user_agent, "Accept-Encoding": "gzip, deflate"})
    return s


def _get(sess: requests.Session, url: str) -> requests.Response:
    resp = sess.get(url, timeout=30)
    resp.raise_for_status()
    time.sleep(0.2)  # be polite to SEC (well under the ~10 req/s limit)
    return resp


def _ticker_to_cik(sess: requests.Session) -> dict[str, dict]:
    data = _get(sess, TICKERS_URL).json()
    out: dict[str, dict] = {}
    for row in data.values():
        out[row["ticker"].upper()] = {"cik": int(row["cik_str"]), "title": row["title"]}
    return out


def _latest_10k(sess: requests.Session, cik: int) -> dict | None:
    cik10 = str(cik).zfill(10)
    data = _get(sess, SUBMISSIONS_URL.format(cik10=cik10)).json()
    recent = data["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            return {
                "accession": recent["accessionNumber"][i],
                "primary_doc": recent["primaryDocument"][i],
                "report_date": recent["reportDate"][i],
                "filing_date": recent["filingDate"][i],
            }
    return None


def fetch_ticker(sess: requests.Session, ticker: str, lookup: dict[str, dict], out_dir: Path) -> bool:
    ticker = ticker.upper()
    if ticker not in lookup:
        print(f"  ✗ {ticker}: not found in SEC ticker list")
        return False
    cik, company = lookup[ticker]["cik"], lookup[ticker]["title"]

    filing = _latest_10k(sess, cik)
    if not filing:
        print(f"  ✗ {ticker}: no 10-K found")
        return False

    fy = filing["report_date"][:4]
    acc_nodash = filing["accession"].replace("-", "")
    url = DOC_URL.format(cik=cik, acc_nodash=acc_nodash, doc=filing["primary_doc"])

    print(f"  • {ticker} ({company}) FY{fy} — downloading {filing['primary_doc']}")
    html = _get(sess, url).text

    # strip HTML -> clean text (reuse the same cleaner the loaders use)
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = clean_text(soup.get_text("\n"))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{ticker}_{fy}_10K"
    (out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "company": company,
                "ticker": ticker,
                "fiscal_year": fy,
                "filing_type": "10-K",
                "source_url": url,
                "accession": filing["accession"],
                "filing_date": filing["filing_date"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"    ✓ saved {stem}.txt ({len(text):,} chars)")
    return True


def main() -> None:
    tickers = [t.upper() for t in sys.argv[1:]] or settings.tickers
    print(f"📥 Fetching latest 10-K for: {', '.join(tickers)}")
    print(f"   User-Agent: {settings.edgar_user_agent}")
    sess = _session()
    lookup = _ticker_to_cik(sess)

    ok = 0
    for t in tickers:
        try:
            ok += fetch_ticker(sess, t, lookup, settings.corpus_path)
        except requests.HTTPError as exc:
            print(f"  ✗ {t}: HTTP error {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {t}: {exc}")

    print(f"\n✅ Done. {ok}/{len(tickers)} filings saved to {settings.corpus_path}")


if __name__ == "__main__":
    main()
