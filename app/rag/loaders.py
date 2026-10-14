"""Document loaders: curated seed corpus (markdown + front matter) and a polite web scraper."""

from __future__ import annotations

import re
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlparse

import httpx
import yaml
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from app.core.logging import get_logger

log = get_logger(__name__)

USER_AGENT = "GetnetSupportKB/1.0 (+https://github.com/Bepaiva/bbspj-multi-agent-support)"
_FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def load_seed_corpus(directory: Path) -> list[Document]:
    """Load curated markdown files. Front matter provides title and source URLs."""
    docs: list[Document] = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        meta: dict = {}
        if m := _FRONT_MATTER.match(text):
            meta = yaml.safe_load(m.group(1)) or {}
            text = text[m.end():]
        sources = meta.get("sources") or []
        docs.append(Document(
            page_content=text.strip(),
            metadata={
                "title": meta.get("title", path.stem),
                "source": sources[0] if sources else f"seed://{path.name}",
                "all_sources": ", ".join(sources),
                "origin": "seed",
                "doc_id": f"seed:{path.stem}",
                "last_reviewed": str(meta.get("last_reviewed", "")),
            },
        ))
    log.info("ingest.seed_loaded", documents=len(docs))
    return docs


class WebScraper:
    """Fetches pages, honours robots.txt and extracts the main readable text."""

    def __init__(self, timeout: float = 15.0) -> None:
        self.client = httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT})
        self._robots: dict[str, robotparser.RobotFileParser | None] = {}

    def _allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robots:
            rp = robotparser.RobotFileParser()
            try:
                resp = self.client.get(f"{base}/robots.txt")
                rp.parse(resp.text.splitlines() if resp.status_code == 200 else [])
                self._robots[base] = rp
            except httpx.HTTPError:
                self._robots[base] = None  # robots unreachable -> be permissive but log
        rp = self._robots[base]
        return True if rp is None else rp.can_fetch(USER_AGENT, url)

    @staticmethod
    def extract_text(html: str) -> tuple[str, str]:
        soup = BeautifulSoup(html, "html.parser")
        title = (soup.title.string or "").strip() if soup.title else ""
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe"]):
            tag.decompose()
        for sel in ("[class*=cookie]", "[id*=cookie]", "[class*=menu]", "[class*=breadcrumb]"):
            for tag in soup.select(sel):
                tag.decompose()
        main = soup.find("main") or soup.body or soup
        lines = [ln.strip() for ln in main.get_text("\n").splitlines()]
        text = "\n".join(ln for ln in lines if len(ln) > 2)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return title, text

    def fetch(self, url: str) -> Document | None:
        if not self._allowed(url):
            log.info("ingest.robots_disallowed", url=url)
            return None
        try:
            resp = self.client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("ingest.fetch_failed", url=url, error=str(exc))
            return None
        title, text = self.extract_text(resp.text)
        if len(text) < 200:
            log.info("ingest.page_too_short", url=url, chars=len(text))
            return None
        return Document(page_content=text, metadata={
            "title": title or url, "source": url, "all_sources": url, "origin": "web", "doc_id": f"web:{url}",
        })

    def fetch_many(self, urls: list[str]) -> list[Document]:
        docs = [d for url in urls if (d := self.fetch(url))]
        log.info("ingest.web_loaded", requested=len(urls), loaded=len(docs))
        return docs
