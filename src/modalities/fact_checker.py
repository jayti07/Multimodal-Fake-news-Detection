"""
Fact Verification RAG Module
Includes Domain Whitelisting, Publisher Authority Checking, Refutation Stance Detection,
and Automatic Speech Query Extraction (cleaning radio intros & speech filler words).
"""

import os
import re
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote, urlparse

# Trusted High-Authority News Domains & Fact-Checkers
TRUSTED_NEWS_DOMAINS = [
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "indianexpress.com", "ndtv.com", "timesofindia.indiatimes.com",
    "thehindu.com", "hindustantimes.com", "news18.com", "indiatoday.in",
    "business-standard.com", "financialexpress.com", "economic-times", "pib.gov.in",
    "factcheck.org", "snopes.com", "altnews.in", "boomlive.in",
    "poynter.org", "factly.in", "fullfact.org", "politifact.com",
    "theguardian.com", "nytimes.com", "washingtonpost.com"
]


class FactCheckerRAG:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_FACT_CHECK_API_KEY", "")

    def extract_core_query(self, raw_text: str) -> str:
        """
        Cleans long transcribed audio text by removing radio intros, station tags,
        and speech filler phrases to extract the core news claim keywords for search.
        """
        if not raw_text:
            return ""

        text = raw_text.strip()
        
        # 1. Remove common radio / podcast intro filler phrases
        filler_patterns = [
            r"this is \w+", r"welcome to \w+", r"weekly program", r"current affairs",
            r"now we bring \w+", r"your discussion on", r"discussion on",
            r"the participants of \w+", r"economic analysis", r"and the moderator is",
            r"good morning", r"good evening", r"broadcast", r"listening to"
        ]
        
        cleaned = text
        for pattern in filler_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # 2. Clean extra whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # 3. If cleaned string has meaningful content, return top keywords (up to 8 words)
        words = cleaned.split()
        if len(words) >= 2:
            return " ".join(words[:8])

        # Fallback to first 8 words of original text if cleaning removed everything
        return " ".join(text.split()[:8])

    def is_trusted_domain(self, url: str, publisher_name: str = "") -> bool:
        """
        Checks whether the article URL or publisher belongs to a trusted news authority.
        """
        if not url or url == "#":
            return False

        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]

            for trusted in TRUSTED_NEWS_DOMAINS:
                if trusted in domain or domain.endswith(trusted):
                    return True
        except Exception:
            pass

        pub_lower = publisher_name.lower()
        trusted_publishers = [
            "reuters", "associated press", "bbc", "indian express", "ndtv",
            "times of india", "the hindu", "hindustan times", "business standard",
            "financial express", "economic times", "pib", "fact check",
            "snopes", "alt news", "boom live", "politifact"
        ]
        return any(tp in pub_lower for tp in trusted_publishers)

    def search_google_news_rss(self, claim_text: str) -> dict:
        """
        Searches Google News RSS feed for real-time news articles from verified sources.
        """
        try:
            # Clean spoken transcript into a focused core search query
            search_query = self.extract_core_query(claim_text)
            print(f"[RAG Query Extractor] Original: '{claim_text}' -> Search Query: '{search_query}'")

            encoded_query = quote(search_query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(rss_url, headers=headers, timeout=5)

            if response.status_code == 200:
                root = ET.fromstring(response.content)
                items = root.findall(".//item")

                for item in items[:5]:
                    title = item.find("title").text if item.find("title") is not None else claim_text
                    link = item.find("link").text if item.find("link") is not None else "#"
                    pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                    source_elem = item.find("source")
                    publisher = source_elem.text if source_elem is not None else "Unknown Source"

                    # 1. Verify Domain / Publisher Authority
                    trusted = self.is_trusted_domain(link, publisher)
                    if not trusted:
                        print(f"[RAG Ignored] Untrusted source domain: {link} ({publisher})")
                        continue

                    # 2. Detect if headline indicates a debunked / fake claim
                    title_lower = title.lower()
                    refutation_keywords = [
                        "fact check", "fake", "false", "hoax", "busted",
                        "debunked", "misleading", "rumor", "untrue", "myth"
                    ]
                    is_debunked = any(kw in title_lower for kw in refutation_keywords)

                    if is_debunked:
                        return {
                            "matched": True,
                            "is_real": False,
                            "is_trusted": True,
                            "title": title,
                            "publisher": publisher,
                            "rating": "Debunked / Fake Claim",
                            "url": link,
                            "pub_date": pub_date,
                            "search_query": search_query,
                            "source_type": "Verified Fact-Check Engine"
                        }

                    return {
                        "matched": True,
                        "is_real": True,
                        "is_trusted": True,
                        "title": title,
                        "publisher": publisher,
                        "rating": "Verified Published News Event",
                        "url": link,
                        "pub_date": pub_date,
                        "search_query": search_query,
                        "source_type": "Trusted News Publisher"
                    }
        except Exception as e:
            print(f"[Google News RSS Error] {e}")

        return {"matched": False}

    def verify_claim(self, claim_text: str) -> dict:
        """
        Retrieves candidate fact-checked claims or verified news articles.
        """
        if not claim_text or not claim_text.strip():
            return {"matched": False, "article": None}

        # 1. Google Fact Check Tools API (if API Key provided)
        if self.api_key:
            search_query = self.extract_core_query(claim_text)
            url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            params = {"query": search_query, "key": self.api_key}
            try:
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    claims = data.get("claims", [])
                    if claims:
                        top_claim = claims[0]
                        claim_review = top_claim.get("claimReview", [{}])[0]
                        return {
                            "matched": True,
                            "is_real": claim_review.get("textualRating", "").lower() in ["true", "correct"],
                            "is_trusted": True,
                            "title": top_claim.get("text", ""),
                            "claimant": top_claim.get("claimant", "Unknown"),
                            "rating": claim_review.get("textualRating", "Unverified"),
                            "publisher": claim_review.get("publisher", {}).get("name", "Verified Source"),
                            "url": claim_review.get("url", "#"),
                            "source_type": "Google Fact Check API"
                        }
            except Exception as e:
                print(f"[RAG API Warning] {e}")

        # 2. Live News Search with Query Extractor & Domain Whitelisting
        rss_result = self.search_google_news_rss(claim_text)
        if rss_result.get("matched"):
            return rss_result

        return {
            "matched": False,
            "message": "No verified match found from trusted news publishers.",
            "url": None
        }
