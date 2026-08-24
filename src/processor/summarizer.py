"""LLM-based content summarization for newsletter."""

import re
from typing import List, Dict
from src.llm.ollama_client import OllamaClient
from src.collector.rss_collector import Article


# The newsletter template renders LLM output as plain text (Jinja2 only
# HTML-escapes it, it does not interpret markdown), so any markdown emphasis
# the model adds — e.g. "**Salesforce Updates**" — shows up on the live page
# as literal asterisks instead of bold text. Strip common markdown emphasis
# syntax from every LLM response before it reaches the template so this can't
# happen even if a prompt tweak upstream doesn't fully stop the model from
# using it.
def strip_markdown_emphasis(text: str) -> str:
    """Remove markdown bold/italic markers, keeping the inner text."""
    if not text:
        return text
    # **bold** or __bold__ -> bold
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    # *italic* or _italic_ -> italic (single markers)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
    text = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"\1", text)
    return text


class Summarizer:
    """Summarizes articles using LLM."""

    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client

    def summarize_article(self, article: Article) -> str:
        """Summarize a single article."""
        prompt = f"""Summarize the following article in 2-3 sentences for a newsletter.
Keep it concise, informative, and engaging. Respond in plain text only, with no
markdown formatting (no **bold**, no _italics_, no headers or bullet points).

Title: {article.title}
Source: {article.source}
Content: {article.summary}

Summary:"""

        try:
            return strip_markdown_emphasis(self.llm.generate(prompt, temperature=0.5, max_tokens=200))
        except Exception:
            return article.summary[:300]

    def generate_section_summary(self, section_name: str, articles: List[Article]) -> str:
        """Generate a summary for a newsletter section."""
        articles_text = "\n".join(
            f"- {a.title}: {a.summary[:150]}" for a in articles[:5]
        )

        prompt = f"""Write a brief 2-3 sentence introduction for a newsletter section called "{section_name}".
The section contains these articles:
{articles_text}

Write an engaging introduction that highlights the key themes. Respond in plain
text only, with no markdown formatting (no **bold**, no _italics_, no headers):"""

        try:
            return strip_markdown_emphasis(self.llm.generate(prompt, temperature=0.7, max_tokens=150))
        except Exception:
            # Let the caller apply a meaningful per-section fallback.
            return ""

    def generate_title(self, articles: List[Article]) -> str:
        """Generate a catchy newsletter title."""
        headlines = "\n".join(f"- {a.title}" for a in articles[:10])

        prompt = f"""Generate a catchy, professional newsletter title for this week's
Salesforce AAA UVCE newsletter. These are the top stories:
{headlines}

Respond with the title only, in plain text with no markdown formatting
(no **bold**, no quotes, no trailing punctuation).

Title:"""

        try:
            return strip_markdown_emphasis(self.llm.generate(prompt, temperature=0.8, max_tokens=50))
        except Exception:
            return "Weekly Digest"
