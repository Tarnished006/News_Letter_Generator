"""Newsletter content generator using LLM."""

from typing import List, Dict
from src.llm.ollama_client import OllamaClient
from src.collector.rss_collector import Article
from src.processor.summarizer import Summarizer, strip_markdown_emphasis


class NewsletterGenerator:
    """Generates newsletter content using LLM."""

    def __init__(self, llm_client: OllamaClient, config: dict):
        self.llm = llm_client
        self.config = config
        self.summarizer = Summarizer(llm_client)

    def generate(self, categorized_articles: Dict[str, List[Article]]) -> Dict:
        """Generate complete newsletter content."""
        newsletter_config = self.config.get("newsletter", {})

        # Get all articles flat
        all_articles = []
        for articles in categorized_articles.values():
            all_articles.extend(articles)

        # Generate title
        title = self.summarizer.generate_title(all_articles)

        # Generate introduction
        intro = self._generate_introduction(all_articles)

        # Generate sections
        sections = []
        category_names = {
            "salesforce": "Salesforce Updates",
            "ai": "AI & Machine Learning",
            "tech": "Tech Industry News"
        }

        # How many stories each section should carry. Driven by config so the
        # digest length is fully controllable (see config/newsletter.yaml).
        max_per_section = newsletter_config.get("max_articles_per_section", 5)

        for category, articles in categorized_articles.items():
            if articles:
                section_name = category_names.get(category, category.title())
                section = self._generate_section(
                    section_name,
                    articles,
                    max_per_section
                )
                sections.append(section)

        # Generate conclusion
        conclusion = self._generate_conclusion(all_articles)

        return {
            "title": title.strip() if title else newsletter_config.get("name", "Weekly Digest"),
            "tagline": newsletter_config.get("tagline", "Your weekly dose of updates"),
            "organization": newsletter_config.get("organization", "Salesforce AAA UVCE"),
            "introduction": intro.strip(),
            "sections": sections,
            "conclusion": conclusion.strip()
        }

    def _generate_introduction(self, articles: List[Article]) -> str:
        """Generate newsletter introduction."""
        headlines = "\n".join(f"- {a.title}" for a in articles[:8])

        prompt = f"""Write a warm, engaging introduction (3-4 sentences) for the Salesforce AAA UVCE weekly newsletter.
This week's top stories include:
{headlines}

Make it professional but friendly. Mention key themes. Respond in plain text
only, with no markdown formatting (no **bold**, no _italics_, no headers):"""

        try:
            return strip_markdown_emphasis(self.llm.generate(prompt, temperature=0.7, max_tokens=200))
        except Exception:
            return "Welcome to this week's newsletter! Here are the latest updates from Salesforce and the AI world."

    def _generate_section(self, section_name: str, articles: List[Article],
                          max_per_section: int = 5) -> Dict:
        """Generate a newsletter section."""
        # Summarize up to the configured number of articles for this section
        # instead of a hard-coded cap.
        summarized = []
        for article in articles[:max_per_section]:
            summary = self.summarizer.summarize_article(article)
            summarized.append({
                "title": article.title,
                "link": article.link,
                "summary": summary.strip() if summary else article.summary[:200],
                "source": article.source,
                "image_url": article.image_url
            })

        # Generate section intro
        intro = self.summarizer.generate_section_summary(section_name, articles)

        # Meaningful fallbacks so a failed LLM call never leaves the section
        # with a placeholder-style description like "Latest updates in ...".
        section_fallbacks = {
            "Salesforce Updates": "Handpicked insights on the latest Salesforce releases, platform updates, and trailblazer stories.",
            "AI & Machine Learning": "A curated look at AI research breakthroughs, new model releases, and machine learning in practice.",
            "Tech Industry News": "The week's most relevant developments shaping the broader technology landscape.",
        }
        fallback = section_fallbacks.get(
            section_name,
            "A curated selection of stories handpicked for the UVCE community."
        )

        return {
            "name": section_name,
            "introduction": intro.strip() if intro else fallback,
            "articles": summarized
        }

    def _generate_conclusion(self, articles: List[Article]) -> str:
        """Generate newsletter conclusion."""
        prompt = f"""Write a brief conclusion (2-3 sentences) for the Salesforce AAA UVCE weekly newsletter.
End with a call to action encouraging readers to stay connected and share feedback.
Make it warm and professional. Respond in plain text only, with no markdown
formatting (no **bold**, no _italics_, no headers):"""

        try:
            return strip_markdown_emphasis(self.llm.generate(prompt, temperature=0.7, max_tokens=150))
        except Exception:
            return "Thanks for reading! Stay connected with Salesforce AAA UVCE for more updates."
