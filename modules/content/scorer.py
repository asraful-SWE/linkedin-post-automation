"""
Content Scorer - Local heuristic and AI-based scoring for LinkedIn posts.

Evaluates four quality dimensions:
  - hook_strength       : How compelling the opening line is
  - clarity             : Readability and structural coherence
  - engagement_potential: Likelihood of likes, comments, and shares
  - goal_alignment      : How well the post serves its stated goal

Fast local scoring is the default (zero API cost); AI scoring is
available for deeper analysis with automatic fallback to local.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ContentScore:
    """Immutable quality report for a single LinkedIn post."""

    hook_strength: float  # 0-10: how strong is the opening line
    clarity: float  # 0-10: is it clear and readable
    engagement_potential: float  # 0-10: likely to get engagement
    overall_score: float  # weighted average of all four dimensions
    feedback: List[str]  # specific, actionable feedback points
    passed: bool  # True if overall_score >= scorer threshold
    goal_alignment: float  # 0-10: how well it matches the post goal

    def __str__(self) -> str:  # pragma: no cover
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        return (
            f"ContentScore [{status}] overall={self.overall_score:.2f} | "
            f"hook={self.hook_strength:.1f} clarity={self.clarity:.1f} "
            f"engagement={self.engagement_potential:.1f} goal={self.goal_alignment:.1f}"
        )


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class ContentScorer:
    """
    Scores LinkedIn post content using local heuristics or OpenAI analysis.

    Local scoring is fast, free, and deterministic — preferred for tight
    generation loops.  AI scoring provides nuanced feedback and is used for
    final quality gates or manual review workflows.

    Usage
    -----
    scorer = ContentScorer(threshold=6.0)

    # Fast, no API call
    score = scorer.score_post_locally(content, goal="educational")

    # Deep analysis with GPT fallback
    score = scorer.score_post_with_ai(content, goal="viral", openai_provider=provider)
    """

    # ------------------------------------------------------------------
    # Class-level keyword banks
    # ------------------------------------------------------------------

    # Strong opener words that signal a compelling hook (Bengali/Banglish)
    STRONG_OPENER_WORDS: List[str] = [
        "কখনো",
        "সত্যি",
        "আপনি কি",
        "ধরেন",
        "মনে করুন",
        "একটু",
        "এক",
        "আমি",
    ]

    # Learning / educational signal words
    EDUCATIONAL_KEYWORDS: List[str] = [
        "শেখা",
        "শিখুন",
        "শিখেছি",
        "শেখার",
        "শিখতে",
        "learn",
        "learning",
        "learned",
        "tips",
        "tip",
        "how to",
        "কিভাবে",
        "গাইড",
        "guide",
        "tutorial",
        "step",
        "ধাপ",
    ]

    # Authority / experience signal words
    AUTHORITY_KEYWORDS: List[str] = [
        "বছর",
        "year",
        "years",
        "experience",
        "অভিজ্ঞতা",
        "industry",
        "career",
        "ক্যারিয়ার",
        "professional",
        "expert",
        "সিনিয়র",
        "senior",
        "lead",
        "লিড",
    ]

    # Opinion / controversy signal words (good for viral posts)
    OPINION_KEYWORDS: List[str] = [
        "মনে হয়",
        "বিশ্বাস করি",
        "আসলে",
        "সত্যি কথা",
        "honestly",
        "controversial",
        "unpopular opinion",
        "disagree",
        "ভুল",
        "myth",
        "reality check",
        "real talk",
        "সত্যি বলতে",
        "আসল কথা",
    ]

    # Professional / technical terminology (authority posts)
    TECH_TERMS: List[str] = [
        "scalability",
        "architecture",
        "infrastructure",
        "strategy",
        "framework",
        "methodology",
        "performance",
        "optimization",
        "স্কেলেবিলিটি",
        "আর্কিটেকচার",
        "ইন্ডাস্ট্রি",
        "system design",
        "best practice",
        "roadmap",
    ]

    # Practical example signal words
    EXAMPLE_KEYWORDS: List[str] = [
        "উদাহরণ",
        "example",
        "যেমন",
        "like",
        "such as",
        "for instance",
        "ধরুন",
        "মনে করুন",
        "suppose",
        "e.g.",
    ]

    # CTA / discussion invitation patterns (checked against last line)
    CTA_PATTERNS: List[str] = [
        "জানান",
        "বলুন",
        "comment",
        "share",
        "like",
        "কি মনে করেন",
        "আপনার মতামত",
        "আপনি কি",
        "আলোচনা করুন",
        "discuss",
        "tag",
        "follow",
        "রিপ্লাই",
        "reply",
        "আপনার experience",
        "চিন্তা করুন",
    ]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, threshold: float = 6.0) -> None:
        """
        Args:
            threshold: Minimum overall_score for a post to be considered
                       'passed'.  Defaults to 6.0 (out of 10).
        """
        if not (0.0 <= threshold <= 10.0):
            raise ValueError(f"threshold must be between 0 and 10, got {threshold}")
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Static / private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cap(value: float, low: float = 0.0, high: float = 10.0) -> float:
        """Clamp value to [low, high]."""
        return max(low, min(high, value))

    @staticmethod
    def _count_words(text: str) -> int:
        """Simple whitespace-based word count."""
        return len(text.split())

    @staticmethod
    def _count_emojis(text: str) -> int:
        """Count Unicode emoji characters in *text*."""
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # flags
            "\U00002600-\U000027bf"  # miscellaneous symbols
            "\U0001f900-\U0001f9ff"  # supplemental symbols & pictographs
            "\U00002702-\U000027b0"  # dingbats
            "]+",
            flags=re.UNICODE,
        )
        return len(emoji_pattern.findall(text))

    @staticmethod
    def _get_paragraphs(content: str) -> List[str]:
        """Split on one or more blank lines; return non-empty paragraphs."""
        return [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]

    @staticmethod
    def _has_numbered_or_bullet_list(content: str) -> bool:
        """Return True if the post contains a numbered list or bullet list."""
        numbered = re.search(r"(?m)^\s*\d+[.।)\-]\s+\S", content)
        bullet = re.search(r"(?m)^\s*[-*•✓➤]\s+\S", content)
        return bool(numbered or bullet)

    @staticmethod
    def _get_last_non_empty_line(content: str) -> str:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        return lines[-1] if lines else ""

    def _ends_with_cta_or_question(self, content: str) -> bool:
        """Check whether the closing line is a question or a clear CTA."""
        last = self._get_last_non_empty_line(content)
        if not last:
            return False
        if last.endswith("?"):
            return True
        last_lower = last.lower()
        return any(pattern in last_lower for pattern in self.CTA_PATTERNS)

    # ------------------------------------------------------------------
    # Dimension scorers — each returns (score: float, feedback: List[str])
    # ------------------------------------------------------------------

    def _score_hook(self, content: str) -> Tuple[float, List[str]]:
        """Score the opening line (hook) of the post.

        Max achievable raw points: 3+2+1+2 = 8  (ceiling-capped to 10).
        Penalty:  -2 for excessively long first line.
        """
        score: float = 0.0
        feedback: List[str] = []

        # Isolate first non-empty line
        first_line = next(
            (line.strip() for line in content.splitlines() if line.strip()), ""
        )

        if not first_line:
            return 0.0, ["Hook is missing — post starts with an empty line."]

        length = len(first_line)

        # Ideal length bracket
        if 10 <= length <= 80:
            score += 3
        elif length > 120:
            score -= 2
            feedback.append(
                "Opening line is too long (>120 chars). Shorten it for a stronger hook."
            )

        # Question hook
        if first_line.rstrip().endswith("?"):
            score += 2

        # Contains a number (adds specificity)
        if re.search(r"\d", first_line):
            score += 1

        # Uses a strong Bengali/Banglish opener word
        first_lower = first_line.lower()
        if any(opener.lower() in first_lower for opener in self.STRONG_OPENER_WORDS):
            score += 2

        if score < 4:
            feedback.append(
                "Hook is weak. Try opening with a question, a surprising number, "
                "or a bold statement to stop the scroll."
            )

        return self._cap(score), feedback

    def _score_clarity(self, content: str) -> Tuple[float, List[str]]:
        """Score readability and structural coherence.

        Max raw points: 3+2+2+2+1 = 10.
        """
        score: float = 0.0
        feedback: List[str] = []

        paragraphs = self._get_paragraphs(content)
        word_count = self._count_words(content)

        # Average paragraph word count
        if paragraphs:
            avg_para_words = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
            if avg_para_words < 50:
                score += 3
            else:
                feedback.append(
                    "Paragraphs are too long on average. Break them into shorter "
                    "chunks (< 50 words each) for easier LinkedIn scanning."
                )

        # Minimum paragraph count
        if len(paragraphs) >= 3:
            score += 2
        else:
            feedback.append(
                f"Post has only {len(paragraphs)} paragraph(s). "
                "Use at least 3 paragraphs for better visual structure."
            )

        # No single paragraph should exceed 100 words
        long_paras = [p for p in paragraphs if len(p.split()) > 100]
        if not long_paras:
            score += 2
        else:
            feedback.append(
                f"{len(long_paras)} paragraph(s) exceed 100 words. "
                "Split them for readability."
            )

        # Proper blank-line spacing: at least 2 paragraph-break occurrences
        double_newline_count = len(re.findall(r"\n\s*\n", content))
        if double_newline_count >= 2:
            score += 2
        else:
            feedback.append(
                "Add blank lines between paragraphs — LinkedIn renders spacing "
                "and it dramatically improves readability."
            )

        # Overall length in the sweet spot
        if 100 <= word_count <= 500:
            score += 1
        elif word_count < 100:
            feedback.append(
                f"Post is very short ({word_count} words). Add more substance "
                "so readers get genuine value."
            )
        else:
            feedback.append(
                f"Post is long ({word_count} words). LinkedIn readers skim — "
                "consider trimming to under 500 words."
            )

        return self._cap(score), feedback

    def _score_engagement(self, content: str) -> Tuple[float, List[str]]:
        """Score the engagement potential of the post.

        Max raw points: 2+2+1+1+2+2 = 10.
        """
        score: float = 0.0
        feedback: List[str] = []

        word_count = self._count_words(content)
        emoji_count = self._count_emojis(content)

        # Contains at least one question
        if "?" in content:
            score += 2
        else:
            feedback.append(
                "Add at least one question to invite readers to engage in the comments."
            )

        # Addresses the reader directly
        if "আপনি" in content or "আপনার" in content:
            score += 2
        else:
            feedback.append(
                "Address your reader directly with 'আপনি' or 'আপনার' to make "
                "the post feel personal and relevant."
            )

        # Emoji count: 1-3 is optimal for LinkedIn
        if 1 <= emoji_count <= 3:
            score += 1
        elif emoji_count > 3:
            feedback.append(
                f"Found {emoji_count} emojis — that's too many. Keep it to 1–3 "
                "for a professional but human feel."
            )

        # Structured list helps scannability
        if self._has_numbered_or_bullet_list(content):
            score += 1

        # Optimal LinkedIn word count for engagement
        if 150 <= word_count <= 350:
            score += 2
        else:
            feedback.append(
                f"Word count is {word_count}. "
                "LinkedIn engagement peaks at 150–350 words — aim for that range."
            )

        # Ends with a CTA or a question (drives comments)
        if self._ends_with_cta_or_question(content):
            score += 2
        else:
            feedback.append(
                "End the post with a question or call-to-action to boost comment count."
            )

        return self._cap(score), feedback

    def _score_goal_alignment(self, content: str, goal: str) -> Tuple[float, List[str]]:
        """Score how well the post serves its stated communication goal.

        Max raw points for each goal: 3+3+2+2 = 10.
        Unknown goals fall back to a generic engagement-oriented rubric.
        """
        score: float = 0.0
        feedback: List[str] = []

        content_lower = content.lower()
        word_count = self._count_words(content)
        paragraphs = self._get_paragraphs(content)
        has_list = self._has_numbered_or_bullet_list(content)

        # ── Educational ──────────────────────────────────────────────
        if goal == "educational":
            # Numbered / bullet lists are the backbone of educational posts
            if has_list:
                score += 3
            else:
                feedback.append(
                    "Educational posts benefit greatly from numbered lists or "
                    "bullet points — they make lessons easy to scan and save."
                )

            # Learning-oriented language
            if any(kw in content_lower for kw in self.EDUCATIONAL_KEYWORDS):
                score += 3
            else:
                feedback.append(
                    "Include learning-oriented language like 'শেখা', 'tips', "
                    "or 'how to' to frame the post as educational."
                )

            # Structured format: list OR multiple paragraphs acting as sections
            if has_list or len(paragraphs) >= 4:
                score += 2
            else:
                feedback.append(
                    "Structure the post with clear sections or numbered steps "
                    "so readers can follow along easily."
                )

            # Practical examples ground the theory
            if any(kw in content_lower for kw in self.EXAMPLE_KEYWORDS):
                score += 2
            else:
                feedback.append(
                    "Add a concrete example ('যেমন', 'for instance', 'ধরুন') "
                    "to make the educational content tangible."
                )

        # ── Viral ────────────────────────────────────────────────────
        elif goal == "viral":
            # Short, punchy paragraphs are essential for shareable content
            if paragraphs:
                avg_words = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
                if avg_words < 30:
                    score += 3
                else:
                    feedback.append(
                        "Viral posts need short, punchy paragraphs "
                        f"(avg < 30 words). Current average: {avg_words:.0f} words."
                    )

            # Strong hook is non-negotiable for virality
            hook_score, _ = self._score_hook(content)
            if hook_score >= 6:
                score += 3
            else:
                feedback.append(
                    "A viral post lives or dies by its opening line. "
                    "Make the hook irresistible — bold opinion, surprising stat, "
                    "or a provocative question."
                )

            # Bold opinion or controversy drives shares
            if any(kw in content_lower for kw in self.OPINION_KEYWORDS):
                score += 2
            else:
                feedback.append(
                    "Add a bold opinion or surprising insight. "
                    "Bland posts don't get shared — take a clear stance."
                )

            # Question at the end sparks debate
            if self._ends_with_cta_or_question(content):
                score += 2
            else:
                feedback.append(
                    "End with a question that sparks debate. "
                    "Controversial questions drive comment threads and organic reach."
                )

        # ── Authority ────────────────────────────────────────────────
        elif goal == "authority":
            # Specific numbers make claims credible
            if re.search(r"\b\d+\b", content):
                score += 3
            else:
                feedback.append(
                    "Include specific numbers or statistics — "
                    "authority posts need quantifiable claims to be credible."
                )

            # Experience / tenure references
            if any(kw in content_lower for kw in self.AUTHORITY_KEYWORDS):
                score += 3
            else:
                feedback.append(
                    "Reference your experience, years in the field, or industry "
                    "context to establish credibility."
                )

            # Professional / technical vocabulary
            if any(term in content_lower for term in self.TECH_TERMS):
                score += 2
            else:
                feedback.append(
                    "Use professional or technical terminology appropriate to "
                    "your domain to signal deep expertise."
                )

            # Authority posts need depth
            if word_count > 200:
                score += 2
            else:
                feedback.append(
                    f"Post is only {word_count} words. Authority posts need "
                    "depth — aim for at least 200 words to substantiate your position."
                )

        # ── Story ────────────────────────────────────────────────────
        elif goal == "story":
            # Personal pronouns indicate a first-person narrative
            personal_pronouns = ["আমি", "আমার", "আমরা", "আমাদের"]
            if any(p in content for p in personal_pronouns):
                score += 3
            else:
                feedback.append(
                    "Story posts need a first-person voice. "
                    "Use 'আমি' and 'আমার' to make it personal and relatable."
                )

            # Narrative arc: beginning, middle, end ≈ multiple paragraphs
            if len(paragraphs) >= 4:
                score += 3
            else:
                feedback.append(
                    "A compelling story needs structure: setup, conflict, and "
                    "resolution. Use 4+ paragraphs to build a proper arc."
                )

            # Emotional or relatable language
            emotional_kw = [
                "শিখেছি",
                "বুঝেছি",
                "অনুভব",
                "feel",
                "realised",
                "realized",
                "ভুল",
                "failure",
                "success",
                "সফল",
                "কঠিন",
                "struggle",
            ]
            if any(kw in content_lower for kw in emotional_kw):
                score += 2
            else:
                feedback.append(
                    "Add emotional or reflective language to help readers connect "
                    "with the story on a personal level."
                )

            # Lesson or takeaway at the end
            takeaway_kw = ["শেখা", "lesson", "takeaway", "মনে রাখবেন", "বুঝলাম"]
            if any(kw in content_lower for kw in takeaway_kw):
                score += 2
            else:
                feedback.append(
                    "End the story with a clear lesson or takeaway — "
                    "readers should walk away with something actionable."
                )

        # ── Engagement ───────────────────────────────────────────────
        elif goal == "engagement":
            # A question in the hook immediately invites interaction
            first_line = next(
                (ln.strip() for ln in content.splitlines() if ln.strip()), ""
            )
            if first_line.endswith("?") or "?" in first_line:
                score += 3
            else:
                feedback.append(
                    "Start with a thought-provoking question to grab attention "
                    "and immediately signal that you want discussion."
                )

            # Multiple questions throughout amplify engagement
            question_count = content.count("?")
            if question_count >= 2:
                score += 3
            else:
                feedback.append(
                    "Use 2+ questions throughout the post — each one is an "
                    "invitation to comment."
                )

            # Relatable or opinion-driven content drives comments
            if any(kw in content_lower for kw in self.OPINION_KEYWORDS):
                score += 2
            else:
                feedback.append(
                    "Share a relatable experience or a controversial opinion — "
                    "content that provokes a reaction drives discussion."
                )

            # Explicit CTA at the end
            if self._ends_with_cta_or_question(content):
                score += 2
            else:
                feedback.append(
                    "Close with an explicit call-to-action: ask readers to share "
                    "their experience or opinion in the comments."
                )

        # ── Fallback / unknown goal ───────────────────────────────────
        else:
            logger.debug("Unknown goal '%s' — applying generic engagement rubric", goal)
            if "?" in content:
                score += 3
            if word_count >= 150:
                score += 3
            if self._count_emojis(content) <= 3:
                score += 2
            if len(paragraphs) >= 3:
                score += 2

        return self._cap(score), feedback

    # ------------------------------------------------------------------
    # Public scoring API
    # ------------------------------------------------------------------

    def score_post_locally(
        self, content: str, goal: str = "educational"
    ) -> ContentScore:
        """
        Score a LinkedIn post using purely local heuristics.

        Zero API cost, deterministic, suitable for tight generation loops.

        Weights
        -------
        hook_strength       30 %
        clarity             25 %
        engagement_potential 30 %
        goal_alignment      15 %

        Args:
            content : Raw post text to evaluate.
            goal    : Intended communication goal.
                      Supported: 'educational', 'viral', 'authority',
                                 'story', 'engagement'.

        Returns:
            Populated :class:`ContentScore` instance.
        """
        if not content or not content.strip():
            logger.warning("score_post_locally called with empty content")
            return ContentScore(
                hook_strength=0.0,
                clarity=0.0,
                engagement_potential=0.0,
                overall_score=0.0,
                feedback=["Post content is empty."],
                passed=False,
                goal_alignment=0.0,
            )

        hook_score, hook_fb = self._score_hook(content)
        clarity_score, clarity_fb = self._score_clarity(content)
        engage_score, engage_fb = self._score_engagement(content)
        goal_score, goal_fb = self._score_goal_alignment(content, goal)

        overall = self._cap(
            hook_score * 0.30
            + clarity_score * 0.25
            + engage_score * 0.30
            + goal_score * 0.15
        )

        # Deduplicate feedback while preserving order
        seen: set = set()
        all_feedback: List[str] = []
        for item in hook_fb + clarity_fb + engage_fb + goal_fb:
            if item not in seen:
                seen.add(item)
                all_feedback.append(item)

        logger.debug(
            "Local score | hook=%.1f | clarity=%.1f | engagement=%.1f "
            "| goal_alignment=%.1f | overall=%.2f | goal=%s | passed=%s",
            hook_score,
            clarity_score,
            engage_score,
            goal_score,
            overall,
            goal,
            overall >= self.threshold,
        )

        return ContentScore(
            hook_strength=round(hook_score, 2),
            clarity=round(clarity_score, 2),
            engagement_potential=round(engage_score, 2),
            overall_score=round(overall, 2),
            feedback=all_feedback,
            passed=overall >= self.threshold,
            goal_alignment=round(goal_score, 2),
        )

    def score_post_with_ai(
        self,
        content: str,
        goal: str,
        openai_provider,
    ) -> ContentScore:
        """
        Score a LinkedIn post using OpenAI for nuanced, context-aware analysis.

        The AI evaluates the post on the same four dimensions as local scoring
        and returns a JSON payload that is parsed into a :class:`ContentScore`.
        Automatically falls back to :meth:`score_post_locally` if the API call
        fails or the response cannot be parsed.

        Args:
            content         : Raw post text to evaluate.
            goal            : Intended communication goal (e.g. 'viral').
            openai_provider : An :class:`~ai.openai_provider.OpenAIProvider`
                              instance used for the API call.

        Returns:
            Populated :class:`ContentScore` instance (from AI or local fallback).
        """
        if not content or not content.strip():
            return self.score_post_locally(content, goal)

        prompt = f"""Analyze the following LinkedIn post and return a JSON scoring object.

POST CONTENT:
\"\"\"{content}\"\"\"

POST GOAL: {goal}

Score each dimension strictly from 0 to 10 (decimals are fine).

Return ONLY valid JSON — no markdown, no explanation, no extra text:
{{
  "hook_strength": <float 0-10>,
  "clarity": <float 0-10>,
  "engagement_potential": <float 0-10>,
  "goal_alignment": <float 0-10>,
  "feedback": [
    "<specific, actionable feedback point 1>",
    "<specific, actionable feedback point 2>",
    "<specific, actionable feedback point 3>"
  ]
}}

Scoring guidelines:
- hook_strength      : Is the opening line compelling enough to stop the scroll?
- clarity            : Is the post well-structured, easy to scan, free of jargon overload?
- engagement_potential: How likely is this to earn likes, comments, and shares on LinkedIn?
- goal_alignment     : How effectively does the post serve its stated goal ({goal})?
- feedback           : 2–5 specific improvements. Be direct and actionable, not generic."""

        system_message = (
            "You are a LinkedIn content strategist specialising in Bengali and Banglish "
            "posts for Bangladeshi tech professionals. "
            "You understand tone, cultural nuance, and what drives engagement in this niche. "
            "Respond with valid JSON only — no preamble, no markdown fences."
        )

        try:
            logger.debug(
                "Requesting AI content score | goal=%s | word_count=%d",
                goal,
                self._count_words(content),
            )

            response_text = openai_provider.generate_completion(
                prompt=prompt,
                system_message=system_message,
                max_tokens=400,
                temperature=0.2,  # low temperature → consistent, reproducible scores
            )

            # Strip markdown code fences the model might add despite instructions
            cleaned = re.sub(r"```(?:json)?\s*", "", response_text, flags=re.IGNORECASE)
            cleaned = cleaned.replace("```", "").strip()

            data: dict = json.loads(cleaned)

            hook = self._cap(float(data.get("hook_strength", 5.0)))
            clarity = self._cap(float(data.get("clarity", 5.0)))
            engagement = self._cap(float(data.get("engagement_potential", 5.0)))
            goal_align = self._cap(float(data.get("goal_alignment", 5.0)))

            raw_feedback = data.get("feedback", [])
            if isinstance(raw_feedback, str):
                # Model occasionally returns a single string instead of a list
                feedback_list: List[str] = [raw_feedback]
            elif isinstance(raw_feedback, list):
                feedback_list = [str(item) for item in raw_feedback if item]
            else:
                feedback_list = []

            overall = self._cap(
                hook * 0.30 + clarity * 0.25 + engagement * 0.30 + goal_align * 0.15
            )

            logger.info(
                "AI score | hook=%.1f | clarity=%.1f | engagement=%.1f "
                "| goal_alignment=%.1f | overall=%.2f | passed=%s",
                hook,
                clarity,
                engagement,
                goal_align,
                overall,
                overall >= self.threshold,
            )

            return ContentScore(
                hook_strength=round(hook, 2),
                clarity=round(clarity, 2),
                engagement_potential=round(engagement, 2),
                overall_score=round(overall, 2),
                feedback=feedback_list,
                passed=overall >= self.threshold,
                goal_alignment=round(goal_align, 2),
            )

        except json.JSONDecodeError as exc:
            logger.warning(
                "AI score response was not valid JSON (%s) — "
                "falling back to local scoring.",
                exc,
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "AI score response parsing failed (%s) — "
                "falling back to local scoring.",
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "AI scoring API call failed (%s) — falling back to local scoring.",
                exc,
            )

        return self.score_post_locally(content, goal)
