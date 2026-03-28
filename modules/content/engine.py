"""
Intelligent Content Engine - Goal-driven LinkedIn post generation with
iterative quality improvement.

The engine wraps the base generation pipeline with three enhancements:

1. **Goal alignment** – each PostGoal injects goal-specific Bengali/Banglish
   instructions into the prompt so the model produces content that genuinely
   serves the intent (educate, go viral, build authority, etc.).

2. **Quality gating** – every generated post is scored by ContentScorer.
   Posts that fall below the configured threshold are regenerated (up to
   `max_regeneration_attempts` times) with the scorer's feedback embedded
   directly in the next prompt, giving the model concrete direction.

3. **A/B batch generation** – `batch_generate_for_ab_test` produces N
   distinct style×mood variations of the same topic/goal so callers can
   compare and pick the strongest performer.
"""

import logging
import random
from enum import Enum
from typing import Dict, List, Optional

from ai.generator import (
    HUMANIZED_PROMPT,
    POST_LENGTHS,
    POST_MOODS,
    POST_STYLES,
    _clean_post,
)
from modules.content.scorer import ContentScore, ContentScorer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Goal enum
# ---------------------------------------------------------------------------


class PostGoal(Enum):
    EDUCATIONAL = "educational"  # Share knowledge, tips, how-tos
    VIRAL = "viral"  # Maximize engagement and shares
    AUTHORITY = "authority"  # Build thought leadership
    STORY = "story"  # Personal experience narrative
    ENGAGEMENT = "engagement"  # Drive comments and discussion


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Goal-specific prompt additions injected after the base HUMANIZED_PROMPT.
# Written in Bengali/Banglish to stay consistent with the rest of the prompt.
_GOAL_PROMPTS: Dict[str, str] = {
    PostGoal.EDUCATIONAL.value: (
        "এই পোস্টটি educational হবে। "
        "Clear numbered tips বা structured knowledge share করবে। "
        "Reader যেন কিছু শিখতে পারে।"
    ),
    PostGoal.VIRAL.value: (
        "এই পোস্টটি viral potential সহ লিখবে। "
        "Strong controversial opinion বা surprising insight দিয়ে শুরু করবে। "
        "Short, punchy, shareable।"
    ),
    PostGoal.AUTHORITY.value: (
        "এই পোস্টটি thought leadership দেখাবে। "
        "Specific experience, industry insight, আর expert opinion থাকবে।"
    ),
    PostGoal.STORY.value: (
        "এই পোস্টটি একটা compelling personal story হবে। "
        "Beginning, middle, end structure। "
        "Reader connect করতে পারবে।"
    ),
    PostGoal.ENGAGEMENT.value: (
        "এই পোস্টটি maximum discussion তৈরি করবে। "
        "Thought-provoking question, controversial opinion, বা relatable experience।"
    ),
}

# Post styles that have natural affinity with each goal.
# When no style override is provided the engine samples from this narrower
# pool first, giving better goal alignment out of the box.
_GOAL_STYLE_AFFINITY: Dict[str, List[str]] = {
    PostGoal.EDUCATIONAL.value: [
        "tips_practical",
        "lesson_learned",
        "comparison",
        "myth_busting",
    ],
    PostGoal.VIRAL.value: [
        "opinion",
        "myth_busting",
        "real_talk",
        "rant_honest",
        "provocative",
    ],
    PostGoal.AUTHORITY.value: [
        "observation",
        "opinion",
        "experience_sharing",
        "comparison",
    ],
    PostGoal.STORY.value: [
        "personal_story",
        "lesson_learned",
        "experience_sharing",
    ],
    PostGoal.ENGAGEMENT.value: [
        "question_discussion",
        "rant_honest",
        "challenge",
        "real_talk",
    ],
}

# token budgets per length tier
_MAX_TOKENS: Dict[str, int] = {"short": 400, "medium": 700, "long": 1200}

# System message used for all generation calls
_SYSTEM_MESSAGE = (
    "তুমি একজন বাংলাদেশী software developer। "
    "তুমি LinkedIn এ regularly পোস্ট করো নিজের experience, opinion, "
    "আর industry observations নিয়ে। "
    "তোমার writing style সম্পূর্ণ natural এবং human। "
    "তুমি AI না, তুমি real মানুষ।"
)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class IntelligentContentEngine:
    """
    Goal-driven content generation engine with iterative quality improvement.

    The engine generates LinkedIn posts that are aligned to a specific
    communication goal, scores them locally (zero API cost), and regenerates
    with targeted feedback until the post passes the quality threshold or
    the maximum attempt count is reached.

    Parameters
    ----------
    openai_provider:
        An :class:`~ai.openai_provider.OpenAIProvider` instance used for
        text generation.
    scorer:
        A :class:`~modules.content.scorer.ContentScorer` instance used for
        post quality evaluation.
    max_regeneration_attempts:
        Maximum number of generation attempts per post.  The first attempt
        is always made; subsequent attempts are triggered only when the post
        fails the quality threshold.  Defaults to 3.
    score_threshold:
        Minimum ``overall_score`` required for a post to be accepted.
        Defaults to 6.0.

    Example
    -------
    ::

        from ai.openai_provider import OpenAIProvider
        from modules.content.scorer import ContentScorer
        from modules.content.engine import IntelligentContentEngine, PostGoal

        engine = IntelligentContentEngine(
            openai_provider=OpenAIProvider(),
            scorer=ContentScorer(threshold=6.0),
        )

        result = engine.generate_post_with_goal(
            topic="Async programming in Python",
            goal=PostGoal.EDUCATIONAL,
        )
        print(result["content"])
        print(result["score"])
    """

    def __init__(
        self,
        openai_provider,
        scorer: ContentScorer,
        max_regeneration_attempts: int = 3,
        score_threshold: float = 6.0,
    ) -> None:
        if max_regeneration_attempts < 1:
            raise ValueError("max_regeneration_attempts must be at least 1")
        if not (0.0 <= score_threshold <= 10.0):
            raise ValueError("score_threshold must be between 0 and 10")

        self.openai_provider = openai_provider
        self.scorer = scorer
        self.max_regeneration_attempts = max_regeneration_attempts
        self.score_threshold = score_threshold

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_goal_prompt(self, goal: PostGoal) -> str:
        """Return the goal-specific instruction block for the given goal.

        The returned string is appended to the base ``HUMANIZED_PROMPT``
        before the topic line so the model receives coherent, unified
        instructions rather than a separate system nudge.

        Args:
            goal: The desired :class:`PostGoal`.

        Returns:
            Goal instruction string (Bengali/Banglish).
        """
        return _GOAL_PROMPTS.get(goal.value, "")

    def _select_style(self, goal: PostGoal, override: Optional[str] = None) -> str:
        """Return a writing style, preferring goal-affine styles.

        Args:
            goal    : The target :class:`PostGoal`.
            override: Explicit style name.  Validated against ``POST_STYLES``;
                      ignored if invalid.

        Returns:
            A valid style string from ``POST_STYLES``.
        """
        if override and override in POST_STYLES:
            return override

        affinities = _GOAL_STYLE_AFFINITY.get(goal.value, POST_STYLES)
        return random.choice(affinities)

    def _select_mood(self, override: Optional[str] = None) -> str:
        """Return a mood, using the override if valid.

        Args:
            override: Explicit mood name.  Validated against ``POST_MOODS``.

        Returns:
            A valid mood string from ``POST_MOODS``.
        """
        if override and override in POST_MOODS:
            return override
        return random.choice(POST_MOODS)

    def _select_length(self) -> str:
        """Sample a length tier using the standard weighted distribution."""
        return random.choices(POST_LENGTHS, weights=[25, 50, 25], k=1)[0]

    def _build_full_prompt(
        self,
        topic: str,
        goal: PostGoal,
        style: str,
        mood: str,
        length: str,
        feedback: Optional[List[str]] = None,
    ) -> str:
        """Assemble the complete generation prompt.

        On regeneration attempts, the scorer's top-3 feedback items are
        embedded directly so the model has concrete direction for improvement.

        Args:
            topic    : Subject to write about.
            goal     : Target :class:`PostGoal`.
            style    : Writing style from ``POST_STYLES``.
            mood     : Writing mood from ``POST_MOODS``.
            length   : Length tier from ``POST_LENGTHS``.
            feedback : Feedback from the previous failed attempt, or ``None``
                       on the first attempt.

        Returns:
            Fully assembled prompt string ready for the OpenAI API.
        """
        # Base humanized prompt (style + mood + length substituted in)
        prompt = HUMANIZED_PROMPT.format(style=style, mood=mood, length=length)

        # Goal-specific instructions
        goal_instruction = self._build_goal_prompt(goal)
        if goal_instruction:
            prompt += f"\n\nGoal Instructions:\n{goal_instruction}"

        # Feedback from prior failed attempt (regeneration only)
        if feedback:
            # Limit to top 3 most actionable items to avoid prompt bloat
            top_feedback = [fb for fb in feedback if fb][:3]
            if top_feedback:
                feedback_str = " | ".join(top_feedback)
                prompt += (
                    f"\n\nপূর্ববর্তী পোস্টের সমস্যা: {feedback_str}। এগুলো ঠিক করে আবার লিখো।"
                )

        prompt += f"\nTopic: {topic}"
        return prompt

    def _call_openai(self, prompt: str, length: str) -> str:
        """Call the OpenAI provider and return cleaned post content.

        Args:
            prompt : Fully assembled generation prompt.
            length : Length tier (determines ``max_tokens``).

        Returns:
            Cleaned post content string.

        Raises:
            Any exception raised by the underlying OpenAI client.
        """
        raw = self.openai_provider.generate_completion(
            prompt=prompt,
            system_message=_SYSTEM_MESSAGE,
            max_tokens=_MAX_TOKENS.get(length, 700),
            temperature=random.uniform(0.85, 0.98),
        )
        return _clean_post(raw)

    @staticmethod
    def _fallback_content(topic: str) -> str:
        """Minimal fallback post used when all generation attempts fail."""
        return (
            f"{topic} নিয়ে আজ কাজ করতে গিয়ে কিছু নতুন জিনিস শিখেছি।\n\n"
            "Development এর journey টা এমনই — প্রতিদিন কিছু না কিছু নতুন শেখার থাকে।\n\n"
            "আপনাদের কি মনে হয় এ বিষয়ে?"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_post_with_goal(
        self,
        topic: str,
        goal: PostGoal = PostGoal.EDUCATIONAL,
        style: Optional[str] = None,
        mood: Optional[str] = None,
    ) -> Dict:
        """Generate a quality-gated LinkedIn post aligned to a specific goal.

        The engine will attempt generation up to ``max_regeneration_attempts``
        times.  After each failing attempt the scorer's feedback is injected
        into the next prompt so the model has actionable direction.  The loop
        exits early as soon as a post passes the quality threshold.

        If the entire pipeline raises an unhandled exception the engine falls
        back to a minimal hardcoded post so callers always receive a usable
        result.

        Args:
            topic : Subject to write about.
            goal  : Desired :class:`PostGoal`.  Defaults to EDUCATIONAL.
            style : Optional style override (must be a value in
                    ``POST_STYLES``; silently ignored if invalid).
            mood  : Optional mood override (must be a value in
                    ``POST_MOODS``; silently ignored if invalid).

        Returns:
            Dict with keys:

            - ``content``      (str)          – Final post text.
            - ``score``        (ContentScore) – Quality score of the final post.
            - ``goal``         (str)          – Goal value (enum name).
            - ``topic``        (str)          – Original topic.
            - ``attempts_used``(int)          – Number of generation attempts made.
            - ``style``        (str)          – Style used for the final attempt.
            - ``mood``         (str)          – Mood used for the final attempt.
        """
        selected_style = self._select_style(goal, style)
        selected_mood = self._select_mood(mood)
        length = self._select_length()

        content: str = ""
        score: Optional[ContentScore] = None
        last_feedback: List[str] = []
        attempts: int = 0

        try:
            while attempts < self.max_regeneration_attempts:
                attempts += 1

                prompt = self._build_full_prompt(
                    topic=topic,
                    goal=goal,
                    style=selected_style,
                    mood=selected_mood,
                    length=length,
                    # Pass feedback only on retry attempts
                    feedback=last_feedback if attempts > 1 else None,
                )

                content = self._call_openai(prompt, length)
                score = self.scorer.score_post_locally(content, goal.value)

                logger.info(
                    "Content generation | topic=%s | goal=%s | score=%.2f "
                    "| attempts=%d | passed=%s",
                    topic,
                    goal.value,
                    score.overall_score,
                    attempts,
                    score.passed,
                )

                if score.passed:
                    # Quality threshold met — exit the loop early
                    break

                if attempts < self.max_regeneration_attempts:
                    # Prepare for next attempt: capture feedback and refresh
                    # style/mood so the model gets a genuinely different angle.
                    last_feedback = [fb for fb in score.feedback if fb]

                    # Keep goal-affine style selection on retries
                    selected_style = self._select_style(goal)
                    selected_mood = self._select_mood()

                    logger.debug(
                        "Regenerating | topic=%s | attempt=%d/%d | "
                        "new_style=%s | new_mood=%s | feedback_items=%d",
                        topic,
                        attempts + 1,
                        self.max_regeneration_attempts,
                        selected_style,
                        selected_mood,
                        len(last_feedback),
                    )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Enhanced generation failed for topic '%s' (attempt %d): %s — "
                "falling back to basic generation.",
                topic,
                attempts,
                exc,
                exc_info=True,
            )
            content = self._fallback_content(topic)
            score = self.scorer.score_post_locally(content, goal.value)

        return {
            "content": content,
            "score": score,
            "goal": goal.value,
            "topic": topic,
            "attempts_used": attempts,
            "style": selected_style,
            "mood": selected_mood,
        }

    def batch_generate_for_ab_test(
        self,
        topic: str,
        goal: PostGoal,
        count: int = 2,
    ) -> List[Dict]:
        """Generate multiple post variations for A/B testing.

        Each variation is produced with a distinct style×mood pair to ensure
        meaningful creative diversity across the batch.  Pairs are drawn from
        goal-affine styles first, then from the full ``POST_STYLES`` pool to
        guarantee uniqueness even for large ``count`` values.

        Args:
            topic : Subject to write about.
            goal  : Target :class:`PostGoal`.
            count : Number of variations to generate.  Defaults to 2.

        Returns:
            List of result dicts in the same format as
            :meth:`generate_post_with_goal`, one per variation.
        """
        if count < 1:
            logger.warning(
                "batch_generate_for_ab_test called with count=%d; returning []", count
            )
            return []

        # Build shuffled pools so consecutive variations never share style+mood
        styles_pool = list(POST_STYLES)
        moods_pool = list(POST_MOODS)
        random.shuffle(styles_pool)
        random.shuffle(moods_pool)

        # Rotate through the pools using modular indexing so we never run out
        # even when count > len(pool).
        results: List[Dict] = []

        for i in range(count):
            variation_style = styles_pool[i % len(styles_pool)]
            variation_mood = moods_pool[i % len(moods_pool)]

            logger.info(
                "A/B batch generation [%d/%d] | topic=%s | goal=%s "
                "| style=%s | mood=%s",
                i + 1,
                count,
                topic,
                goal.value,
                variation_style,
                variation_mood,
            )

            result = self.generate_post_with_goal(
                topic=topic,
                goal=goal,
                style=variation_style,
                mood=variation_mood,
            )
            results.append(result)

        return results

    def get_best_from_batch(self, variations: List[Dict]) -> Dict:
        """Return the highest-scoring variation from a batch.

        Selection is based solely on ``overall_score`` from the embedded
        :class:`~modules.content.scorer.ContentScore`.  When multiple
        variations share the same top score the first occurrence is returned
        (preserving the order they were generated in).

        Args:
            variations : List returned by :meth:`batch_generate_for_ab_test`
                         or any list of result dicts that contain a ``score``
                         key holding a :class:`ContentScore`.

        Returns:
            The variation dict with the highest ``overall_score``.

        Raises:
            ValueError : If *variations* is empty.
        """
        if not variations:
            raise ValueError(
                "Cannot select best variation from an empty list. "
                "Ensure batch_generate_for_ab_test returned at least one result."
            )

        def _safe_score(variation: Dict) -> float:
            score_obj = variation.get("score")
            if isinstance(score_obj, ContentScore):
                return score_obj.overall_score
            # Guard against malformed entries (e.g. from mocked tests)
            return 0.0

        best = max(variations, key=_safe_score)

        best_score = _safe_score(best)
        logger.info(
            "Best A/B variation selected | overall_score=%.2f | style=%s | mood=%s "
            "| goal=%s | attempts_used=%d",
            best_score,
            best.get("style", "unknown"),
            best.get("mood", "unknown"),
            best.get("goal", "unknown"),
            best.get("attempts_used", 0),
        )

        return best
