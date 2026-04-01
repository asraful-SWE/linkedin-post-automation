"""
Pipeline Service - Orchestrates the full content processing pipeline.

Pipeline stages:
1. Collect: Fetch from all sources
2. Deduplicate: Remove duplicates
3. Extract: Get full article text and images
4. Filter: Remove irrelevant content
5. Enrich: AI summarization and analysis
6. Score: Calculate importance scores
7. Store: Save to database
"""

import asyncio
import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Type

from news_bot.config import config
from news_bot.collectors.base_collector import BaseCollector
from news_bot.collectors.hackernews_collector import HackerNewsCollector
from news_bot.collectors.reddit_collector import RedditCollector
from news_bot.collectors.devto_collector import DevtoCollector
from news_bot.collectors.medium_collector import MediumCollector
from news_bot.collectors.techcrunch_collector import TechCrunchCollector
from news_bot.extractors.article_extractor import ArticleExtractor
from news_bot.extractors.image_extractor import ImageExtractor
from news_bot.processors.deduplicator import Deduplicator
from news_bot.processors.content_filter import ContentFilter
from news_bot.processors.ai_enricher import AIEnricher
from news_bot.processors.scorer import ContentScorer
from news_bot.services.content_service import ContentService
from news_bot.models import ContentItem, ContentStatus

logger = logging.getLogger(__name__)


class PipelineService:
    """
    Orchestrates the complete content processing pipeline.
    
    Coordinates collectors, extractors, processors, and storage
    to transform raw content into enriched, scored content items.
    """
    
    # Available collectors
    COLLECTORS: Dict[str, Type[BaseCollector]] = {
        "hackernews": HackerNewsCollector,
        "reddit": RedditCollector,
        "devto": DevtoCollector,
        "medium": MediumCollector,
        "techcrunch": TechCrunchCollector,
    }
    
    def __init__(
        self,
        db_path: str = None,
        enabled_sources: Optional[List[str]] = None,
        max_items_per_source: int = None,
        enable_ai_enrichment: bool = None,
        extraction_concurrency: int = None,
        extract_images: bool = None,
    ):
        """
        Initialize pipeline service.
        
        Args:
            db_path: Path to SQLite database
            enabled_sources: List of enabled sources (None = use config)
            max_items_per_source: Max items to fetch per source (None = use config)
            enable_ai_enrichment: Whether to use AI for enrichment (None = use config)
            extraction_concurrency: Concurrent extraction requests (None = use config)
            extract_images: Whether to extract images (None = use config)
        """
        # Use config defaults if not specified
        self.db_path = db_path or "data/news_bot.db"
        self.enabled_sources = enabled_sources or [
            name for name, collector_config in config.collectors.items()
            if collector_config.enabled
        ]
        self.max_items_per_source = max_items_per_source or 30
        self.enable_ai_enrichment = enable_ai_enrichment if enable_ai_enrichment is not None else config.ai.enabled
        self.extraction_concurrency = extraction_concurrency or config.extraction.concurrency
        self.extract_images = extract_images if extract_images is not None else config.images.enabled
        
        # Initialize components
        self.content_service = ContentService(self.db_path)
        self.deduplicator = Deduplicator(self.db_path)
        self.content_filter = ContentFilter()
        self.scorer = ContentScorer()
        
        if enable_ai_enrichment:
            self.ai_enricher = AIEnricher()
        else:
            self.ai_enricher = None
        
        # Extractors (created per run to manage connections)
        self._article_extractor: Optional[ArticleExtractor] = None
        self._image_extractor: Optional[ImageExtractor] = None
    
    async def run_full_pipeline(self) -> Dict[str, int]:
        """
        Run the complete pipeline.
        
        Returns:
            Statistics dict with counts for each stage
        """
        stats = {
            "collected": 0,
            "deduplicated": 0,
            "extracted": 0,
            "filtered": 0,
            "enriched": 0,
            "stored": 0,
            "errors": 0,
        }
        
        start_time = datetime.now(timezone.utc)
        logger.info("Starting content pipeline...")
        
        try:
            # Stage 1: Collect from all sources
            logger.info("Stage 1: Collecting content...")
            all_items = await self._collect_all()
            stats["collected"] = len(all_items)
            
            if not all_items:
                logger.warning("No items collected, stopping pipeline")
                return stats
            
            # Stage 2: Deduplicate
            logger.info("Stage 2: Deduplicating...")
            unique_items = self.deduplicator.filter_duplicates(all_items)
            stats["deduplicated"] = len(unique_items)
            
            if not unique_items:
                logger.info("All items were duplicates")
                return stats
            
            # Stage 3: Extract full articles and images
            logger.info("Stage 3: Extracting articles...")
            extracted_items = await self._extract_all(unique_items)
            stats["extracted"] = len([i for i in extracted_items if i.full_text])
            
            # Stage 4: Filter
            logger.info("Stage 4: Filtering content...")
            filtered_items = self.content_filter.filter_items(extracted_items)
            stats["filtered"] = len(filtered_items)
            
            if not filtered_items:
                logger.info("All items filtered out")
                return stats
            
            # Stage 5: AI Enrichment
            if self.ai_enricher:
                logger.info("Stage 5: AI enrichment...")
                enriched_items = self.ai_enricher.enrich_batch(filtered_items)
                stats["enriched"] = len([i for i in enriched_items if i.summary])
            else:
                enriched_items = filtered_items
                # Calculate scores without AI
                for item in enriched_items:
                    item.score = self.scorer.score(item)
                    item.status = ContentStatus.PROCESSED.value
                stats["enriched"] = len(enriched_items)
            
            # Stage 6: Final scoring and storage
            logger.info("Stage 6: Scoring and storing...")
            scored_items = self.scorer.score_batch(enriched_items)
            saved_count = self.content_service.save_batch(scored_items)
            stats["stored"] = saved_count
            
            # Log summary
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(
                f"Pipeline completed in {duration:.1f}s | "
                f"Collected: {stats['collected']} | "
                f"Unique: {stats['deduplicated']} | "
                f"Extracted: {stats['extracted']} | "
                f"Filtered: {stats['filtered']} | "
                f"Stored: {stats['stored']}"
            )
            
            return stats
            
        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
            stats["errors"] += 1
            return stats
            
        finally:
            await self._cleanup()
    
    async def run_collection_only(self) -> List[ContentItem]:
        """
        Run only the collection stage.
        
        Returns:
            List of collected items (not stored)
        """
        try:
            items = await self._collect_all()
            unique_items = self.deduplicator.filter_duplicates(items)
            return unique_items
        finally:
            await self._cleanup()
    
    async def run_extraction_for_pending(self) -> int:
        """
        Extract articles for pending items in database.
        
        Returns:
            Number of items processed
        """
        try:
            pending = self.content_service.list_pending(limit=50)
            if not pending:
                logger.info("No pending items to extract")
                return 0
            
            processed = await self._extract_all(pending)
            
            # Update in database
            for item in processed:
                self.content_service.update(item)
            
            return len(processed)
            
        finally:
            await self._cleanup()
    
    async def _collect_all(self) -> List[ContentItem]:
        """Collect from all enabled sources."""
        all_items = []
        
        for source_name in self.enabled_sources:
            if source_name not in self.COLLECTORS:
                logger.warning(f"Unknown source: {source_name}")
                continue
            
            try:
                collector_class = self.COLLECTORS[source_name]
                collector = collector_class(max_items=self.max_items_per_source)
                
                items = await collector.collect()
                all_items.extend(items)
                
                logger.info(f"Collected {len(items)} items from {source_name}")
                
            except Exception as e:
                logger.error(f"Error collecting from {source_name}: {e}")
                continue
        
        return all_items
    
    async def _extract_all(
        self,
        items: List[ContentItem],
    ) -> List[ContentItem]:
        """Extract full articles and images for all items."""
        # Create extractors
        self._article_extractor = ArticleExtractor()
        self._image_extractor = ImageExtractor()
        
        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.extraction_concurrency)
        
        async def extract_one(item: ContentItem) -> ContentItem:
            async with semaphore:
                return await self._extract_item(item)
        
        # Process concurrently
        tasks = [extract_one(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors
        extracted = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Extraction error for {items[i].url}: {result}")
                items[i].status = ContentStatus.FAILED.value
                items[i].error_message = str(result)
                extracted.append(items[i])
            else:
                extracted.append(result)
        
        return extracted
    
    async def _extract_item(self, item: ContentItem) -> ContentItem:
        """Extract article and image for a single item."""
        try:
            # Skip Reddit discussion pages
            if "reddit.com/r/" in item.url and "/comments/" in item.url:
                item.status = ContentStatus.SKIPPED.value
                return item
            
            # Extract article with fallback to RSS summary
            fallback_summary = getattr(item, 'summary', None) or getattr(item, 'description', None)
            article_data = await self._article_extractor.extract(
                item.url, 
                fallback_summary=fallback_summary
            )
            
            if article_data["success"]:
                item.full_text = article_data["full_text"]
                
                # Update title if better one found
                if article_data.get("title") and len(article_data["title"]) > len(item.title or ""):
                    item.title = article_data["title"]
                
                # Update author if found
                if article_data.get("author"):
                    item.author = article_data["author"]
                
                # Store reading time and word count
                item.reading_time = article_data.get("reading_time", 0)
                item.word_count = article_data.get("word_count", 0)
                
                # Extract image if enabled
                if self.extract_images:
                    try:
                        image_url = await self._image_extractor.extract(item.url, item.full_text)
                        if image_url:
                            item.image_url = image_url
                    except Exception as e:
                        logger.debug(f"Image extraction failed for {item.url}: {e}")
                
                item.status = ContentStatus.EXTRACTED.value
                
            else:
                # Extraction failed but fallback might be available
                if article_data.get("extraction_method") == "rss_fallback":
                    item.full_text = article_data["full_text"]
                    item.status = ContentStatus.EXTRACTED.value
                else:
                    item.status = ContentStatus.FAILED.value
                    item.error_message = article_data.get("error", "Extraction failed")
                    logger.debug(f"Article extraction failed for {item.url}: {item.error_message}")
            
            return item
            
        except Exception as e:
            logger.warning(f"Error extracting item {item.url}: {e}")
            item.status = ContentStatus.FAILED.value
            item.error_message = str(e)
            return item
    
    async def _cleanup(self):
        """Cleanup resources."""
        if self._article_extractor:
            await self._article_extractor.close()
            self._article_extractor = None
        
        if self._image_extractor:
            await self._image_extractor.close()
            self._image_extractor = None


# Convenience function for running pipeline from scheduler
async def run_content_pipeline(db_path: str) -> Dict[str, int]:
    """
    Run the content pipeline.
    
    Convenience function for scheduler integration.
    
    Args:
        db_path: Path to database
        
    Returns:
        Pipeline statistics
    """
    pipeline = PipelineService(db_path)
    return await pipeline.run_full_pipeline()
