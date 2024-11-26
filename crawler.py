import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import openai
from typing import List, Set, Dict, Tuple, Optional, AsyncIterator, Any
import json
from dataclasses import dataclass
import logging
from urllib.robotparser import RobotFileParser
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from asyncio import Lock
from sentence_transformers import SentenceTransformer
from datetime import datetime
from pydantic import BaseSettings, validator
import signal
import psutil

@dataclass
class CrawlResult:
    url: str
    content_summary: str
    relevant_links: List[str]
    metadata: Dict

@dataclass
class CrawlerMetrics:
    start_time: float
    urls_processed: int = 0
    total_processing_time: float = 0.0
    errors: int = 0
    successful_requests: int = 0

class AICrawler:
    """
    AI-powered web crawler that intelligently traverses websites based on relevance to a topic.
    
    Args:
        initial_url (str): The starting URL for crawling
        topic (str): The topic to focus crawling on
        config (CrawlerConfig, optional): Configuration settings
        
    Attributes:
        visited_urls (Set[str]): Set of processed URLs
        url_queue (asyncio.Queue): Queue of URLs to process
    """
    MAX_CONTENT_LENGTH = 1000
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    MEMORY_CLEANUP_INTERVAL = 300  # 5 minutes
    def __init__(self, 
                 initial_url: str, 
                 topic: str,
                 config: Optional[CrawlerConfig] = None) -> None:
        self.config = config or CrawlerConfig()
        self.max_concurrent = self.config.max_concurrent
        openai.api_key = self.config.openai_api_key
        self.initial_url = initial_url
        self.topic = topic
        self.output_file = "crawl_results.jsonl"
        self.visited_urls: Set[str] = set()
        self.url_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self.domain = urlparse(initial_url).netloc
        self.robot_parser = RobotFileParser()
        self.robot_parser.set_url(urljoin(initial_url, "/robots.txt"))
        self.robot_parser.read()
        self.crawler_name = "YourCrawlerName/1.0"  # Identify your crawler
        self.content_store = ContentStore()
        self.context_manager = ContextManager(topic)
        self.rate_limiter = RateLimiter(tokens_per_second=2.0)
        self.progress_tracker = {
            'processed': 0,
            'queued': 1,
            'errors': 0,
            'start_time': time.time()
        }
        self.metrics = CrawlerMetrics(time.time())
        self._shutdown = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('crawler.log'),
                logging.StreamHandler()
            ]
        )
        
    def _signal_handler(self, signum, frame):
        self._shutdown = True
        logging.info("Shutdown signal received, cleaning up...")

    async def analyze_with_ai(self, content: str, url: str) -> Tuple[str, List[str], float]:
        """Enhanced AI analysis with relevance scoring"""
        prompt = f"""
        Topic: {self.topic}
        URL: {url}
        Content: {content[:1000]}...
        
        Tasks:
        1. Summarize the key points relevant to the topic
        2. Rate relevance to topic (0-1)
        3. Identify most promising links to follow
        4. Extract key concepts/terms
        
        Format: JSON
        """
        
        try:
            headers = {
                "Authorization": f"Bearer {openai.api_key}",
                "HTTP-Referer": "https://crawlcombiner.local",
                "X-Title": "CrawlCombiner"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": "openai/gpt-4-mini",
                        "messages": [{"role": "user", "content": prompt}]
                    }
                ) as response:
                    result = await response.json()
                    
            analysis = json.loads(result['choices'][0]['message']['content'])
            return (
                analysis.get('summary', ''),
                analysis.get('relevant_links', []),
                analysis.get('relevance_score', 0.0)
            )
        except json.JSONDecodeError:
            logging.error(f"Failed to parse AI response for {url}")
            return ('', [], 0.0)
        except Exception as e:
            logging.error(f"AI analysis failed for {url}: {e}")
            return ('', [], 0.0)

    async def can_fetch(self, url: str) -> bool:
        """Check if we're allowed to fetch this URL"""
        return self.robot_parser.can_fetch(self.crawler_name, url)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def process_url(self, url: str) -> Optional[CrawlResult]:
        """Process a single URL"""
        while not await self.rate_limiter.acquire():
            await asyncio.sleep(0.1)
        
        if not await self.can_fetch(url):
            logging.warning(f"Robots.txt prevents accessing {url}")
            return None

        # Get crawl delay from robots.txt
        crawl_delay = self.robot_parser.crawl_delay(self.crawler_name) or 1
        await asyncio.sleep(crawl_delay)

        timeout = aiohttp.ClientTimeout(total=30)  # 30 seconds timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                content_type = response.headers.get('content-type', '')
                
                if 'text/html' in content_type:
                    return await self.process_html(response)
                elif 'application/json' in content_type:
                    return await self.process_json(response)
                elif 'text/markdown' in content_type:
                    return await self.process_markdown(response)

    async def worker(self) -> None:
        """Worker to process URLs from the queue"""
        while not self._shutdown:
            url = await self.url_queue.get()
            try:
                if self.is_valid_url(url):
                    result = await self.process_url(url)
                    if result:
                        await self.content_store.store(result)
                        self.progress_tracker['processed'] += 1
                        
                        for link in result.relevant_links:
                            if link not in self.visited_urls:
                                await self.url_queue.put(link)
                                self.visited_urls.add(link)
                                self.progress_tracker['queued'] += 1
                else:
                    logging.info(f"Skipping invalid URL: {url}")
            except Exception as e:
                logging.error(f"Error processing {url}: {e}")
                self.progress_tracker['errors'] += 1
            finally:
                self.url_queue.task_done()

    def save_result(self, result: CrawlResult):
        """Save result to output file"""
        with open(self.output_file, 'a') as f:
            json.dump(result.__dict__, f)
            f.write('\n')

    async def run(self):
        try:
            # Initialize with first URL
            await self.url_queue.put(self.initial_url)
            self.visited_urls.add(self.initial_url)
            
            # Create workers
            workers = [
                asyncio.create_task(self.worker())
                for _ in range(self.max_concurrent)
            ]
            
            # Wait for initial queue to be processed
            await self.url_queue.join()
            
            # Cancel workers
            for w in workers:
                w.cancel()

            progress_task = asyncio.create_task(self.report_progress())
        finally:
            await self.cleanup()

    def is_valid_url(self, url: str) -> bool:
        """Validate URLs before processing"""
        parsed = urlparse(url)
        # Stay within same domain
        if parsed.netloc != self.domain:
            return False
        # Skip common non-content URLs
        if any(ext in parsed.path.lower() for ext in ['.jpg', '.png', '.css', '.js']):
            return False
        # Custom rules based on topic
        if self.topic == "code":
            return any(path in parsed.path.lower() for path in ['/src/', '/lib/', '/docs/'])
        return True

    async def process_html(self, response) -> CrawlResult:
        content = await response.text()
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script, style elements
        for element in soup(['script', 'style', 'nav', 'footer']):
            element.decompose()
        
        text = soup.get_text(separator=' ', strip=True)
        summary, relevant_links, relevance = await self.analyze_with_ai(text, str(response.url))
        
        return CrawlResult(
            url=str(response.url),
            content_summary=summary,
            relevant_links=[urljoin(str(response.url), link) for link in relevant_links],
            metadata={"type": "html", "relevance": relevance}
        )

    async def process_json(self, response) -> CrawlResult:
        content = await response.json()
        summary, relevant_links, relevance = await self.analyze_with_ai(
            json.dumps(content, indent=2), 
            str(response.url)
        )
        
        return CrawlResult(
            url=str(response.url),
            content_summary=summary,
            relevant_links=relevant_links,
            metadata={"type": "json", "relevance": relevance}
        )

    async def process_markdown(self, response) -> CrawlResult:
        content = await response.text()
        summary, relevant_links, relevance = await self.analyze_with_ai(content, str(response.url))
        
        return CrawlResult(
            url=str(response.url),
            content_summary=summary,
            relevant_links=relevant_links,
            metadata={"type": "markdown", "relevance": relevance}
        )

    async def cleanup_memory(self):
        """Periodically clean up memory"""
        while True:
            if len(self.visited_urls) > 10000:
                # Keep only recent URLs
                self.visited_urls = set(list(self.visited_urls)[-5000:])
            await asyncio.sleep(300)  # Run every 5 minutes

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    async def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if not self._shutdown else "shutting_down",
            "queue_size": self.url_queue.qsize(),
            "processed_urls": len(self.visited_urls),
            "error_rate": self.metrics.errors / max(self.metrics.urls_processed, 1),
            "avg_processing_time": self.metrics.total_processing_time / max(self.metrics.urls_processed, 1),
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }

    async def cleanup(self):
        """Cleanup resources"""
        await self.content_store.close()
        # Cancel any remaining tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def report_progress(self):
        while not self._shutdown:
            logging.info(f"Progress: Processed {self.progress_tracker['processed']} URLs, "
                        f"Queued {self.progress_tracker['queued']}, "
                        f"Errors {self.progress_tracker['errors']}")
            await asyncio.sleep(10)  # Report every 10 seconds

class RateLimiter:
    def __init__(self, tokens_per_second: float):
        self.tokens = 1.0
        self.tokens_per_second = tokens_per_second
        self.last_update = time.time()
        self.lock = Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(1.0, self.tokens + time_passed * self.tokens_per_second)
            self.last_update = now
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

class ContentStore:
    def __init__(self):
        self.output_file = "crawl_results.jsonl"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    async def store(self, result: CrawlResult):
        # Generate embeddings for vector search
        embedding = self.model.encode(result.content_summary)
        
        # Store as JSONL with embedding and timestamp
        data = {
            **result.__dict__,
            'embedding': embedding.tolist(),
            'created_at': datetime.utcnow().isoformat()
        }
        
        with open(self.output_file, 'a') as f:
            json.dump(data, f)
            f.write('\n')

    async def close(self):
        pass  # No cleanup needed for file-based storage

class CrawlerConfig(BaseSettings):
    max_concurrent: int = 5
    max_depth: int = 3
    min_relevance_score: float = 0.5
    requests_per_second: float = 2.0
    openai_api_key: str
    
    class Config:
        env_file = '.env'

    @validator('requests_per_second')
    def validate_rate_limit(cls, v):
        if v <= 0 or v > 10:
            raise ValueError('requests_per_second must be between 0 and 10')
        return v

    @validator('min_relevance_score')
    def validate_relevance_score(cls, v):
        if v < 0 or v > 1:
            raise ValueError('min_relevance_score must be between 0 and 1')
        return v

class ContextManager:
    def __init__(self, initial_context: str):
        self.context = initial_context
        self.recent_summaries = []
    
    async def update_context(self, new_summary: str):
        self.recent_summaries.append(new_summary)
        if len(self.recent_summaries) > 5:
            self.recent_summaries.pop(0)
        
        headers = {
            "Authorization": f"Bearer {openai.api_key}",
            "HTTP-Referer": "https://crawlcombiner.local",  # Local identifier
            "X-Title": "CrawlCombiner"  # Your app name
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": "anthropic/claude-2",
                    "messages": [{
                        "role": "user",
                        "content": f"""
                        Current context: {self.context}
                        Recent findings: {self.recent_summaries}
                        
                        Update the context to reflect new information while maintaining focus.
                        """
                    }]
                }
            ) as response:
                result = await response.json()
                self.context = result['choices'][0]['message']['content']
