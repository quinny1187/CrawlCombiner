# CrawlCombiner - AI-Powered Smart Web Crawler

CrawlCombiner is an intelligent web crawler that uses GPT-4 to analyze content relevance and make smart decisions about which links to follow based on topical relevance.

## Features

- 🤖 AI-powered content analysis using GPT-4
- 🎯 Topic-focused crawling
- 📊 Semantic understanding with embeddings
- ⚡ Async processing for high performance
- 🛡️ Built-in rate limiting and robots.txt compliance
- 📝 JSONL output format with embeddings
- 📈 Real-time progress tracking

## Installation

1. Clone the repository
2. Install dependencies:

pip install -r requirements.txt3. Create a .env file with your OpenAI API key:OPENAI_API_KEY=your_key_here## Usage1. Modify run_crawler.py with your target URL and topic:python
initial_url = "https://example.com"
topic = "your topic of interest"


2. Run the crawler:

bash
python run_crawler.py


## Output

Results are stored in `crawl_results.jsonl` with:
- URL
- Content summary
- Relevance score
- Semantic embeddings
- Metadata
- Timestamps

## Configuration

Key settings in `CrawlerConfig`:
- `max_concurrent`: Number of concurrent crawlers (default: 5)
- `max_depth`: Maximum crawl depth (default: 3)
- `min_relevance_score`: Minimum relevance threshold (default: 0.5)
- `requests_per_second`: Rate limiting (default: 2.0)

## Features in Detail

- **Smart Navigation**: Uses GPT-4 to analyze content relevance and decide which links to follow
- **Context Awareness**: Maintains crawling context through the ContextManager
- **Rate Limiting**: Respects website rate limits and robots.txt
- **Error Handling**: Built-in retries and error tracking
- **Memory Management**: Periodic cleanup of visited URLs
- **Progress Tracking**: Real-time logging of crawler progress

## Logs

Crawler progress is logged to:
- Console output
- `crawler.log` file

## Requirements

See `requirements.txt` for full dependencies:
- aiohttp
- beautifulsoup4
- openai
- tenacity
- sentence-transformers
- pydantic
- psutil

## License

MIT License