import asyncio
import logging
from crawler import AICrawler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler.log'),
        logging.StreamHandler()
    ]
)

async def main():
    initial_url = "https://github.com/modelcontextprotocol"
    topic = "Put together everything I need to know about the model context protocol."
    
    async with AICrawler(initial_url, topic) as crawler:
        await crawler.run()

if __name__ == "__main__":
    asyncio.run(main())