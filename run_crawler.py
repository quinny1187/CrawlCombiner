import asyncio
from crawler import AICrawler

async def main():
    # Example settings - modify these
    initial_url = "https://openai.com/blog"  # Replace with your target website
    topic = "your topic"                 # Replace with your topic of interest
    
    async with AICrawler(initial_url, topic) as crawler:
        await crawler.run()

if __name__ == "__main__":
    asyncio.run(main())