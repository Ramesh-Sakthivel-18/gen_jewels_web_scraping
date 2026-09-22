import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        print("Navigating...")
        await page.goto("https://www.tanishq.co.in/shop/necklaces", wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)
        html = await page.content()
        with open("tanishq_test2.html", "w", encoding="utf-8") as f:
            f.write(html)
        print("Done")
        await browser.close()

asyncio.run(main())
