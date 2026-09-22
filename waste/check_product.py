import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        print("Navigating to product...")
        await page.goto("https://www.tanishq.co.in/product/lattice-canopy-gold-hoop-earrings-50d6c5hhuaaa00.html?lang=en_IN", wait_until="domcontentloaded")
        await asyncio.sleep(3)
        html = await page.content()
        with open("product_test.html", "w", encoding="utf-8") as f:
            f.write(html)
        
        img_elements = await page.query_selector_all('img')
        for img in img_elements:
            src = await img.get_attribute('src')
            if src:
                print("IMG:", src)
        print("Done")
        await browser.close()

asyncio.run(main())
