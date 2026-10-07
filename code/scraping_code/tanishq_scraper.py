import os
import time
import hashlib
import requests
import re
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image, UnidentifiedImageError
from io import BytesIO

# Your specific local paths
CATEGORIES = {
    "https://www.tanishq.co.in/shop/earring?lang=en_IN": r"D:\gen jewels\web scraping\Dataset\Tanishq\Earrings",
    "https://www.tanishq.co.in/shop/finger-rings?lang=en_IN": r"D:\gen jewels\web scraping\Dataset\Tanishq\Rings",
    "https://www.tanishq.co.in/shop/pendants?lang=en_IN": r"D:\gen jewels\web scraping\Dataset\Tanishq\Pendants",
    "https://www.tanishq.co.in/shop/bangles?lang=en_IN": r"D:\gen jewels\web scraping\Dataset\Tanishq\Bangles"
}

def setup_driver():
    """Sets up a fresh Chrome WebDriver session to avoid bot-detection."""
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=chrome_options)

def scroll_page_smoothly(driver):
    """Scrolls smoothly down the page to trigger all lazy-loaded products."""
    print("Scrolling smoothly to load all products...")
    
    # Remove annoying popups/overlays using JS so they don't block scrolling
    driver.execute_script("""
        document.querySelectorAll('.modal, .overlay, [id*="popup"], .cookie-banner').forEach(el => el.remove());
        document.body.style.overflow = 'auto';
    """)
    
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    for i in range(40):
        # Scroll down by 800 pixels at a time (human-like)
        driver.execute_script("window.scrollBy(0, 800);")
        time.sleep(1.5) 
        
        new_height = driver.execute_script("return document.body.scrollHeight")
        
        # If we hit the bottom, try to bounce a bit to see if more loads
        if new_height == last_height:
            driver.execute_script("window.scrollBy(0, -300);")
            time.sleep(1)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
                
        last_height = new_height
        print(f"Scroll iteration {i + 1}/40")

def get_high_res_url(url):
    """
    Forces the URL to request a high-quality 2K image by manipulating URL parameters.
    Example: changes ?sw=250&sh=250 to ?sw=2000&sh=2000
    """
    url = re.sub(r'sw=\d+', 'sw=2000', url)
    url = re.sub(r'sh=\d+', 'sh=2000', url)
    url = re.sub(r'w=\d+', 'w=2000', url)
    url = re.sub(r'h=\d+', 'h=2000', url)
    
    # If it has no size params but is a product image, we can append them
    if "sw=" not in url and "?" in url:
        url += "&sw=2000&sh=2000"
    elif "sw=" not in url and "?" not in url:
        url += "?sw=2000&sh=2000"
        
    return url

def extract_image_urls(driver, base_url):
    """Extracts images and immediately filters out SVGs to reduce noise."""
    js_script = """
        let urls = new Set();
        
        document.querySelectorAll('img').forEach(img => {
            let attrs = ['src', 'data-src', 'data-lazy-src', 'data-original'];
            attrs.forEach(attr => {
                let val = img.getAttribute(attr);
                if (val && !val.includes('.svg') && !val.includes('.gif')) urls.add(val);
            });
            
            let srcset = img.getAttribute('srcset') || img.getAttribute('data-srcset');
            if (srcset) {
                srcset.split(',').forEach(item => {
                    let url = item.trim().split(' ')[0];
                    if (url && !url.includes('.svg')) urls.add(url);
                });
            }
        });
        
        return Array.from(urls);
    """
    raw_urls = driver.execute_script(js_script)
    valid_urls = set()
    
    for url in raw_urls:
        if url.startswith("data:"):
            continue 
            
        full_url = urljoin(base_url, url)
        if full_url.startswith("http"):
            # Upgrade URL to High-Resolution right away
            high_res_url = get_high_res_url(full_url)
            valid_urls.add(high_res_url)
            
    return valid_urls

def process_category(session, url, output_dir):
    """Processes a single category URL using a fresh driver."""
    print(f"\n{'='*50}")
    print(f"Processing Category: {output_dir}")
    print(f"URL: {url}")
    print(f"{'='*50}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Open a fresh driver for EACH category to prevent Tanishq from blocking us
    driver = setup_driver()
    
    try:
        driver.get(url)
        time.sleep(8) # Wait longer initially for bot-checks to pass
        
        scroll_page_smoothly(driver)
        image_urls = extract_image_urls(driver, url)
        
        print(f"\nFound {len(image_urls)} potential images. Downloading High-Res (2K) PNGs...")
        
        saved_count = 0
        for index, img_url in enumerate(sorted(image_urls), start=1):
            try:
                response = session.get(img_url, timeout=20)
                if response.status_code != 200:
                    continue
                    
                image_data = BytesIO(response.content)
                img = Image.open(image_data)
                
                # Filter out small UI/Junk images (logos, buttons). Only keep large images (min 300px)
                if img.width < 300 or img.height < 300:
                    continue
                
                # Convert to RGBA for PNG to handle any transparency perfectly
                if img.mode not in ("RGBA", "RGB"):
                    img = img.convert("RGBA")
                    
                url_hash = hashlib.md5(img_url.encode()).hexdigest()[:8]
                filename = f"tanishq_{img.width}x{img.height}_{url_hash}.png"
                filepath = os.path.join(output_dir, filename)
                
                img.save(filepath, format="PNG")
                saved_count += 1
                print(f"[{saved_count}] Saved High-Quality PNG: {filename} (Size: {img.width}x{img.height})")
                
            except UnidentifiedImageError:
                pass # Silently skip invalid images
            except Exception as e:
                print(f"Error downloading {img_url}: {str(e)}")
                
        print(f"\nSuccessfully downloaded {saved_count} High-Quality images for this category.")
        
    finally:
        driver.quit() # Close browser before opening the next one

def main():
    # Use requests session for rapid downloading and to carry headers
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    })
    
    for url, folder_path in CATEGORIES.items():
        process_category(session, url, folder_path)
        # Brief pause between categories to ensure network resets properly
        time.sleep(3)

    print("\nAll categories scraped successfully! Check your D: drive folders.")

if __name__ == "__main__":
    main()