import os
import time
import hashlib
import requests
import re
import cv2
import numpy as np
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from PIL import Image, UnidentifiedImageError
from io import BytesIO

# --- CONFIGURE YOUR PATHS & LIMITS HERE ---
BASE_DIR = r"D:\gen jewels\web scraping\Dataset\DarJewellery"

MAX_IMAGES_PER_CATEGORY = 600
MAX_PAGES_PER_CATEGORY = 7

CATEGORIES = {
    "https://www.darjewellery.com/gold-jewellery/women/necklace": os.path.join(BASE_DIR, "Gold_Necklace"),
    "https://www.darjewellery.com/diamond-jewellery/women/necklace": os.path.join(BASE_DIR, "Diamond_Necklace"),
    "https://www.darjewellery.com/click-and-collect/click-and-collect/bangles": os.path.join(BASE_DIR, "CC_Bangles"),
    "https://www.darjewellery.com/click-and-collect/click-and-collect/necklace": os.path.join(BASE_DIR, "CC_Necklace"),
    "https://www.darjewellery.com/click-and-collect/click-and-collect/ladies-rings": os.path.join(BASE_DIR, "CC_Ladies_Rings"),
    "https://www.darjewellery.com/click-and-collect/click-and-collect/earrings": os.path.join(BASE_DIR, "CC_Earrings")
}

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    return webdriver.Chrome(options=chrome_options)

def establish_session(driver, session):
    print("Visiting homepage to bypass bot-detection...")
    driver.get("https://www.darjewellery.com/")
    time.sleep(8) 
    try:
        driver.execute_script("""
            document.querySelectorAll('[role="dialog"], .modal, .popup, [class*="Popup"], .overlay, iframe').forEach(e => e.remove());
            document.body.style.overflow = 'auto';
        """)
    except: pass
    for cookie in driver.get_cookies():
        session.cookies.set(cookie['name'], cookie['value'], domain=cookie['domain'])

def is_valid_product_image(img_pil):
    """
    Uses OpenCV Pixel Math to block Model (Wear) images and Measurement images.
    Returns (True, "") if pure product. Returns (False, "Reason") if it's junk.
    """
    try:
        cv_img = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
        h, w = cv_img.shape[:2]
        
        # --- 1. THE MODEL FILTER (Background Brightness Check) ---
        corners = [
            cv_img[0:50, 0:50],          # Top-Left
            cv_img[0:50, w-50:w],        # Top-Right
            cv_img[h-50:h, 0:50],        # Bottom-Left
            cv_img[h-50:h, w-50:w]       # Bottom-Right
        ]
        corner_brightness = sum(np.mean(c) for c in corners) / 4
        
        if corner_brightness > 15: 
            return False, "Skipped (Model/Studio Background Detected)"
            
        # --- 2. THE MEASUREMENT FILTER (Straight Line Detection) ---
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=120, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x1 - x2) < 3 and abs(y1 - y2) > 120:
                    return False, "Skipped (Measurement Lines Detected)"
                if abs(y1 - y2) < 3 and abs(x1 - x2) > 120:
                    return False, "Skipped (Measurement Lines Detected)"
                    
        return True, "Valid"
    except Exception as e:
        return True, "Valid"

def collect_product_links(driver):
    js_script = """
        let links = new Set();
        document.querySelectorAll('a').forEach(a => {
            let href = a.href;
            let bad = ['cart', 'login', 'category', 'contact', 'about', 'wishlist'];
            if (!bad.some(word => href.toLowerCase().includes(word)) && a.querySelector('img') && href.startsWith('http')) {
                 links.add(href);
            }
        });
        return Array.from(links);
    """
    return driver.execute_script(js_script)

def go_to_next_page(driver):
    try:
        elements = driver.find_elements(By.XPATH, "//a[contains(translate(normalize-space(.), 'NEXT', 'next'), 'next')] | //li[contains(@class, 'next')]/a")
        for el in elements:
            if el.is_displayed():
                parent_class = el.find_element(By.XPATH, "..").get_attribute("class") or ""
                if "disabled" in parent_class.lower() or "disabled" in (el.get_attribute("class") or ""): continue
                driver.execute_script("arguments[0].click();", el)
                return True
    except: pass
    return False

def extract_gallery_images(driver):
    js_script = """
        let urls = new Set();
        document.querySelectorAll('img').forEach(img => {
            let src = img.getAttribute('src') || img.getAttribute('data-src') || '';
            if (src.includes('/product_image/')) {
                let highResSrc = src.replace(/\\/s\\d+__/, '/s1200__');
                urls.add(highResSrc);
            }
        });
        return Array.from(urls);
    """
    raw_urls = driver.execute_script(js_script)
    return [url for url in raw_urls if url.startswith("http")]

def process_category(session, driver, category_url, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}\nScanning Category: {os.path.basename(output_dir)}\n{'='*60}")
    
    driver.get(category_url)
    time.sleep(5)
    
    global_seen_links = set()
    downloaded_pixel_hashes = set() 
    
    page = 1
    total_category_images_saved = 0
    limit_reached = False
    
    while not limit_reached:
        # --- LIMIT CHECK: PAGES ---
        if page > MAX_PAGES_PER_CATEGORY:
            print(f"\n[LIMIT REACHED] Scanned {MAX_PAGES_PER_CATEGORY} pages. Moving to next category.")
            break
            
        print(f"\n--- Loading Page {page} ---")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
        
        page_links = collect_product_links(driver)
        new_links_to_scrape = [link for link in page_links if link not in global_seen_links]
        global_seen_links.update(new_links_to_scrape)
                
        if not new_links_to_scrape:
            print("No new products found. Reached the end of the category!")
            break
            
        print(f"Found {len(new_links_to_scrape)} new products on Page {page}.")
        main_window = driver.current_window_handle
        
        for count, product_url in enumerate(new_links_to_scrape, 1):
            
            # --- LIMIT CHECK: IMAGES ---
            if limit_reached: break
            
            try:
                driver.switch_to.new_window('tab')
                driver.get(product_url)
                time.sleep(4) 
                
                if driver.current_url.rstrip('/') == "https://www.darjewellery.com":
                    continue

                driver.execute_script("window.scrollBy(0, 500);")
                time.sleep(1)

                gallery_urls = extract_gallery_images(driver)
                valid_angles_saved = 0
                
                for attempt_url in gallery_urls:
                    try:
                        res = session.get(attempt_url, timeout=10)
                        if res.status_code == 200:
                            img = Image.open(BytesIO(res.content))
                            
                            if img.width < 500 or img.height < 500: continue
                            aspect_ratio = img.width / img.height
                            if aspect_ratio < 0.85 or aspect_ratio > 1.15: continue
                            
                            if img.mode not in ("RGBA", "RGB"): img = img.convert("RGBA")
                            img_byte_arr = BytesIO()
                            img.save(img_byte_arr, format='PNG')
                            content_hash = hashlib.md5(img_byte_arr.getvalue()).hexdigest()
                            
                            if content_hash in downloaded_pixel_hashes:
                                continue 
                            
                            is_valid, skip_reason = is_valid_product_image(img)
                            if not is_valid:
                                print(f"  -> {skip_reason}")
                                continue
                            
                            # SAVE
                            valid_angles_saved += 1
                            total_category_images_saved += 1
                            filename = f"{os.path.basename(output_dir)}_Prod{count}_Angle{valid_angles_saved}_{img.width}x{img.height}_{content_hash[:6]}.png"
                            
                            with open(os.path.join(output_dir, filename), "wb") as f:
                                f.write(img_byte_arr.getvalue())
                                
                            downloaded_pixel_hashes.add(content_hash)
                            print(f"[Total: {total_category_images_saved}/{MAX_IMAGES_PER_CATEGORY}] Saved Angle {valid_angles_saved}: {filename}")
                            
                            # --- Check Image Limit exactly after saving ---
                            if total_category_images_saved >= MAX_IMAGES_PER_CATEGORY:
                                print(f"\n[LIMIT REACHED] Downloaded {MAX_IMAGES_PER_CATEGORY} images. Moving to next category.")
                                limit_reached = True
                                break 
                                
                    except Exception: pass
            finally:
                if len(driver.window_handles) > 1:
                    driver.close()
                    driver.switch_to.window(main_window)
                    
        # Go to next page if limits aren't reached yet
        if not limit_reached:
            if go_to_next_page(driver):
                time.sleep(4)
                page += 1
            else:
                print("End of category reached.")
                break

def main():
    driver = setup_driver()
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
    
    try:
        establish_session(driver, session)
        for url, folder_path in CATEGORIES.items():
            process_category(session, driver, url, folder_path)
    except KeyboardInterrupt:
        print("\nScript manually stopped.")
    finally:
        try: driver.quit()
        except: pass

if __name__ == "__main__":
    main()