import os
import time
import hashlib
import re
import cv2
import numpy as np
import base64
import requests
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from PIL import Image
from io import BytesIO

# --- CONFIGURE YOUR PATHS & LIMITS HERE ---
BASE_DIR = r"D:\gen jewels\web scraping\Dataset\Candere"

MAX_IMAGES_PER_CATEGORY = 600

# Base URLs
CATEGORIES = {
    "https://www.candere.com/jewellery/necklaces.html": os.path.join(BASE_DIR, "Necklaces"),
    "https://www.candere.com/jewellery/earrings.html": os.path.join(BASE_DIR, "Earrings"),
    "https://www.candere.com/jewellery/bangles-and-bracelets.html": os.path.join(BASE_DIR, "Bangles_Bracelets"),
    "https://www.candere.com/jewellery/rings.html": os.path.join(BASE_DIR, "Rings"),
    "https://www.candere.com/jewellery/mangalsutra.html": os.path.join(BASE_DIR, "Mangalsutra")
}

def setup_driver():
    options = uc.ChromeOptions()
    driver = uc.Chrome(options=options, version_main=153) 
    driver.maximize_window()
    return driver

def destroy_popups(driver):
    """Forcefully destroys Candere's popups to prevent DOM freezing."""
    try:
        driver.execute_script("""
            Array.from(document.querySelectorAll('button, a, div')).forEach(b => {
                let text = (b.innerText || '').toLowerCase();
                if(text === 'x' || text.includes('dismiss') || text.includes('close') || text.includes('no thanks') || text.includes('later')) {
                    b.click();
                }
            });
            let style = document.createElement('style');
            style.innerHTML = '.moe-popup, .chat-widget, .overlay, [id*="onesignal"], iframe { display: none !important; opacity: 0 !important; pointer-events: none !important; }';
            document.head.appendChild(style);
            document.body.style.overflow = 'auto';
        """)
    except: pass

def establish_session(driver, session):
    print("Visiting homepage to establish session...")
    driver.get("https://www.candere.com/")
    time.sleep(8) 
    destroy_popups(driver)
    
    # Sync cookies to requests session for the fallback downloader
    for cookie in driver.get_cookies():
        session.cookies.set(cookie['name'], cookie['value'], domain=cookie['domain'])
    print("Session verified!")

def download_image(driver, session, url):
    """Dual-Engine Downloader: Tries JS internal fetch, falls back to Requests."""
    driver.set_script_timeout(10)
    js_script = """
        var uri = arguments[0];
        var callback = arguments[1];
        fetch(uri)
            .then(response => response.blob())
            .then(blob => {
                var reader = new FileReader();
                reader.onloadend = function() { callback(reader.result); }
                reader.readAsDataURL(blob);
            })
            .catch(error => callback('ERROR'));
    """
    try:
        # Try JS Download
        result = driver.execute_async_script(js_script, url)
        if isinstance(result, str) and result.startswith('data:image'):
            base64_str = result.split(',')[1]
            return base64.b64decode(base64_str)
    except: pass
    
    # Fallback to Requests Download if JS fails due to CORS
    try:
        res = session.get(url, timeout=10)
        if res.status_code == 200:
            return res.content
    except: pass
    
    return None

def is_valid_product_image(img_pil):
    """Checks for White Backgrounds, Face Detection, and blocks Grayscale Placeholders."""
    try:
        # Handle transparent PNGs turning black
        if img_pil.mode in ('RGBA', 'LA') or (img_pil.mode == 'P' and 'transparency' in img_pil.info):
            alpha = img_pil.convert('RGBA').split()[-1]
            bg = Image.new("RGB", img_pil.size, (255, 255, 255))
            bg.paste(img_pil, mask=alpha)
            cv_img = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)
        else:
            cv_img = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
            
        h, w = cv_img.shape[:2]
        
        # 1. GRAYSCALE PLACEHOLDER FILTER (Kills 'Image Not Found')
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        mean_saturation = np.mean(hsv[:, :, 1])
        if mean_saturation < 1: 
            return False, "Skipped (Grayscale Placeholder Detected)"

        # 2. WHITE BACKGROUND FILTER (Kills Wear/Model images)
        corners = [
            cv_img[0:50, 0:50],          
            cv_img[0:50, w-50:w],        
            cv_img[h-50:h, 0:50],        
            cv_img[h-50:h, w-50:w]       
        ]
        
        corner_brightness = sum(np.mean(c) for c in corners) / 4
        # Lowered to 160 to safely allow strong jewelry shadows to pass
        if corner_brightness < 160: 
            return False, f"Skipped (Model/Studio Background. Brightness: {corner_brightness:.1f})"
            
        # 3. FACE DETECTOR
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        
        if len(faces) > 0 or len(profiles) > 0:
            return False, "Skipped (Human Face Detected)"
            
        return True, "Valid"
    except Exception:
        return True, "Valid" 

def extract_all_links(driver):
    js_script = """
        let links = new Set();
        let bad = ['cart', 'login', 'contact', 'about', 'wishlist', 'gift', 'store', 'blog', 'policy'];
        document.querySelectorAll('a').forEach(a => {
            let href = a.href || '';
            if (href.startsWith('http') && !bad.some(word => href.toLowerCase().includes(word))) {
                if (a.querySelector('img')) {
                    links.add(href);
                }
            }
        });
        return Array.from(links);
    """
    return driver.execute_script(js_script)

def fetch_links_via_pagination(driver, base_url, target_amount=250):
    print(f"Fetching links via ?p= pagination up to {target_amount} products...")
    all_links = set()
    page = 1
    
    while len(all_links) < target_amount:
        paginated_url = f"{base_url}?p={page}"
        print(f"Loading Page {page}: {paginated_url}")
        
        driver.get(paginated_url)
        time.sleep(4)
        destroy_popups(driver)
        
        driver.execute_script("window.scrollBy(0, 800);")
        time.sleep(1)
        
        links = extract_all_links(driver)
        if not links:
            print("No more products found on this page. Stopping pagination.")
            break
            
        all_links.update(links)
        print(f"Loaded {len(all_links)} actual product links so far...")
        
        page += 1
        if page > 12: break 
            
    return list(all_links)

def extract_exact_gallery_images(driver):
    """
    UNIVERSAL PATH EXTRACTOR: Looks at all images and extracts anything 
    hosted in Candere's jewelry database, completely ignoring class names.
    """
    js_script = """
        let urls = new Set();
        document.querySelectorAll('img').forEach(img => {
            let src = img.getAttribute('data-imgurl') || img.getAttribute('data-zoom-image') || img.getAttribute('data-src') || img.src || '';
            if (src && src.startsWith('http')) {
                // Check if it's an actual product image path
                if (src.includes('/media/jewellery/images/') || src.includes('/catalog/product/')) {
                    if (!src.includes('banner') && !src.includes('logo')) {
                        urls.add(src.split('?')[0]);
                    }
                }
            }
        });
        return Array.from(urls);
    """
    return driver.execute_script(js_script)

def get_high_res_urls(url):
    variants = []
    if '?' in url: url = url.split('?')[0]
    
    # Strip thumbnail dimensions to force original database image
    clean_res = re.sub(r'/\d+x\d+/', '/', url)
    if clean_res != url:
        variants.append(clean_res)

    variants.append(url) 
    return list(dict.fromkeys(variants))

def process_category(session, driver, category_url, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}\nScanning Category: {os.path.basename(output_dir)}\n{'='*60}")
    
    product_links = fetch_links_via_pagination(driver, category_url, target_amount=250)
    
    if not product_links:
        print("No products found. Skipping category.")
        return
        
    print(f"Found {len(product_links)} verified products. Beginning deep dive...")
    downloaded_pixel_hashes = set() 
    total_category_images_saved = 0
    main_window = driver.current_window_handle
    
    for count, product_url in enumerate(product_links, 1):
        if total_category_images_saved >= MAX_IMAGES_PER_CATEGORY:
            break
            
        try:
            driver.execute_script(f"window.open('{product_url}', '_blank');")
            driver.switch_to.window(driver.window_handles[-1])
            time.sleep(4) 

            gallery_urls = extract_exact_gallery_images(driver)
            
            if not gallery_urls:
                print(f"  -> [Product {count}] No gallery images found on page.")
                continue

            valid_angles_saved = 0
            
            for exact_url in gallery_urls:
                if total_category_images_saved >= MAX_IMAGES_PER_CATEGORY:
                    break
                    
                for attempt_url in get_high_res_urls(exact_url):
                    try:
                        img_bytes = download_image(driver, session, attempt_url)
                        
                        if img_bytes:
                            img = Image.open(BytesIO(img_bytes))
                            
                            if img.width < 450 or img.height < 450: continue
                            aspect_ratio = img.width / img.height
                            if aspect_ratio < 0.6 or aspect_ratio > 1.4: continue
                            
                            if img.mode not in ("RGBA", "RGB"): img = img.convert("RGBA")
                            img_byte_arr = BytesIO()
                            img.save(img_byte_arr, format='PNG')
                            content_hash = hashlib.md5(img_byte_arr.getvalue()).hexdigest()
                            
                            if content_hash in downloaded_pixel_hashes:
                                continue 
                            
                            is_valid, skip_reason = is_valid_product_image(img)
                            if not is_valid:
                                print(f"  -> {skip_reason}")
                                break # Failed visual filter, move to next image
                            
                            valid_angles_saved += 1
                            total_category_images_saved += 1
                            filename = f"Candere_{os.path.basename(output_dir)}_Prod{count}_Angle{valid_angles_saved}_{img.width}x{img.height}_{content_hash[:6]}.png"
                            
                            with open(os.path.join(output_dir, filename), "wb") as f:
                                f.write(img_byte_arr.getvalue())
                                
                            downloaded_pixel_hashes.add(content_hash)
                            print(f"[Total: {total_category_images_saved}/{MAX_IMAGES_PER_CATEGORY}] Saved Angle {valid_angles_saved}: {filename}")
                            break # Found High-Res successfully, move to next gallery image
                            
                    except Exception as e:
                        pass
        finally:
            if len(driver.window_handles) > 1:
                driver.close()
                driver.switch_to.window(main_window)
                
    print(f"\n[LIMIT REACHED/END] Downloaded {total_category_images_saved} images for {os.path.basename(output_dir)}.")

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