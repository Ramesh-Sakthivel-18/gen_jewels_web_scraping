import os
import time
import hashlib
import re
import cv2
import numpy as np
import base64
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from PIL import Image
from io import BytesIO

# --- CONFIGURE YOUR PATHS & LIMITS HERE ---
BASE_DIR = r"D:\gen jewels\web scraping\Dataset\KalyanJewellers"

MAX_IMAGES_PER_CATEGORY = 600

CATEGORIES = {
    "https://www.kalyanjewellers.net/Jewellery/Gold/necklace.php": os.path.join(BASE_DIR, "Necklaces"),
    "https://www.kalyanjewellers.net/Jewellery/Gold/earrings.php": os.path.join(BASE_DIR, "Earrings"),
    "https://www.kalyanjewellers.net/Jewellery/Gold/rings.php": os.path.join(BASE_DIR, "Rings"),
    "https://www.kalyanjewellers.net/Jewellery/Gold/pendant.php": os.path.join(BASE_DIR, "Pendants"),
    "https://www.kalyanjewellers.net/Jewellery/Gold/bangles.php": os.path.join(BASE_DIR, "Bangles")
}

def setup_driver():
    options = uc.ChromeOptions()
    # Matches standard Chrome version 153. Update if your Chrome updates!
    driver = uc.Chrome(options=options, version_main=153) 
    driver.maximize_window()
    return driver

def destroy_popups(driver):
    """Destroys Kalyan's location, cookie, or newsletter popups."""
    try:
        driver.execute_script("""
            Array.from(document.querySelectorAll('button, a')).forEach(b => {
                let text = (b.innerText || '').toLowerCase();
                if(text.includes('dismiss') || text.includes('close') || text.includes('skip') || text.includes('no thanks')) {
                    b.click();
                }
            });
            let style = document.createElement('style');
            style.innerHTML = '.modal, .popup, .overlay, [role="dialog"], iframe { display: none !important; opacity: 0 !important; pointer-events: none !important; }';
            document.head.appendChild(style);
            document.body.style.overflow = 'auto';
        """)
    except: pass

def establish_session(driver):
    print("Visiting homepage to establish session...")
    driver.get("https://www.kalyanjewellers.net/")
    time.sleep(8) 
    destroy_popups(driver)
    print("Session verified!")

def download_image_with_js(driver, url):
    """Downloads image via browser internals to bypass CDN blocks."""
    driver.set_script_timeout(15)
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
        result = driver.execute_async_script(js_script, url)
        if isinstance(result, str) and result.startswith('data:image'):
            base64_str = result.split(',')[1]
            return base64.b64decode(base64_str)
    except: pass
    return None

def is_valid_product_image_white_bg(img_pil):
    """Checks for White backgrounds (removes studios/models) and runs Face Detection."""
    try:
        cv_img = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
        h, w = cv_img.shape[:2]
        
        corners = [
            cv_img[0:50, 0:50],          
            cv_img[0:50, w-50:w],        
            cv_img[h-50:h, 0:50],        
            cv_img[h-50:h, w-50:w]       
        ]
        
        corner_brightness = sum(np.mean(c) for c in corners) / 4
        
        # Kalyan uses very bright white backgrounds for pure products. 
        # This will perfectly delete skin tones, brown backgrounds, and dark nature backgrounds.
        if corner_brightness < 210: 
            return False, f"Skipped (Model/Nature Background. Brightness: {corner_brightness:.1f})"
            
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
    """Grabs product links by looking for a-tags with images, excluding standard UI links."""
    js_script = """
        let links = new Set();
        let bad = ['cart', 'login', 'category', 'contact', 'about', 'wishlist', 'gift', 'investor', 'store'];
        document.querySelectorAll('a').forEach(a => {
            let href = a.href || '';
            if (href.startsWith('http') && !bad.some(word => href.toLowerCase().includes(word))) {
                // If the link wraps an image, it is a product in the grid
                if (a.querySelector('img')) {
                    links.add(href);
                }
            }
        });
        return Array.from(links);
    """
    return driver.execute_script(js_script)

def scroll_until_loaded(driver, target_amount=200):
    print(f"Scrolling smoothly to fetch {target_amount} products...")
    destroy_popups(driver)
    
    all_links = set()
    
    for _ in range(30): 
        links = extract_all_links(driver)
        all_links.update(links)
        
        print(f"Loaded {len(all_links)} actual product links so far...")
        
        if len(all_links) >= target_amount:
            print(f"Reached {len(all_links)} links. Stopping scroll immediately.")
            break
            
        # Human-like smooth scroll
        driver.execute_script("window.scrollBy(0, 800);")
        time.sleep(1.5)
        driver.execute_script("window.scrollBy(0, 800);")
        time.sleep(2) 
        
    return list(all_links)

def trigger_slider_and_extract_images(driver):
    # Click any possible slider arrows invisibly
    try:
        driver.execute_script("""
            let nextArrows = document.querySelectorAll('.slick-next, .swiper-button-next, [aria-label="Next"], [class*="next"], [class*="Arrow"]');
            nextArrows.forEach(arrow => {
                for (let i = 0; i < 5; i++) {
                    setTimeout(() => arrow.click(), i * 400);
                }
            });
        """)
        time.sleep(2.5) 
    except: pass

    # Grab all images
    js_script = """
        let urls = new Set();
        document.querySelectorAll('img').forEach(img => {
            let src = img.getAttribute('src') || '';
            let dataSrc = img.getAttribute('data-src') || '';
            let srcset = img.getAttribute('srcset') || '';
            
            let bestUrl = src || dataSrc;
            if (srcset) {
                let sources = srcset.split(',').map(s => s.trim());
                bestUrl = sources[sources.length - 1].split(' ')[0] || bestUrl;
            }
            
            if (bestUrl && bestUrl.startsWith('http')) {
                urls.add(bestUrl);
            }
        });
        return Array.from(urls);
    """
    return driver.execute_script(js_script)

def get_high_res_urls(url):
    """Attempts to remove thumbnail sizing parameters from the URL."""
    variants = []
    # Maximize size parameters if they exist
    high_res = re.sub(r'\b(w|width|h|height|sw|sh|size)=\d+', r'\1=2000', url)
    # Remove dimension suffixes like -300x300.jpg or _500x500.jpg
    high_res = re.sub(r'[-_]\d+x\d+(\.[a-zA-Z]+)$', r'\1', high_res)
    
    if high_res != url:
        variants.append(high_res)
    variants.append(url)
    return variants

def process_category(driver, category_url, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}\nScanning Category: {os.path.basename(output_dir)}\n{'='*60}")
    
    driver.get(category_url)
    time.sleep(6)
    
    product_links = scroll_until_loaded(driver, target_amount=250)
    
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
            
            # Make sure we didn't get redirected back to home
            if driver.current_url.rstrip('/') == "https://www.kalyanjewellers.net":
                continue

            gallery_urls = trigger_slider_and_extract_images(driver)
            valid_angles_saved = 0
            
            for base_img_url in gallery_urls:
                if total_category_images_saved >= MAX_IMAGES_PER_CATEGORY:
                    break
                    
                for attempt_url in get_high_res_urls(base_img_url):
                    try:
                        img_bytes = download_image_with_js(driver, attempt_url)
                        
                        if img_bytes:
                            img = Image.open(BytesIO(img_bytes))
                            
                            # 1. Size Filter: Throws away UI logos and buttons
                            if img.width < 450 or img.height < 450: continue
                            
                            # 2. Shape Filter: Throws away tall models and wide banners
                            aspect_ratio = img.width / img.height
                            if aspect_ratio < 0.6 or aspect_ratio > 1.4: continue
                            
                            if img.mode not in ("RGBA", "RGB"): img = img.convert("RGBA")
                            img_byte_arr = BytesIO()
                            img.save(img_byte_arr, format='PNG')
                            content_hash = hashlib.md5(img_byte_arr.getvalue()).hexdigest()
                            
                            # 3. Duplicate Filter
                            if content_hash in downloaded_pixel_hashes:
                                continue 
                            
                            # 4. Background & Face Filter (Removes Model Wear Images)
                            is_valid, skip_reason = is_valid_product_image_white_bg(img)
                            if not is_valid:
                                print(f"  -> {skip_reason}")
                                break 
                            
                            valid_angles_saved += 1
                            total_category_images_saved += 1
                            filename = f"Kalyan_{os.path.basename(output_dir)}_Prod{count}_Angle{valid_angles_saved}_{img.width}x{img.height}_{content_hash[:6]}.png"
                            
                            with open(os.path.join(output_dir, filename), "wb") as f:
                                f.write(img_byte_arr.getvalue())
                                
                            downloaded_pixel_hashes.add(content_hash)
                            print(f"[Total: {total_category_images_saved}/{MAX_IMAGES_PER_CATEGORY}] Saved Angle {valid_angles_saved}: {filename}")
                            break 
                            
                    except Exception: pass
        finally:
            if len(driver.window_handles) > 1:
                driver.close()
                driver.switch_to.window(main_window)
                
    print(f"\n[LIMIT REACHED/END] Downloaded {total_category_images_saved} images for {os.path.basename(output_dir)}.")

def main():
    driver = setup_driver()
    
    try:
        establish_session(driver)
        for url, folder_path in CATEGORIES.items():
            process_category(driver, url, folder_path)
    except KeyboardInterrupt:
        print("\nScript manually stopped.")
    finally:
        try: driver.quit()
        except: pass

if __name__ == "__main__":
    main()