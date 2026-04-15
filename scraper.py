"""
scraper.py — Telecom Egypt Structured Scraper
يعتمد على URLs ثابتة (مش crawling)
ويحفظ البيانات بشكل نظيف وجاهز لـ RAG
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import hashlib

# =========================================
# 📌 URLs المهمة (من navigation الموقع)
# =========================================
URLS = [
    # Main
    "https://te.eg/wps/portal/te/Personal",
    
    #5G
    "https://te.eg/wps/portal/te/Personal/5G",
    "https://te.eg/wps/portal/te/Personal/WE-Air-Postpaid",
    "https://te.eg/wps/portal/te/Personal/WE-Air-Prepaid",

    "https://te.eg/wps/portal/te/Personal/5G/Mobile-Postpaid",
    "https://te.eg/wps/portal/te/Personal/5G/Mobile-Prepaid",
    "https://te.eg/wps/portal/te/Personal/5G/Mobile-MI",
    "https://te.eg/wps/portal/te/Personal/Mobile/Nitro-mobile-internet",
    

    # Mobile
    "https://te.eg/wps/portal/te/Personal/Mobile",
    "https://te.eg/wps/portal/te/Personal/Mobile/Prepaid-12PT",
    "https://te.eg/wps/portal/te/Personal/Mobile/Prepaid-Agda3-Sha7na",

    "https://te.eg/wps/portal/te/Personal/Mobile/Control-WE-MIX",
    "https://te.eg/wps/portal/te/Personal/Mobile/SuperKix",
    "https://te.eg/wps/portal/te/Personal/Mobile/Control-Tazbeet",
    "https://te.eg/wps/portal/te/Personal/Mobile/Control-WE-Club",
    "https://te.eg/wps/portal/te/Personal/Mobile/WE-Gold",
    "https://te.eg/wps/portal/te/Personal/Mobile/WEGoldMobile",

    "https://te.eg/wps/portal/te/Personal/Mobile/Nitro-mifi",
    "https://te.eg/wps/portal/te/Personal/Mobile/Nitro-mobile-internet",
    

    # Home
    
    "https://te.eg/wps/portal/te/Personal/WEInternet",
    "https://te.eg/wps/portal/te/Personal/WELandline",
    "https://te.eg/wps/portal/te/Personal/WE-Air-Prepaid",
    "https://te.eg/wps/portal/te/Personal/WE-Air-Postpaid",
    
    

    # Services
    "https://www.te.eg/wps/portal/te/Personal/Mobile/Services",
    "https://www.te.eg/wps/portal/te/Personal/Mobile/Balance-Services",
    "https://www.te.eg/wps/portal/te/Personal/Mobile/Fixed-Call-Services",
    "https://www.te.eg/wps/portal/te/Personal/Mobile/Fixed-Broadband",
    "https://www.te.eg/wps/portal/te/Personal/Mobile/International-Roaming",
    "https://www.te.eg/wps/portal/te/Personal/Mobile/Entertainment",   
    "https://www.te.eg/wps/portal/te/Personal/Mobile/Other-Services",

    # Devices
    
    "https://te.eg/wps/portal/te/Personal/Devices/Routers",
    "https://te.eg/wps/portal/te/Personal/Devices/Fixed-Landline-Phones",
    "https://te.eg/wps/portal/te/Personal/Devices/Mobile-Phones",
    "https://te.eg/wps/portal/te/Personal/Devices/Accessories",
    "https://te.eg/wps/portal/te/Personal/Devices/USB-Modems",
    "https://te.eg/wps/portal/te/Personal/Devices/4G-Routers",
    "https://te.eg/wps/portal/te/Personal/Devices/5G-Devices-Mobile",
    "https://te.eg/wps/portal/te/Personal/Devices/5G-Devices-Home",

    # Promotions
    "https://te.eg/wps/portal/te/Personal/Promotions",
    "https://te.eg/wps/portal/te/Personal/Promotions/Internet-Promotions/",
    "https://te.eg/wps/portal/te/Personal/Promotions/Landline-Promotions/",
    "https://te.eg/wps/portal/te/Personal/Promotions/Mobile-Promotions/",
    "https://te.eg/wps/portal/te/Personal/Promotions/Other-Promotions/",
    "https://te.eg/wps/portal/te/Personal/All-Promotions/",  
    
]
URLS = list(dict.fromkeys(URLS))

# =========================================
# Headers (مهم عشان مايتعملكش block)
# =========================================
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "ar,en;q=0.9"
}

# =========================================
# تنظيف النص
# =========================================
def clean_text(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


# =========================================
# استخراج المحتوى
# =========================================
def extract_text(html):
    soup = BeautifulSoup(html, "lxml")

    # remove noise
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "button"]):
        tag.decompose()

    # حاول تجيب main content بس
    main = soup.find("main") or soup.find("article")

    if main:
        text = main.get_text(separator="\n")
    else:
        text = soup.get_text(separator="\n")

    return clean_text(text)


# =========================================
# scrape page
# =========================================
def scrape_page(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=20)
        res.raise_for_status()

        text = extract_text(res.text)

        if len(text) < 200:
            print(f"⚠️ تجاهل (قصير): {url}")
            return None

        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.title.string if soup.title else url.split("/")[-1]
        title = title.string if title else "No Title"

        return {
            "url": url,
            "title": title.strip(),
            "content": text
        }

    except Exception as e:
        print(f"❌ Error: {url} → {e}")
        return None

def remove_duplicates(pages):
    seen = set()
    unique_pages = []

    for page in pages:
        content_hash = hashlib.md5(page["content"].encode()).hexdigest()

        if content_hash not in seen:
            seen.add(content_hash)
            unique_pages.append(page)

    return unique_pages
# =========================================
# حفظ البيانات
# =========================================

def save_pages(pages, output_dir="data/website_pages"):
    os.makedirs(output_dir, exist_ok=True)

    for i, page in enumerate(pages):
        file_path = f"{output_dir}/page_{i:03d}.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"URL: {page['url']}\n")
            f.write(f"Title: {page['title']}\n")
            f.write("="*50 + "\n")
            f.write(f"Source: {page['url']}\n")
            f.write(f"Title: {page['title']}\n\n")
            f.write(page["content"])

    print(f"\n✅ Saved {len(pages)} pages")


# =========================================
# main run
# =========================================
def run_scraper():
    print("🚀 Start Scraping Telecom Egypt...")

    pages = []

    for i, url in enumerate(URLS, 1):
        print(f"[{i}/{len(URLS)}] {url}")

        page = scrape_page(url)

        if page:
            pages.append(page)
            print(f"   ✅ {page['title']} ({len(page['content'])} chars)")

        time.sleep(1)  # احترام السيرفر
    pages = remove_duplicates(pages)
    save_pages(pages)

    print("\n🎉 Done!")


if __name__ == "__main__":
    run_scraper()