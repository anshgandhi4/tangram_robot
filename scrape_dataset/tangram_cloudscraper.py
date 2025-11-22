#!/usr/bin/env python3
"""
Webscraper to download tangram solution images using cloudscraper
This scraper:
1. Visits category pages (geometrical shapes, letters, people, etc.)
2. Extracts links to individual solution pages
3. Visits each solution page and downloads the solution image
"""

from bs4 import BeautifulSoup
import cloudscraper
import os
import time
from urllib.parse import urljoin, urlparse

# Configuration
CATEGORY_URLS = [
    "https://www.tangram-channel.com/tangram-solutions/geometrical-shapes/",
    "https://www.tangram-channel.com/tangram-solutions/letters-numbers-signs/",
    "https://www.tangram-channel.com/tangram-solutions/people/",
    "https://www.tangram-channel.com/tangram-solutions/animals/",
    "https://www.tangram-channel.com/tangram-solutions/usual-objects/",
    "https://www.tangram-channel.com/tangram-solutions/boats/",
    "https://www.tangram-channel.com/tangram-solutions/miscellaneous/",
]
OUTPUT_DIR = "tangrams"

def create_output_directory():
    """Create the output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")
    else:
        print(f"Directory already exists: {OUTPUT_DIR}")

def download_image(scraper, img_url, filename):
    """Download an image from a URL and save it to the output directory"""
    try:
        response = scraper.get(img_url, timeout=15)
        response.raise_for_status()

        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return False

def get_solution_links_from_category(scraper, category_url):
    """Extract all solution page links from a category page"""
    try:
        print(f"\nFetching category page: {category_url}")
        response = scraper.get(category_url, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all links that point to solution pages
        # Solution pages have URLs like: tangram-channel.com/tangrams-pages/tangram-*-solution-*/
        solution_links = []

        for a in soup.find_all('a', href=True):
            href = a['href']
            # Look for links to solution pages
            if '/tangrams-pages/' in href and 'solution' in href:
                full_url = urljoin(category_url, href)
                solution_links.append(full_url)

        # Remove duplicates
        solution_links = list(set(solution_links))
        print(f"Found {len(solution_links)} solution pages")
        return solution_links

    except Exception as e:
        print(f"Error fetching category {category_url}: {e}")
        return []

def get_solution_image_from_page(scraper, solution_url):
    """Extract the solution image from a specific solution page"""
    try:
        response = scraper.get(solution_url, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # The solution image is within <div class="jtpl-section__gutter cc-content-parent">
        # Navigate the specific structure to find the image
        gutter_div = soup.find('div', class_='jtpl-section__gutter')

        if gutter_div:
            # Find the j-imageSubtitle div within the gutter
            image_subtitle_div = gutter_div.find('div', class_=lambda x: x and 'j-imageSubtitle' in x)

            if image_subtitle_div:
                img = image_subtitle_div.find('img')
                if img:
                    # Try to get the src from various attributes
                    src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')

                    # Make sure it's a jimcdn.com URL (the actual solution images)
                    if src and 'jimcdn.com' in src:
                        print(f"  Found solution image: {os.path.basename(src)}")
                        return urljoin(solution_url, src)

        # Fallback: Look for ANY image with jimcdn.com and tangram in the URL
        all_imgs = soup.find_all('img')
        for img in all_imgs:
            src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
            if src and 'jimcdn.com' in src and 'tangram' in src.lower():
                # Skip very small dimensions (thumbnails)
                width = img.get('data-src-width') or img.get('width')
                if width:
                    try:
                        if int(width) >= 300:  # Only images 300px or larger
                            print(f"  Found solution image (fallback): {os.path.basename(src)}")
                            return urljoin(solution_url, src)
                    except:
                        pass
                else:
                    # No width info, assume it's good
                    print(f"  Found solution image (fallback): {os.path.basename(src)}")
                    return urljoin(solution_url, src)

        print(f"  Warning: No solution image found with expected structure")
        return None

    except Exception as e:
        print(f"Error fetching solution page {solution_url}: {e}")
        return None

def scrape_tangram_solutions():
    """Main scraping function"""
    create_output_directory()

    # Create a cloudscraper session
    scraper = cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'windows',
            'mobile': False
        }
    )

    try:
        all_solution_links = []

        # Step 1: Get all solution page links from all categories
        print("="*60)
        print("STEP 1: Collecting solution page links from all categories")
        print("="*60)

        for category_url in CATEGORY_URLS:
            links = get_solution_links_from_category(scraper, category_url)
            all_solution_links.extend(links)
            time.sleep(1)  # Be respectful to the server

        print(f"\n{'='*60}")
        print(f"Total solution pages found: {len(all_solution_links)}")
        print(f"{'='*60}")

        # Step 2: Visit each solution page and download the image
        print("\nSTEP 2: Downloading solution images")
        print("="*60)

        downloaded = 0
        failed = 0

        for i, solution_url in enumerate(all_solution_links, 1):
            print(f"\n[{i}/{len(all_solution_links)}] Processing: {solution_url}")

            # Get the image URL from the solution page
            img_url = get_solution_image_from_page(scraper, solution_url)

            if img_url:
                # Generate filename from the solution URL
                # Extract solution name from URL
                url_parts = solution_url.rstrip('/').split('/')
                solution_name = url_parts[-1] if url_parts else f"solution_{i}"

                # Get file extension from image URL
                img_filename = os.path.basename(urlparse(img_url).path)
                ext = img_filename.split('.')[-1].split('?')[0] if '.' in img_filename else 'jpg'

                filename = f"{solution_name}.{ext}"

                if download_image(scraper, img_url, filename):
                    downloaded += 1
                else:
                    failed += 1
            else:
                print(f"  No image found on this page")
                failed += 1

            # Be respectful to the server
            time.sleep(0.5)

        print(f"\n{'='*60}")
        print(f"DOWNLOAD COMPLETE!")
        print(f"Successfully downloaded: {downloaded}")
        print(f"Failed: {failed}")
        print(f"Total solution pages: {len(all_solution_links)}")
        print(f"Images saved to: {OUTPUT_DIR}/")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    scrape_tangram_solutions()
