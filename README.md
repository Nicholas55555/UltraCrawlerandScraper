![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/Nicholas55555/UltraCrawlerandScraper/badge)
[![OpenSSF Best Practices](https://bestpractices.devbadge.org/projects/YOUR_PROJECT_ID/badge)](https://bestpractices.dev/projects/YOUR_PROJECT_ID)

# 🕷️ UltraCrawlerAndScraper

A powerful and customizable multithreaded web crawler and scraper built with **Scrapy**, **Tkinter**, and **EasyGUI**. It supports features such as:
- Recursive crawling
- Full web content extraction
- Proxy validation
- User-agent rotation
- Parallel processing
- Custom scraping/crawling modes
- Media downloads (images, videos, audio)
- Custom run time and throttling options

---

## 📦 Features

- ✅ **Proxy support with validation**
- ✅ **Rotating user agents**
- ✅ **Recursive depth-limited crawling**
- ✅ **Scrape or full scrape mode**
- ✅ **Parallel crawler processes**
- ✅ **AutoThrottle and robots.txt support**
- ✅ **HTML, metadata, text, and media downloads**
- ✅ **EasyGUI interface for parameters**
- ✅ **Terminal suppression for cleaner operation**

---

## 🧰 Requirements

- Python 3.7+
- `scrapy`
- `requests`
- `bs4`
- `easygui`

Install dependencies with:

```bash
pip install -r requirements.txt
```

_**requirements.txt**_:
```
scrapy
requests
beautifulsoup4
easygui
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/UltraCrawlerAndScraper.git
cd UltraCrawlerAndScraper
```

### 2. Run the script

```bash
python your_script_name.py
```

### 3. Choose your settings using the GUI:
- Select your **action** (crawl, scrape, full-scrape)
- Input **URL** or **load a list of URLs**
- Select **proxy file** and **user-agent file**
- Choose **depth**, **delay**, **number of proxies**, and more
- Define whether to respect `robots.txt` and enable `AutoThrottle`
- Set **output folder** to store the results

---

## 🧪 Modes Explained

| Mode         | Description |
|--------------|-------------|
| `Crawl`      | Discovers and stores new links recursively |
| `Scrape`     | Downloads only the raw HTML of listed URLs |
| `Full-Scrape`| Extracts metadata, headings, paragraphs, links, and media (images/videos/audio) |

---

## 📂 Output Structure

```
your_output_folder/
├── discovered_urls.txt
├── scraped_urls.txt
├── example_com.html
├── example_com_metadata.txt
├── example_com_paragraphs.txt
├── example_com_links.txt
├── images/
│   ├── img1.jpeg
│   └── ...
├── videos/
│   ├── video1.mp4
├── audio/
│   ├── sound1.mp3
```

---

## 🧠 How it Works

1. User selects the mode and parameters via GUI.
2. The program validates proxies concurrently.
3. Multiple crawler instances run in parallel using `multiprocessing`.
4. Data is saved to disk: HTML, media files, and extracted content.
5. URLs are tracked to avoid duplicates and ensure efficient recursion.

---

## 🔐 Notes

- **Proxy and User-Agent files** should be plain text files with one entry per line.
- This tool does **not** support JavaScript-rendered content. Consider integrating Selenium for advanced scraping needs.
- Proxy validation uses websites like Google, Bing, and HttpBin.

---

## ❗ Disclaimer

This tool is for **educational and ethical use only**. Always respect website terms of service and robots.txt rules.

---

## 👨‍💻 Author

Developed by [Nicholas Taylor].

