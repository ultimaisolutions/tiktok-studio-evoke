import json
import sys

def extract_urls(json_path: str, output_path: str = "urls.txt"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    urls = []
    for item in data:
        if "url" in item and item["url"]:
            urls.append(item["url"])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(urls))

    print(f"Extracted {len(urls)} URLs to {output_path}")

if __name__ == "__main__":
    json_file = sys.argv[1]
    extract_urls(json_file)
#usage : python extract_urls.py path/to/input.json