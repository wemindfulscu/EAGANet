# models/download_models.py
import os
import requests
from tqdm import tqdm

MODELS = {
    "medsam_vit_b.pth": "https://dl.fbaipublicfiles.com/segment_anything/medsam_vit_b.pth",
    "groundingdino_swint_ogc.pth": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
}

def download_file(url, local_filename):
    """Downloads a file from a URL with a progress bar."""
    print(f"Downloading {local_filename} from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=block_size):
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong during download.")
    else:
        print(f"{local_filename} downloaded successfully.")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    if not os.path.exists("."):
        os.makedirs(".")

    for filename, url in MODELS.items():
        if not os.path.exists(filename):
            download_file(url, filename)
        else:
            print(f"{filename} already exists. Skipping download.")