import nltk

def download_nltk_dependencies():
    required_packages = [
        'punkt',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng'
    ]
    
    for package in required_packages:
        print(f"Downloading {package}...")
        try:
            nltk.download(package)
            print(f"Successfully downloaded {package}")
        except Exception as e:
            print(f"Error downloading {package}: {e}")

if __name__ == "__main__":
    print("Starting NLTK dependencies download...")
    download_nltk_dependencies()
    print("Finished downloading NLTK dependencies") 