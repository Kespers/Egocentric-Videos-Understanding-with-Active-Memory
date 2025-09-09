import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configurazione
source_dir = "/workspace/amego/enigma-51/AMEGO"
AMEGO_IMG_SIZE = (456, 256)
num_threads = 8

test_dirs = [
    "46"
    # "47","49","53","65","66","85","86","88","89",
    # "95","107","131","141","143","144",
]

def resize_and_overwrite(file_path):
    """
    Ridimensiona un'immagine e sovrascrive il file originale.
    """
    try:
        with Image.open(file_path) as img:
            img = img.resize(AMEGO_IMG_SIZE)
            img.save(file_path)
        # Facoltativo: puoi commentare la stampa se sono molte immagini
        # print(f"Immagine ridimensionata: {file_path}")
    except Exception as e:
        print(f"Errore con {file_path}: {e}")

# Creazione della lista completa dei file da ridimensionare
all_files = []
for folder in test_dirs:
    rgb_folder = os.path.join(source_dir, folder, "rgb_frames")
    if not os.path.isdir(rgb_folder):
        print(f"Cartella non trovata: {rgb_folder}")
        continue
    for f in os.listdir(rgb_folder):
        if f.lower().endswith(".jpg"):
            all_files.append(os.path.join(rgb_folder, f))

print(f"Totale immagini da processare: {len(all_files)}")

# Parallelizza il resize con barra di avanzamento
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(resize_and_overwrite, f) for f in all_files]
    for _ in tqdm(as_completed(futures), total=len(futures), desc="Ridimensionamento immagini"):
        pass
