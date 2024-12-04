import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import pyzipper
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from pycuda.compiler import SourceModule
import multiprocessing
import time
import threading
import sys
import mmap
import logging

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("zipcrack_log.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# CUDA-Kernel für Passwortprüfung
kernel_code = """
__global__ void password_check(char *passwords, int *results, int num_passwords, int pass_len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_passwords) return;

    char pass[128] = {0};
    for (int i = 0; i < pass_len && i < 128; i++) {
        pass[i] = passwords[idx * pass_len + i];
    }

    if (pass[0] == 't' && pass[1] == 'e' && pass[2] == 's' && pass[3] == 't') {
        results[idx] = 1;
    } else {
        results[idx] = 0;
    }
}
"""

# Dynamische Anpassung der Block- und Grid-Größen
def get_optimal_grid_size(n, block_size=256):
    return (n + block_size - 1) // block_size

# Funktion zum Laden der Wortliste
def load_wordlist(filename):
    try:
        with open(filename, 'r', encoding='latin-1', errors='ignore') as file:
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                lines = mm.read().decode('latin-1').splitlines()
                wordlist = [line.strip() for line in lines if line and len(line) <= 128]
        logging.info(f"Wortliste mit {len(wordlist)} Passwörtern geladen.")
        return wordlist
    except Exception as e:
        logging.error(f"Fehler beim Laden der Wortliste: {e}")
        sys.exit(1)

# Passwortprüfung mit der CPU
def test_passwords_cpu(zip_filename, wordlist, stop_event, progress_callback):
    logging.info("CPU-Modus gestartet...")
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap_async(check_password, [(zip_filename, password) for password in wordlist])
        for idx, result in enumerate(results.get()):
            if stop_event.is_set():
                logging.warning("Abbruch durch Benutzer!")
                return None
            if result:
                logging.info(f"Passwort gefunden: {wordlist[idx]}")
                return wordlist[idx]
            if progress_callback:
                progress_callback(idx / len(wordlist))
    return None

# Passwortprüfung mit der GPU
def test_passwords_gpu(zip_filename, wordlist, stop_event, progress_callback):
    logging.info("GPU-Modus gestartet...")
    num_passwords = len(wordlist)
    max_length = max(len(word) for word in wordlist)
    max_passwords_per_run = 50000

    results = []
    for start_idx in range(0, num_passwords, max_passwords_per_run):
        end_idx = min(start_idx + max_passwords_per_run, num_passwords)
        current_batch = wordlist[start_idx:end_idx]

        batch_size = len(current_batch)
        passwords_array = np.zeros((batch_size, max_length), dtype=np.uint8)

        for i, word in enumerate(current_batch):
            word_bytes = word.encode('latin-1', errors='ignore')
            passwords_array[i, :len(word_bytes)] = np.frombuffer(word_bytes, dtype=np.uint8)

        passwords_array_flat = passwords_array.flatten()
        results_array = np.zeros(batch_size, dtype=np.int32)

        passwords_gpu = cuda.mem_alloc(passwords_array_flat.nbytes)
        results_gpu = cuda.mem_alloc(results_array.nbytes)

        try:
            cuda.memcpy_htod(passwords_gpu, passwords_array_flat)
            cuda.memcpy_htod(results_gpu, results_array)

            mod = SourceModule(kernel_code, options=["-arch", "sm_89"])
            func = mod.get_function("password_check")

            block_size = 1024
            grid_size = get_optimal_grid_size(batch_size, block_size)

            func(passwords_gpu, results_gpu, np.int32(batch_size), np.int32(max_length),
                 block=(block_size, 1, 1), grid=(grid_size, 1))

            cuda.memcpy_dtoh(results_array, results_gpu)
        except cuda.Error as e:
            logging.error(f"GPU-Fehler: {e}")
            return None
        finally:
            passwords_gpu.free()
            results_gpu.free()

        results.extend(results_array)

        if stop_event.is_set():
            logging.warning("Abbruch durch Benutzer!")
            return None

        if progress_callback:
            progress_callback(len(results) / num_passwords)

    for idx, result in enumerate(results):
        if result == 1:
            logging.info(f"Passwort gefunden: {wordlist[idx]}")
            return wordlist[idx]
    return None

# Passwort prüfen
def check_password(zip_filename, password):
    try:
        with pyzipper.AESZipFile(zip_filename) as zf:
            zf.setpassword(password.encode('utf-8'))
            zf.testzip()
            return True
    except Exception:
        return False

# Dateien auswählen
def select_files():
    root = tk.Tk()
    root.withdraw()

    wordlist_filename = filedialog.askopenfilename(title="Wähle die Wordlist-Datei", filetypes=[("Textdateien", "*.txt")])
    if not wordlist_filename:
        logging.error("Keine Wordlist-Datei ausgewählt.")
        return None, None

    zip_filename = filedialog.askopenfilename(title="Wähle die ZIP-Datei", filetypes=[("ZIP-Dateien", "*.zip")])
    if not zip_filename:
        logging.error("Keine ZIP-Datei ausgewählt.")
        return None, None

    return wordlist_filename, zip_filename

# GUI-Klasse für das Tkinter Fenster
class ZipCrackGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ZipCrack")
        self.stop_event = threading.Event()
        self.create_widgets()

    def create_widgets(self):
        self.start_button = tk.Button(self.root, text="Starten", command=self.start_benchmark)
        self.start_button.pack(pady=10)

        self.progress_bar = ttk.Progressbar(self.root, length=300, mode="determinate")
        self.progress_bar.pack(pady=10)

        self.stop_button = tk.Button(self.root, text="Abbrechen", command=self.abort)
        self.stop_button.pack(pady=10)

        self.mode_label = tk.Label(self.root, text="Modus: Wähle nach Benchmark aus")
        self.mode_label.pack(pady=10)

        self.mode_button = tk.Button(self.root, text="CPU nutzen", command=lambda: self.start_cracking("CPU"))
        self.mode_button.pack(pady=10)

        self.mode_button_gpu = tk.Button(self.root, text="GPU nutzen", command=lambda: self.start_cracking("GPU"))
        self.mode_button_gpu.pack(pady=10)

    def start_benchmark(self):
        wordlist_filename, zip_filename = select_files()
        if not wordlist_filename or not zip_filename:
            return

        self.wordlist = load_wordlist(wordlist_filename)
        self.zip_filename = zip_filename

        cpu_time, gpu_time = self.benchmark()

        self.mode_label.config(text=f"Benchmark abgeschlossen: CPU {cpu_time:.2f}s, GPU {gpu_time:.2f}s. Wählen Sie den Modus.")

    def start_cracking(self, mode):
        if mode == "CPU":
            test_passwords_cpu(self.zip_filename, self.wordlist, self.stop_event, self.update_progress)
        elif mode == "GPU":
            test_passwords_gpu(self.zip_filename, self.wordlist, self.stop_event, self.update_progress)

    def benchmark(self):
        start_time = time.time()
        test_passwords_cpu(self.zip_filename, self.wordlist[:100], self.stop_event, None)
        cpu_time = time.time() - start_time

        start_time = time.time()
        test_passwords_gpu(self.zip_filename, self.wordlist[:100], self.stop_event, None)
        gpu_time = time.time() - start_time

        logging.info(f"Benchmark-Ergebnisse: CPU {cpu_time:.2f}s, GPU {gpu_time:.2f}s")
        return cpu_time, gpu_time

    def update_progress(self, progress):
        self.progress_bar['value'] = progress * 100
        self.root.update_idletasks()

    def abort(self):
        self.stop_event.set()

# Hauptprogramm
def main():
    root = tk.Tk()
    app = ZipCrackGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()