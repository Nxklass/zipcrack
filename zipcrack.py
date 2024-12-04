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
from tqdm import tqdm

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
    if (idx < num_passwords) {
        char pass[128];
        for (int i = 0; i < pass_len; i++) {
            pass[i] = passwords[idx * pass_len + i];
        }

        // Beispiel für Passwortprüfung
        if (pass[0] == 't' && pass[1] == 'e' && pass[2] == 's' && pass[3] == 't') {
            results[idx] = 1;  // Passwort gefunden
        } else {
            results[idx] = 0;  // Passwort nicht gefunden
        }
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
                wordlist = [line.strip() for line in lines if line]
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
                progress_callback(idx / len(wordlist))  # Fortschritt aktualisieren
    return None

# Passwortprüfung mit der GPU
def test_passwords_gpu(zip_filename, wordlist, stop_event, progress_callback):
    """Testet die Passwörter auf der GPU mit CUDA."""
    logging.info("GPU-Modus gestartet...")
    num_passwords = len(wordlist)
    max_length = max(len(word) for word in wordlist)

    # Passwort-Wörter auf CUDA-kompatible Form vorbereiten
    passwords_array = np.zeros((num_passwords, max_length), dtype=np.uint8)
    for i, word in enumerate(wordlist):
        word_bytes = word.encode('latin-1', errors='ignore')  # Kodiert jedes Passwort in gültige Bytes
        passwords_array[i, :len(word_bytes)] = np.frombuffer(word_bytes, dtype=np.uint8)

    passwords_array = passwords_array.flatten()

    # Resultate initialisieren (0 für Misserfolg, 1 für Erfolg)
    results_array = np.zeros(num_passwords, dtype=np.int32)

    # CUDA-Streams und Speicher vorbereiten
    try:
        passwords_gpu = cuda.mem_alloc(passwords_array.nbytes)
        results_gpu = cuda.mem_alloc(results_array.nbytes)
    except cuda.Error as e:
        logging.error(f"Fehler beim Zuweisen von GPU-Speicher: {e}")
        return None

    # Kopiere Daten auf die GPU
    try:
        cuda.memcpy_htod(passwords_gpu, passwords_array)
        cuda.memcpy_htod(results_gpu, results_array)
    except cuda.Error as e:
        logging.error(f"Fehler beim Kopieren der Daten auf die GPU: {e}")
        return None

    # CUDA-Kernel kompilieren und starten
    try:
        mod = SourceModule(kernel_code, options=["-arch", "sm_89"])
        func = mod.get_function("password_check")
    except cuda.Error as e:
        logging.error(f"Fehler beim Kompilieren des CUDA-Kernels: {e}")
        return None

    # Block- und Grid-Größe anpassen
    block_size = 256  # Statt 1024, versuche eine kleinere Blockgröße
    grid_size = get_optimal_grid_size(num_passwords, block_size)

    # Führe den Kernel aus
    try:
        func(passwords_gpu, results_gpu, np.int32(num_passwords), np.int32(max_length),
             block=(block_size, 1, 1), grid=(grid_size, 1))
    except cuda.Error as e:
        logging.error(f"Fehler beim Ausführen des CUDA-Kernels: {e}")
        return None

    # Kopiere die Ergebnisse zurück auf die CPU
    try:
        cuda.memcpy_dtoh(results_array, results_gpu)
    except cuda.Error as e:
        logging.error(f"Fehler beim Kopieren der Ergebnisse von der GPU: {e}")
        return None

    # Ergebnisse ausgeben und stoppen, falls abgebrochen
    for idx, result in enumerate(results_array):
        if stop_event.is_set():  # Prüfe, ob der Abbruch-Event ausgelöst wurde
            logging.warning("Abbruch durch Benutzer!")
            return None
        if result == 1:
            print(f"Passwort gefunden: {wordlist[idx]}")
            return wordlist[idx]  # Erfolgreiches Passwort gefunden
        if progress_callback:
            progress_callback(idx / len(results_array))  # Fortschritt aktualisieren
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

# Benchmarking
def benchmark(zip_filename, wordlist, stop_event):
    start_time = time.time()
    test_passwords_cpu(zip_filename, wordlist[:100], stop_event, None)
    cpu_time = time.time() - start_time

    start_time = time.time()
    test_passwords_gpu(zip_filename, wordlist[:100], stop_event, None)
    gpu_time = time.time() - start_time

    logging.info(f"Benchmark-Ergebnisse: CPU {cpu_time:.2f}s, GPU {gpu_time:.2f}s")
    return cpu_time, gpu_time

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

# Abbruch-Thread
def check_abort(stop_event):
    while True:
        user_input = input("Drücke 'c' zum Abbrechen...\n")
        if user_input.strip().lower() == 'c':
            stop_event.set()
            logging.warning("Abbruch durch Benutzer!")

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

        # Verwende ttk.Progressbar für die Fortschrittsanzeige
        self.progress_bar = ttk.Progressbar(self.root, length=300, mode="determinate")
        self.progress_bar.pack(pady=10)

        self.stop_button = tk.Button(self.root, text="Abbrechen", command=self.abort)
        self.stop_button.pack(pady=10)

    def start_benchmark(self):
        wordlist_filename, zip_filename = select_files()
        if not wordlist_filename or not zip_filename:
            return

        wordlist = load_wordlist(wordlist_filename)

        cpu_time, gpu_time = benchmark(zip_filename, wordlist, self.stop_event)

        if cpu_time < gpu_time:
            logging.info("CPU-Modus wird verwendet.")
            test_passwords_cpu(zip_filename, wordlist, self.stop_event, self.update_progress)
        else:
            logging.info("GPU-Modus wird verwendet.")
            test_passwords_gpu(zip_filename, wordlist, self.stop_event, self.update_progress)

    def update_progress(self, progress):
        self.progress_bar['value'] = progress * 100
        self.root.update_idletasks()

    def abort(self):
        self.stop_event.set()

# Hauptprogramm
def main():
    root = tk.Tk()
    app = ZipCrackGUI(root)

    abort_thread = threading.Thread(target=check_abort, args=(app.stop_event,))
    abort_thread.daemon = True
    abort_thread.start()

    root.mainloop()

if __name__ == "__main__":
    main()