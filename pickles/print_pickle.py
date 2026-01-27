import pickle
import os
import argparse
from pprint import pprint

# 🔹 Argumente parsen
parser = argparse.ArgumentParser(description="Pickle-Datei anzeigen")
parser.add_argument("file", type=str, help="Pfad zur .pkl Datei")
args = parser.parse_args()

file_path = args.file

# 🔹 Existenz prüfen
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

# 🔹 Datei laden
with open(file_path, "rb") as f:
    data = pickle.load(f)

# 🔹 Inhalt anzeigen
print(f"\nInhalt von {file_path}:")
if isinstance(data, dict):
    pprint(data)
elif isinstance(data, list):
    print(f"Liste mit {len(data)} Einträgen")
    pprint(data[:10])  # nur die ersten 10 Elemente
else:
    print(type(data))
    pprint(data)

