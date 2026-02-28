# Anleitung zur Einrichtung und Ausführung (Setup Guide)

Um die Analyse-Notebooks (z.B. `data_exploration.ipynb`) auf einem anderen PC auszuführen, folgen Sie bitte diesen Schritten:

## 1. Voraussetzungen
Stellen Sie sicher, dass **Python (Version 3.10 oder höher)** installiert ist. Sie können Python von [python.org](https://www.python.org/) herunterladen.

## 2. Projekt kopieren
Kopieren Sie den gesamten Projektordner auf den neuen PC. Die Struktur sollte wie folgt aussehen:
- `data/` (Hier müssen die Excel-Dateien liegen)
- `data_exploration.ipynb`
- `requirements.txt`
- ... (weitere Dateien)

## 3. Umgebung einrichten (Empfohlen)
Öffnen Sie ein Terminal (CMD oder PowerShell) im Projektordner und erstellen Sie eine virtuelle Umgebung:

```powershell
# Virtuelle Umgebung erstellen
python -m venv venv

# Aktivieren (Windows)
.\venv\Scripts\activate
```

## 4. Bibliotheken installieren
Installieren Sie alle benötigten Pakete mit der bereitgestellten `requirements.txt`:

```powershell
pip install -r requirements.txt
```

## 5. Jupyter Notebook starten
Starten Sie die Jupyter-Oberfläche:

```powershell
jupyter notebook
```

Wählen Sie im Browser die Datei `data_exploration.ipynb` aus.

---

## Wichtige Hinweise zu den Daten
Das Skript erwartet die Excel-Dateien im Unterordner `data/` mit folgenden Namen:
1. `M200200 - Heckrungenanlage.XLSX` (Produktionsdaten)
2. `Stoerliste_Heckrungenanlage_2023_NEU.xlsx` (Störungsdaten)

Falls die Pfade anders sind, müssen die Konstanten `PROD_FILE` und `STOER_FILE` in der ersten Code-Zelle des Notebooks angepasst werden.

## Zusätzliche Info für LSTM (Optional)
Falls Sie auch das Notebook `predictive_maintenance_lstm.ipynb` nutzen möchten, installieren Sie zusätzlich TensorFlow:
```powershell
pip install tensorflow
```
