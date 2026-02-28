# Anleitung zur Einrichtung und Ausführung (Setup Guide)

Um die alle Notebooks (z.B. `data_exploration.ipynb`) auf einem anderen PC auszuführen, folgen Sie bitte diesen Schritten:

## 1. Voraussetzungen
Stellen Sie sicher, dass **Python (Version 3.10 oder höher)** installiert ist. Sie können Python von [python.org](https://www.python.org/) herunterladen.

## 2. Umgebung einrichten (Empfohlen)
Öffnen Sie ein Terminal (CMD oder PowerShell) im Projektordner und erstellen Sie eine virtuelle Umgebung:

```powershell
# Virtuelle Umgebung erstellen
python -m venv venv

# Aktivieren (Windows)
.\venv\Scripts\activate
```

## 3. Bibliotheken installieren
Installieren Sie alle benötigten Pakete mit der bereitgestellten `requirements.txt`:

```powershell
pip install -r requirements.txt
```

## 4. Jupyter Notebook starten
Starten Sie die Jupyter-Oberfläche:

```powershell
jupyter notebook
```



