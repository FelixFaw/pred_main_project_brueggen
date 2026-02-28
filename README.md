# README automatisch generiert! Muss noch angepasst werden!

# pm-lstm-dualplant — Predictive Maintenance mit LSTM (2 Anlagen)

Dieses Projekt implementiert ein **LSTM-basiertes Predictive-Maintenance-Modell** für **zwei Anlagen**. Ziel ist es, aus Zeitreihen-Sensordaten **Ausfälle / Wartungsbedarf** vorherzusagen (z. B. Klassifikation “Failure in X Stunden” oder Regression “Remaining Useful Life”).

## Inhalt
- [Projektidee](#projektidee)
- [Repo-Struktur](#repo-struktur)
- [Voraussetzungen](#voraussetzungen)
- [Setup](#setup)
- [Daten](#daten)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference / Vorhersage](#inference--vorhersage)
- [Konfiguration](#konfiguration)
- [Team-Workflow (Git)](#team-workflow-git)
- [Troubleshooting](#troubleshooting)

---

## Projektidee
Wir bauen ein LSTM-Modell, das Sensordaten zweier Anlagen verarbeitet und daraus ein Predictive-Maintenance-Target ableitet, z. B.:

- **Binary Classification:** `failure_within_horizon` (0/1)
- **Multi-Class:** `failure_class` (z. B. Fehlerarten)
- **Regression:** `RUL` (Remaining Useful Life)

> Hinweis: Welche Zielvariable(n) genutzt werden, wird in `config/` festgelegt (siehe [Konfiguration](#konfiguration)).

---

## Repo-Struktur
Empfohlene Struktur (kann angepasst sein):

pm-lstm-dualplant/
src/
data/ # Laden, Validierung, Preprocessing
features/ # Feature Engineering (Windowing, Scaling, etc.)
models/ # LSTM Modell, Training, Inference
evaluation/ # Metriken, Plots, Reports
utils/ # Helpers (Logging, Seeds, etc.)
scripts/
train.py
evaluate.py
predict.py
config/
config.yaml # zentrale Konfiguration
data/
raw/ # Rohdaten (optional im Repo; siehe Daten-Abschnitt)
processed/ # erzeugte Daten (nicht committen)
models/ # gespeicherte Modelle (nicht committen oder per Git LFS/DVC)
reports/ # Ergebnisse, Grafiken, Tabellen
tests/
.env.example
.gitignore
pyproject.toml # oder requirements.txt
README.md

---

## Voraussetzungen
- **Python**: 3.10+ (empfohlen)
- **IntelliJ IDEA** mit Python Plugin (oder PyCharm)
- Optional:
    - **Git LFS** (wenn große Datendateien versioniert werden sollen)
    - **DVC** (wenn Daten extern versioniert werden sollen)

---

## Setup

### 1) Repo klonen
```bash
git clone <REPO-URL>
cd pm-lstm-dualplant

### 2) Virtuelle Umgebung erstellen & aktivieren

Windows (PowerShell)

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -U pip
pip install -r requirements.txt

...