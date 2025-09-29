# MRI-PromptAI

MRI-PromptAI è un progetto di intelligenza artificiale per l’analisi di immagini mediche 3D in formato **NIfTI**, con un focus sulla diagnosi e lo studio dell’Alzheimer.
Il sistema integra **modelli di deep learning** con tecniche di **prompt engineering** per generare interpretazioni e valutazioni più chiare dei risultati.

---

## 📂 Struttura del progetto

* `src/` → codice Python principale
* `notebooks/` → esperimenti e prototipi in Jupyter Notebook
* `data/` → esempi di dati (non include dataset completi)
* `results/` → output, immagini e visualizzazioni

---

## 🚀 Funzionalità principali

* Preprocessing avanzato di immagini MRI 3D (bias correction, denoising, brain extraction).
* Modelli di **classification** per rilevamento di Alzheimer da scansioni.
* Utilizzo di **Grad-CAM** per la visualizzazione delle aree rilevanti.
* Prompt engineering per arricchire l’interpretazione dei risultati del modello.

---

## 🛠️ Tecnologie utilizzate

* Python 3.x
* PyTorch / TensorFlow
* NiBabel (gestione file NIfTI)
* Scikit-learn
* Matplotlib, Seaborn (visualizzazione)

---

## 📌 Note

Il progetto è stato sviluppato durante uno **stage ITS** come attività di ricerca e sperimentazione.
Il dataset principale utilizzato è **ADNI (Alzheimer’s Disease Neuroimaging Initiative)**.

---

## 👤 Autore

Sviluppato da **Nicolò Petruzzella** durante lo stage presso Lutec (Progetto FAIR).
