# 💊 MediScope

MediScope is an AI-powered medicine label reader built with Streamlit. It uses Optical Character Recognition (OCR) to extract text from medicine packaging images, detects the language and translates non-English text to English, and validates the extracted medicine name against a built-in neural-network model.

> **Disclaimer:** MediScope is a research/educational tool and is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional before making any medical decisions.

---

## Features

- 📸 **Image Upload** – Upload JPG, JPEG, or PNG images of medicine packaging.
- 🔧 **User-Selected Preprocessing** – Choose from four OpenCV preprocessing methods (`simple`, `adaptive`, `otsu`, or `none`) to optimize OCR accuracy for different image conditions.
- 🔍 **Multi-Config OCR** – Runs Tesseract with multiple PSM configurations and aggregates results.
- 🌐 **Translation** – Detects the language of extracted text and translates non-English content to English using `langdetect` and `googletrans`.
- 🤖 **AI Validation** – A lightweight TensorFlow/Keras character-level CNN validates whether the extracted text resembles a known medicine name.
- 🔊 **Text-to-Speech** – Reads the extracted text aloud via `gTTS`.
- 💊 **Medicine Info Lookup** – Matches extracted text against a built-in medicine dictionary (Paracetamol, Dolo-650, Amoxicillin) and displays usage, dosage, and warnings.

> **Note:** `ocr_selector_model.pkl` is included in the repository for future use (automatic preprocessing selection) but is **not yet wired into `app.py`**. Preprocessing selection is currently manual.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI / App framework | [Streamlit](https://streamlit.io/) |
| Image processing | [OpenCV](https://opencv.org/), [Pillow](https://pillow.readthedocs.io/) |
| OCR engine | [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) via [pytesseract](https://pypi.org/project/pytesseract/) |
| Language detection | [langdetect](https://pypi.org/project/langdetect/) |
| Translation | [googletrans 4.0.0-rc1](https://py-googletrans.readthedocs.io/) |
| AI validation | [TensorFlow / Keras](https://www.tensorflow.org/) |
| Text-to-Speech | [gTTS](https://gtts.readthedocs.io/) |
| Numerical computing | [NumPy](https://numpy.org/) |

---

## Project Structure

```
MediScope/
├── app.py                    # Main Streamlit application
├── ai_validation.py          # TensorFlow character-level medicine name validator
├── translation.py            # Language detection & translation helper
├── ocr_selector_model.pkl    # Saved model for preprocessing selection (future use)
├── requirements.txt          # Python dependencies
├── packages.txt              # System-level dependencies (for Streamlit Cloud)
├── MediScopeOCR.ipynb        # OCR experimentation notebook
├── MediScopeOCR (2).ipynb    # OCR experimentation notebook (iteration 2)
├── MediScopeNER.ipynb        # Named Entity Recognition notebook
├── MediScopeKB.ipynb         # Knowledge Base exploration notebook
├── LICENSE
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.8 or higher
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system

#### System dependencies (required for OCR and image processing)

On Debian/Ubuntu:

```bash
sudo apt-get update && sudo apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0
```

> These are the same packages listed in `packages.txt` (used automatically by Streamlit Community Cloud).

### 1. Clone the repository

```bash
git clone https://github.com/Harsh7645/MediScope.git
cd MediScope
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

```
streamlit
opencv-python
pytesseract
numpy
Pillow
gtts
tensorflow
langdetect
googletrans==4.0.0-rc1
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Usage

1. Open the app in your browser.
2. Upload a clear image of a medicine package (JPG, JPEG, or PNG).
3. Select a **preprocessing method** from the dropdown:
   - **simple** – Gaussian blur only.
   - **adaptive** – Adaptive Gaussian thresholding (recommended for most labels).
   - **otsu** – Otsu's global thresholding.
   - **none** – Grayscale only, no thresholding.
4. Toggle **Show processed image** to inspect the result of preprocessing.
5. The app extracts text using multiple Tesseract configurations, translates it if needed, and validates it with the AI model.
6. If a known medicine is detected, dosage and warning information is shown.
7. Optionally click **🔊 Read Aloud** to hear the extracted text.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `MediScopeOCR.ipynb` | Initial OCR pipeline experiments |
| `MediScopeOCR (2).ipynb` | Iterative improvements to the OCR pipeline |
| `MediScopeNER.ipynb` | Named Entity Recognition experiments on medicine text |
| `MediScopeKB.ipynb` | Knowledge Base construction and exploration |

---

## Future Work

- Wire `ocr_selector_model.pkl` into `app.py` for automatic preprocessing method selection based on image characteristics.
- Expand the built-in medicine database.
- Improve NER integration from `MediScopeNER.ipynb`.
- Add batch image processing support.

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in the repository.

---

> Made with ❤️ using Streamlit