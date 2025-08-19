import os
import logging
import fitz  # PyMuPDF
import docx

logging.basicConfig(level=logging.INFO)


# ---- Base Extractor ----
class DocumentExtractor:
    def extract(self, path: str) -> str:
        raise NotImplementedError("Extractor must implement extract()")


# ---- PDF Extractor ----
class PDFExtractor(DocumentExtractor):
    def extract(self, path: str) -> str:
        text = []
        with fitz.open(path) as doc:
            for page in doc:
                text.append(page.get_text("text"))
        return "\n".join(text)


# ---- DOCX Extractor ----
class DocxExtractor(DocumentExtractor):
    def extract(self, path: str) -> str:
        doc = docx.Document(path)
        text = [para.text for para in doc.paragraphs]
        return "\n".join(text)


# ---- TXT Extractor ----
class TxtExtractor(DocumentExtractor):
    def extract(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


# ---- Dispatcher ----
class ExtractorDispatcher:
    _registry = {
        ".pdf": PDFExtractor(),
        ".docx": DocxExtractor(),
        ".txt": TxtExtractor(),
    }

    @classmethod
    def extract(cls, path: str) -> str:
        ext = os.path.splitext(path)[-1].lower()
        extractor = cls._registry.get(ext)
        if not extractor:
            raise ValueError(f"Unsupported file type: {ext}")
        logging.info(f"Extracting from {ext.upper()} file: {path}")
        return extractor.extract(path)


# ---- Normalizer ----
class Normalizer:
    @staticmethod
    def normalize(text: str) -> str:
        import re
        # collapse multiple whitespace into one space
        clean = re.sub(r"\s+", " ", text)
        return clean.strip()


# ---- Public API ----
def process_document(path: str, normalize: bool = True) -> str:
    raw_text = ExtractorDispatcher.extract(path)
    return Normalizer.normalize(raw_text) if normalize else raw_text

text = process_document(r'C:\Users\shenile.a\Downloads\Exploration-Summary-Visual-Testing-AI-Powered.pdf')
print(text)