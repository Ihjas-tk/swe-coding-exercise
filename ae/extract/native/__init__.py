"""Native extraction: PyMuPDF (text/images) + pdfplumber (tables) + Tesseract (OCR).

It reads the PDF's own structure (text layer, placed images, vector graphics) and OCRs only
pages without a usable text layer. Each module is small and explainable; the orchestration
lives in pdf.py.
"""
