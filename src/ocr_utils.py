"""
OCR & Document Extraction Utilities -- Week 3

Functions for extracting text from images and PDFs using
Tesseract (baseline), Marker (modern), and Docling (enterprise).

Functions:
  ocr_with_tesseract(image_path)    - Basic OCR with Tesseract
  ocr_with_surya(image_path)        - Layout-aware OCR with Surya
  extract_pdf_with_marker(pdf_path) - PDF to markdown with Marker
  extract_pdf_with_docling(pdf_path)- PDF extraction with Docling
  compare_ocr_methods(image_path)   - Side-by-side OCR comparison
"""

import os
import json
from typing import Dict, List, Optional, Any


def ocr_with_tesseract(
    image_path: str,
    lang: str = "eng",
    preprocess: bool = True,
) -> Dict[str, Any]:
    """
    Extract text from an image using Tesseract OCR.

    Args:
        image_path: Path to image file
        lang: Tesseract language code
        preprocess: Whether to convert to grayscale first

    Returns:
        Dict with 'text', 'method', 'char_count', 'image_path' keys
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise ImportError("Install: pip install pytesseract Pillow\nAlso: brew install tesseract (macOS) or apt install tesseract-ocr (Linux)")

    import time

    print("=" * 55)
    print("TESSERACT OCR")
    print("=" * 55)
    print(f"Image: {image_path}")

    start = time.time()

    image = Image.open(image_path)
    if preprocess:
        image = image.convert("L")  # grayscale
        print("  Preprocessed: grayscale conversion")

    text = pytesseract.image_to_string(image, lang=lang)
    elapsed = time.time() - start

    result = {
        "text": text.strip(),
        "method": "tesseract",
        "char_count": len(text.strip()),
        "image_path": image_path,
        "elapsed_seconds": round(elapsed, 2),
    }

    print(f"  Extracted {result['char_count']:,} chars in {elapsed:.1f}s")
    print(f"  Preview: {text[:200]}...")
    return result


def ocr_with_surya(
    image_path: str,
    langs: List[str] = None,
) -> Dict[str, Any]:
    """
    Layout-aware OCR using Surya (PyTorch-based).

    Args:
        image_path: Path to image file
        langs: Language codes (default: ["en"])

    Returns:
        Dict with 'text', 'lines', 'method', 'char_count' keys
    """
    try:
        from PIL import Image
        import surya
    except ImportError:
        raise ImportError("Install: pip install surya-ocr")

    import time

    if langs is None:
        langs = ["en"]

    print("=" * 55)
    print("SURYA OCR")
    print("=" * 55)
    print(f"Image: {image_path}")
    print(f"Languages: {langs}")

    start = time.time()
    image = Image.open(image_path)

    # Try newer API (>=0.8) first, then fall back to 0.6.x API
    try:
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
        det = DetectionPredictor()
        rec = RecognitionPredictor()
        predictions = rec([image], [langs], det)
    except (ImportError, TypeError, KeyError):
        try:
            from surya.model.detection.model import load_model as load_det_model
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor
            from surya.detection import batch_text_detection
            from surya.recognition import batch_recognition
            det_model = load_det_model()
            rec_model = load_rec_model()
            rec_processor = load_rec_processor()
            det_predictions = batch_text_detection([image], det_model)
            predictions = batch_recognition([image], [langs], rec_model, rec_processor, det_predictions)
        except Exception as e:
            raise RuntimeError(
                f"Surya OCR failed (v{surya.__version__ if hasattr(surya, '__version__') else '?'}). "
                f"This is likely a version conflict with transformers. "
                f"Use Tesseract instead, or try: pip install surya-ocr==0.6.1 transformers==4.40.0\n"
                f"Original error: {e}"
            )

    lines = []
    for page in predictions:
        for line in page.text_lines:
            lines.append({
                "text": line.text,
                "confidence": line.confidence,
                "polygon": line.polygon,
            })

    full_text = "\n".join(l["text"] for l in lines)
    elapsed = time.time() - start

    result = {
        "text": full_text,
        "lines": lines,
        "method": "surya",
        "char_count": len(full_text),
        "elapsed_seconds": round(elapsed, 2),
    }

    print(f"  Extracted {len(lines)} lines, {result['char_count']:,} chars in {elapsed:.1f}s")
    avg_conf = sum(l["confidence"] for l in lines) / len(lines) if lines else 0
    print(f"  Average confidence: {avg_conf:.3f}")
    return result


def ocr_with_easyocr(
    image_path: str,
    langs: List[str] = None,
) -> Dict[str, Any]:
    """
    OCR using EasyOCR (PyTorch-based, supports 80+ languages).

    EasyOCR is a good alternative to Surya that works with modern
    transformers versions. It uses CRAFT for detection and CRNN for recognition.

    Args:
        image_path: Path to image file
        langs: Language codes (default: ["en"])

    Returns:
        Dict with 'text', 'lines', 'method', 'char_count', 'elapsed_seconds' keys
    """
    try:
        import easyocr
    except ImportError:
        raise ImportError("Install: pip install easyocr")

    import time

    if langs is None:
        langs = ["en"]

    print("=" * 55)
    print("EASYOCR")
    print("=" * 55)
    print(f"Image: {image_path}")
    print(f"Languages: {langs}")

    start = time.time()

    reader = easyocr.Reader(langs, gpu=False, verbose=False)
    results = reader.readtext(image_path)

    lines = []
    for bbox, text, confidence in results:
        lines.append({
            "text": text,
            "confidence": round(confidence, 3),
            "bbox": bbox,
        })

    full_text = " ".join(l["text"] for l in lines)
    elapsed = time.time() - start

    result = {
        "text": full_text,
        "lines": lines,
        "method": "easyocr",
        "char_count": len(full_text),
        "elapsed_seconds": round(elapsed, 2),
    }

    print(f"  Extracted {len(lines)} text regions, {result['char_count']:,} chars in {elapsed:.1f}s")
    avg_conf = sum(l["confidence"] for l in lines) / len(lines) if lines else 0
    print(f"  Average confidence: {avg_conf:.3f}")
    return result


def extract_pdf_with_marker(
    pdf_path: str,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert PDF to markdown using Marker (by VikParuchuri, creator of Surya).

    Marker is the modern evolution of Surya, optimized for full PDF-to-markdown
    conversion with layout awareness, table extraction, and image handling.

    Args:
        pdf_path: Path to PDF file
        save_path: Path to save markdown output (None to skip)

    Returns:
        Dict with 'text', 'method', 'char_count', 'pages' keys
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except ImportError:
        raise ImportError(
            "Install: pip install marker-pdf\n"
            "Note: Marker requires torch. Install torch first if needed."
        )

    import time

    print("=" * 55)
    print("MARKER PDF EXTRACTION")
    print("=" * 55)
    print(f"PDF: {pdf_path}")

    start = time.time()

    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(pdf_path)
    markdown_text = rendered.markdown

    elapsed = time.time() - start

    result = {
        "text": markdown_text,
        "method": "marker",
        "char_count": len(markdown_text),
        "elapsed_seconds": round(elapsed, 2),
    }

    print(f"  Extracted {result['char_count']:,} chars in {elapsed:.1f}s")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)
        print(f"  Saved to: {save_path}")

    return result


def extract_pdf_with_docling(
    pdf_path: str,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract text from PDF using Docling (IBM).

    Docling handles PDF, DOCX, PPTX, XLSX, HTML, images and more,
    with first-class integrations for LlamaIndex and LangChain.

    Args:
        pdf_path: Path to PDF or document file
        save_path: Path to save markdown output (None to skip)

    Returns:
        Dict with 'text', 'method', 'char_count' keys
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise ImportError(
            "Install: pip install docling\n"
            "Note: Docling may require additional system deps."
        )

    import time

    print("=" * 55)
    print("DOCLING DOCUMENT EXTRACTION")
    print("=" * 55)
    print(f"Document: {pdf_path}")

    start = time.time()

    converter = DocumentConverter()
    result_doc = converter.convert(pdf_path)
    markdown_text = result_doc.document.export_to_markdown()

    elapsed = time.time() - start

    result = {
        "text": markdown_text,
        "method": "docling",
        "char_count": len(markdown_text),
        "elapsed_seconds": round(elapsed, 2),
    }

    print(f"  Extracted {result['char_count']:,} chars in {elapsed:.1f}s")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)
        print(f"  Saved to: {save_path}")

    return result


def compare_ocr_methods(
    image_path: str,
    methods: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compare multiple OCR methods on the same image.

    Args:
        image_path: Path to image file
        methods: List of methods to compare (default: ["tesseract", "surya"])

    Returns:
        Dict with results from each method
    """
    if methods is None:
        methods = ["tesseract", "easyocr"]

    print("=" * 55)
    print("OCR METHOD COMPARISON")
    print("=" * 55)
    print(f"Image: {image_path}\n")

    results = {}

    method_funcs = {
        "tesseract": ocr_with_tesseract,
        "easyocr": ocr_with_easyocr,
        "surya": ocr_with_surya,
    }

    for method in methods:
        func = method_funcs.get(method)
        if func is None:
            print(f"  Unknown method: {method}")
            continue
        try:
            results[method] = func(image_path)
        except (ImportError, Exception) as e:
            print(f"  {method} failed: {e}")
            results[method] = {"error": str(e)}

    # Print comparison
    print(f"\n--- Comparison ---")
    for method, res in results.items():
        if "error" in res:
            print(f"  {method}: ERROR - {res['error']}")
        else:
            print(f"  {method}: {res['char_count']:,} chars in {res['elapsed_seconds']:.1f}s")

    return results
