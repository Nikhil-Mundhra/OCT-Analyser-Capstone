"""
clinical_formatter.py
Permanent formatting and sanitization utilities for clinical OCT reports,
markdown tables, and PDF generation pipelines.
"""

import re
import html

def clean_clinical_text(text: str) -> str:
    """
    Sanitizes LaTeX fragments, malformed math delimiters, and legacy entity leaks
    into clean, publication-grade UTF-8 text for Markdown, HTML, and ReportLab.
    """
    if not text:
        return ""

    # 1. Clean existing double-escaped or raw entity leaks
    text = text.replace('&amp;#177;', '±').replace('&#177;', '±').replace('&plusmn;', '±')
    text = text.replace('&amp;#181;', 'µ').replace('&#181;', 'µ').replace('&mu;', 'µ')
    text = text.replace('&amp;nbsp;', ' ')

    # 2. Convert LaTeX and math-wrapped units to standard Unicode
    text = re.sub(r'\\?\$?\\mu(?:\s*\\text\{\{?m\}\}?|\s*m)?\\?\$?', 'µm', text)
    text = re.sub(r'\$\s*\\mu\s*\$\s*m', 'µm', text)
    text = re.sub(r'\\mu\\text\{m\}', 'µm', text)
    text = re.sub(r'\\mu m', 'µm', text)

    # 3. Plus-minus signs: $\pm$, \pm
    text = re.sub(r'\\?\$?\\pm\\?\$?', '±', text)

    # 4. Percentile notations: $P_{95}$, P_{95}, P_{{95}}, $P95$
    text = re.sub(r'\\?\$?P_?\{?95\}?\\?\$?', 'P95', text)

    # 5. Common mathematical operators
    text = re.sub(r'\\?\$?\\le\\?\$?', '≤', text)
    text = re.sub(r'\\?\$?\\ge\\?\$?', '≥', text)
    text = re.sub(r'\\?\$?\\times\\?\$?', '×', text)
    text = re.sub(r'\\?\$?\\to\\?\$?', '→', text)
    text = re.sub(r'\\?\$?\\approx\\?\$?', '≈', text)
    text = re.sub(r'\\?\$?\\Delta\\?\$?', 'Δ', text)

    # 6. Clean stray dollar signs and backslashes around units and symbols
    text = re.sub(r'\$\s*µm', 'µm', text)
    text = re.sub(r'µm\s*\$', 'µm', text)
    text = re.sub(r'\$\s*±', '±', text)
    text = re.sub(r'±\s*\$', '±', text)
    text = re.sub(r'\$\s*P95', 'P95', text)
    text = re.sub(r'P95\s*\$', 'P95', text)

    # Solitary numbers wrapped in single dollars: $0.9441$ -> 0.9441
    text = re.sub(r'(?<!\w)\$\s*([0-9\.]+)\s*\$(?!\w)', r'\1', text)

    # 7. Normalize whitespace around operators and units
    text = re.sub(r'\s*±\s*', ' ± ', text)
    text = re.sub(r'([0-9])\s*µm', r'\1 µm', text)

    return text


def format_reportlab_cell(text: str, is_header: bool = False, nowrap: bool = True) -> str:
    """
    Prepares a table cell string for ReportLab Paragraph rendering:
    - Cleans LaTeX and math.
    - Escapes XML characters.
    - Restores bold (<b>), code (<font>), and subscript (<sub>).
    - Protects numbers and units with non-breaking spaces so they never line-wrap.
    """
    text = clean_clinical_text(text)

    # Escape XML
    s = html.escape(text)

    # Convert markdown bold: **bold** -> <b>bold</b>
    parts = s.split('**')
    s = ''.join(('<b>' + p + '</b>' if i % 2 else p) for i, p in enumerate(parts))

    # Convert markdown code: `code` -> <font name="Courier" size="6.5">code</font>
    parts = s.split('`')
    s = ''.join(('<font name="Courier" size="6.5">' + p + '</font>' if i % 2 else p) for i, p in enumerate(parts))

    # Subscripts for P95: P95 -> P<sub>95</sub>
    s = re.sub(r'\bP95\b', 'P<sub>95</sub>', s)

    # Prevent awkward line wraps in numeric data and headers
    if nowrap:
        s = re.sub(r'([0-9\.]+)\s*±\s*([0-9\.]+)', r'\1&nbsp;±&nbsp;\2', s)
        s = re.sub(r'([0-9\.]+)\s*µm', r'\1&nbsp;µm', s)
        s = re.sub(r'([A-Za-z]+)\s*±\s*([A-Za-z]+)', r'\1&nbsp;±&nbsp;\2', s)

    return s
