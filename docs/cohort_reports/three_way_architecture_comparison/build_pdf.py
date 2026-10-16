from pathlib import Path
import re, html
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter

from clinical_formatter import clean_clinical_text, format_reportlab_cell

root = Path(__file__).resolve().parent
md_path = root / 'research_report_tri_model_comparison.md'
md = clean_clinical_text(md_path.read_text(encoding='utf-8'))
out = root / 'research_report_tri_model_comparison.pdf'

blue = colors.HexColor('#0969da')
green = colors.HexColor('#166534')
ink = colors.HexColor('#1e293b')

styles = {
    'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=15, leading=18, textColor=blue, spaceAfter=8),
    'h2': ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=10.5, leading=13, textColor=green, spaceBefore=10, spaceAfter=4),
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=8.4, leading=11.3, textColor=ink, spaceAfter=5),
    'small': ParagraphStyle('small', fontName='Helvetica', fontSize=7.4, leading=9.3, textColor=ink, spaceAfter=4),
    'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=7.1, leading=9, textColor=ink),
    'cell_right': ParagraphStyle('cell_right', fontName='Helvetica', fontSize=7.1, leading=9, textColor=ink, alignment=TA_RIGHT),
    'head': ParagraphStyle('head', fontName='Helvetica-Bold', fontSize=7.1, leading=9, textColor=ink),
    'head_right': ParagraphStyle('head_right', fontName='Helvetica-Bold', fontSize=7.1, leading=9, textColor=ink, alignment=TA_RIGHT),
}

def rich(s):
    return format_reportlab_cell(s, is_header=False, nowrap=False)

story = []
lines = md.splitlines()
i = 0

while i < len(lines):
    line = lines[i].strip()
    if not line:
        i += 1
        continue
    if line.startswith('# '):
        story.append(Paragraph(rich(line[2:]), styles['title']))
        i += 1
        continue
    if line.startswith('## '):
        if line.startswith('## 2.') or line.startswith('## 3.'):
            story.append(PageBreak())
        story.append(Paragraph(rich(line[3:]), styles['h2']))
        i += 1
        continue
    if line.startswith('!['):
        m = re.search(r'\(([^)]+)\)', line)
        if m:
            p = root / m.group(1)
            from PIL import Image as PILImage
            w, h = PILImage.open(p).size
            img_width = 500 if 'full_length' in p.name else 535
            story.append(Image(str(p), width=img_width, height=img_width * h / w))
        i += 1
        continue
    if line.startswith('|'):
        rows = []
        while i < len(lines) and lines[i].strip().startswith('|'):
            cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?', c or '') for c in cells):
                rows.append(cells)
            i += 1
        
        n_cols = len(rows[0])
        if n_cols == 4:
            first_w = 175
            rem_w = (535 - first_w) / 3
            widths = [first_w] + [rem_w] * 3
        elif n_cols == 6:
            first_w = 125
            rem_w = (535 - first_w) / 5
            widths = [first_w] + [rem_w] * 5
        elif n_cols >= 7:
            first_w = 115
            rem_w = (535 - first_w) / (n_cols - 1)
            widths = [first_w] + [rem_w] * (n_cols - 1)
        else:
            first_w = 185
            rem_w = (535 - first_w) / (n_cols - 1)
            widths = [first_w] + [rem_w] * (n_cols - 1)
        
        data = []
        for ri, row in enumerate(rows):
            row_data = []
            for ci, c in enumerate(row):
                is_right = (ci > 0)
                if ri == 0:
                    st = styles['head_right'] if is_right else styles['head']
                    row_data.append(Paragraph(format_reportlab_cell(c, is_header=True, nowrap=True), st))
                else:
                    st = styles['cell_right'] if is_right else styles['cell']
                    row_data.append(Paragraph(format_reportlab_cell(c, is_header=False, nowrap=True), st))
            data.append(row_data)

        t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eef3f9')),
            ('GRID', (0,0), (-1,-1), 0.35, colors.HexColor('#d8e0e8')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 5),
            ('RIGHTPADDING', (0,0), (-1,-1), 5),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3)
        ]))
        story.extend([t, Spacer(1, 5)])
        continue
    if line.startswith('- '):
        story.append(Paragraph('&#8226; ' + rich(line[2:]), styles['small']))
        i += 1
        continue
    para = []
    while i < len(lines) and lines[i].strip() and not (
        lines[i].startswith('#') or lines[i].startswith('|') or lines[i].startswith('![') or lines[i].startswith('- ')
    ):
        para.append(lines[i].strip())
        i += 1
    if para:
        story.append(Paragraph(rich(' '.join(para)).replace('  ', '<br/>'), styles['body']))

def page(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#d8e0e8'))
    canvas.line(38, 34, 574, 34)
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(colors.HexColor('#64748b'))
    canvas.drawString(38, 23, 'Three-Architecture RNFL Benchmark | 10 October 2026')
    canvas.drawRightString(574, 23, str(doc.page))

SimpleDocTemplate(
    str(out),
    pagesize=letter,
    rightMargin=38,
    leftMargin=38,
    topMargin=34,
    bottomMargin=44,
    title='Three-Architecture RNFL Benchmark'
).build(story, onFirstPage=page, onLaterPages=page)

print(f"Successfully generated PDF: {out}")
