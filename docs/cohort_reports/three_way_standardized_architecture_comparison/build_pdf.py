from pathlib import Path
import re, html
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter

from clinical_formatter import clean_clinical_text, format_reportlab_cell

root = Path(__file__).resolve().parent
md_path = root / 'research_report_tri_model_standardized_comparison.md'
md = clean_clinical_text(md_path.read_text(encoding='utf-8'))
out = root / 'research_report_tri_model_standardized_comparison.pdf'

blue = colors.HexColor('#0969da')
green = colors.HexColor('#166534')
ink = colors.HexColor('#1e293b')

styles = {
    'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=14, leading=17, textColor=blue, spaceAfter=8),
    'h2': ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=10.0, leading=12.5, textColor=green, spaceBefore=9, spaceAfter=3),
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=8.2, leading=11.0, textColor=ink, spaceAfter=4),
    'small': ParagraphStyle('small', fontName='Helvetica', fontSize=7.2, leading=9.0, textColor=ink, spaceAfter=3),
    'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=6.8, leading=8.6, textColor=ink),
    'cell_right': ParagraphStyle('cell_right', fontName='Helvetica', fontSize=6.8, leading=8.6, textColor=ink, alignment=TA_RIGHT),
    'head': ParagraphStyle('head', fontName='Helvetica-Bold', fontSize=6.8, leading=8.6, textColor=ink),
    'head_right': ParagraphStyle('head_right', fontName='Helvetica-Bold', fontSize=6.8, leading=8.6, textColor=ink, alignment=TA_RIGHT),
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
        if line.startswith('## 3.') or line.startswith('## 4.'):
            story.append(PageBreak())
        story.append(Paragraph(rich(line[3:]), styles['h2']))
        i += 1
        continue
    if line.startswith('### '):
        story.append(Paragraph(rich(line[4:]), styles['h2']))
        i += 1
        continue
    if line.startswith('!['):
        m = re.search(r'\(([^)]+)\)', line)
        if m:
            p = root / m.group(1)
            from PIL import Image as PILImage
            w, h = PILImage.open(p).size
            img_width = 490 if 'full_length' in p.name else 535
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
        first_w = 160 if n_cols >= 5 else 180
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
            ('LEFTPADDING', (0,0), (-1,-1), 4),
            ('RIGHTPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 2.5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2.5)
        ]))
        story.extend([t, Spacer(1, 4)])
        continue
    if line.startswith('- ') or line.startswith('* '):
        story.append(Paragraph('&#8226; ' + rich(line[2:]), styles['small']))
        i += 1
        continue
    if re.match(r'^\d+\.\s+', line):
        m = re.match(r'^(\d+)\.\s+(.*)$', line)
        story.append(Paragraph(f'{m.group(1)}. ' + rich(m.group(2)), styles['small']))
        i += 1
        continue
    para = []
    while i < len(lines) and lines[i].strip() and not (
        lines[i].startswith('#') or lines[i].startswith('|') or lines[i].startswith('![') or lines[i].startswith('- ') or lines[i].startswith('* ') or re.match(r'^\d+\.\s+', lines[i])
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
    canvas.drawString(38, 23, 'Standardized Three-Architecture RNFL Benchmark | 13 October 2026')
    canvas.drawRightString(574, 23, str(doc.page))

SimpleDocTemplate(
    str(out),
    pagesize=letter,
    rightMargin=38,
    leftMargin=38,
    topMargin=34,
    bottomMargin=44,
    title='Standardized Three-Architecture RNFL Benchmark'
).build(story, onFirstPage=page, onLaterPages=page)

print(f"Successfully generated PDF: {out}")
