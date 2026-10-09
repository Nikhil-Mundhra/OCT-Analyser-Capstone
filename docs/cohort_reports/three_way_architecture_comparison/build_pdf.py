from pathlib import Path
import re, html
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter

root=Path(__file__).resolve().parent
md=(root/'research_report_tri_model_comparison.md').read_text()
out=root/'research_report_tri_model_comparison.pdf'
blue=colors.HexColor('#0969da'); green=colors.HexColor('#166534'); ink=colors.HexColor('#1e293b')
styles={
 'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=15,leading=18,textColor=blue,spaceAfter=8),
 'h2':ParagraphStyle('h2',fontName='Helvetica-Bold',fontSize=10.5,leading=13,textColor=green,spaceBefore=10,spaceAfter=4),
 'body':ParagraphStyle('body',fontName='Helvetica',fontSize=8.4,leading=11.3,textColor=ink,spaceAfter=5),
 'small':ParagraphStyle('small',fontName='Helvetica',fontSize=7.4,leading=9.3,textColor=ink,spaceAfter=4),
 'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=7.1,leading=9,textColor=ink),
 'head':ParagraphStyle('head',fontName='Helvetica-Bold',fontSize=7.1,leading=9,textColor=ink),
}
def rich(s):
 s=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'\1',s)
 s=html.escape(s)
 s=s.replace('**','<b>',1) if False else s
 parts=s.split('**'); s=''.join(('<b>'+p+'</b>' if i%2 else p) for i,p in enumerate(parts))
 parts=s.split('`'); s=''.join(('<font name="Courier" size="7">'+p+'</font>' if i%2 else p) for i,p in enumerate(parts))
 s=s.replace('µ','&#181;').replace('±','&#177;').replace('–','-').replace('—','-')
 return s
story=[]; lines=md.splitlines(); i=0
while i<len(lines):
 line=lines[i].strip()
 if not line: i+=1; continue
 if line.startswith('# '): story.append(Paragraph(rich(line[2:]),styles['title']));i+=1;continue
 if line.startswith('## '):
  if line.startswith('## 2.') or line.startswith('## 3.'): story.append(PageBreak())
  story.append(Paragraph(rich(line[3:]),styles['h2']));i+=1;continue
 if line.startswith('!['):
  m=re.search(r'\(([^)]+)\)',line)
  if m:
   p=root/m.group(1); from PIL import Image as PILImage; w,h=PILImage.open(p).size; img_width=500 if 'full_length' in p.name else 535; story.append(Image(str(p),width=img_width,height=img_width*h/w))
  i+=1;continue
 if line.startswith('|'):
  rows=[]
  while i<len(lines) and lines[i].strip().startswith('|'):
   cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
   if not all(re.fullmatch(r':?-+:?',c or '') for c in cells): rows.append(cells)
   i+=1
  widths=[185]+[(535-185)/(len(rows[0])-1)]*(len(rows[0])-1)
  data=[[Paragraph(rich(c),styles['head'] if ri==0 else styles['cell']) for c in row] for ri,row in enumerate(rows)]
  t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
  t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eef3f9')),('GRID',(0,0),(-1,-1),0.35,colors.HexColor('#d8e0e8')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
  story.extend([t,Spacer(1,5)]);continue
 if line.startswith('- '):
  story.append(Paragraph('&#8226; '+rich(line[2:]),styles['small']));i+=1;continue
 para=[]
 while i<len(lines) and lines[i].strip() and not (lines[i].startswith('#') or lines[i].startswith('|') or lines[i].startswith('![') or lines[i].startswith('- ')):
  para.append(lines[i].strip());i+=1
 if para: story.append(Paragraph(rich(' '.join(para).strip('*')).replace('  ','<br/>'),styles['body']))

def page(canvas,doc):
 canvas.setStrokeColor(colors.HexColor('#d8e0e8'));canvas.line(38,34,574,34)
 canvas.setFont('Helvetica',7);canvas.setFillColor(colors.HexColor('#64748b'))
 canvas.drawString(38,23,'Three-Architecture RNFL Benchmark | 8 October 2026')
 canvas.drawRightString(574,23,str(doc.page))
SimpleDocTemplate(str(out),pagesize=letter,rightMargin=38,leftMargin=38,topMargin=34,bottomMargin=44,title='Three-Architecture RNFL Benchmark').build(story,onFirstPage=page,onLaterPages=page)
print(out)
