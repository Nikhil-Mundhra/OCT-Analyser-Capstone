import os

def generate_svg(filename, top_nodes, bottom_nodes, align_bottom='left'):
    box_w = 140
    box_h = 50
    gap_x = 40
    gap_y = 60
    
    # Calculate widths
    w_top = len(top_nodes) * box_w + (len(top_nodes) - 1) * gap_x
    w_bot = len(bottom_nodes) * box_w + (len(bottom_nodes) - 1) * gap_x
    total_w = max(w_top, w_bot)
    total_h = 2 * box_h + gap_y
    
    # Offsets for centering or right-aligning
    offset_top = 0
    offset_bot = 0
    if align_bottom == 'right':
        offset_bot = total_w - w_bot
        
    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-20 -20 {total_w + 40} {total_h + 40}" width="100%">')
    
    # Defs for gradients and markers
    svg.append('''
    <defs>
        <marker id="arrowHead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#3B82F6" />
        </marker>
        <marker id="arrowHeadRev" markerWidth="10" markerHeight="7" refX="1" refY="3.5" orient="auto">
            <polygon points="10 0, 0 3.5, 10 7" fill="#3B82F6" />
        </marker>
        <linearGradient id="boxGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:#2a2b2e;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#1c1d1f;stop-opacity:1" />
        </linearGradient>
    </defs>
    <style>
        .box { fill: url(#boxGrad); stroke: #3B82F6; stroke-width: 1.5px; rx: 8px; }
        .text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 14px; font-weight: 600; fill: #F3F4F6; text-anchor: middle; dominant-baseline: middle; }
        .arrow { stroke: #3B82F6; stroke-width: 2px; fill: none; }
    </style>
    ''')
    
    # Draw top row
    for i, name in enumerate(top_nodes):
        x = offset_top + i * (box_w + gap_x)
        y = 0
        svg.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" class="box" />')
        svg.append(f'<text x="{x + box_w/2}" y="{y + box_h/2 + 1}" class="text">{name}</text>')
        # Arrow to next
        if i < len(top_nodes) - 1:
            svg.append(f'<line x1="{x + box_w}" y1="{y + box_h/2}" x2="{x + box_w + gap_x - 4}" y2="{y + box_h/2}" class="arrow" marker-end="url(#arrowHead)" />')
            
    # Draw bottom row (reversed internally)
    # The nodes are given left-to-right (e.g. Output, FC Head, GAP, ...) 
    # But arrows go Right to Left!
    for i, name in enumerate(bottom_nodes):
        x = offset_bot + i * (box_w + gap_x)
        y = box_h + gap_y
        svg.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" class="box" />')
        svg.append(f'<text x="{x + box_w/2}" y="{y + box_h/2 + 1}" class="text">{name}</text>')
        # Arrow goes from right box to left box!
        # So if i > 0, we draw an arrow from current box leftward to previous box.
        if i > 0:
            svg.append(f'<line x1="{x}" y1="{y + box_h/2}" x2="{x - gap_x + 4}" y2="{y + box_h/2}" class="arrow" marker-end="url(#arrowHeadRev)" />')

    # Draw vertical connecting line
    # From last top node to last bottom node (which is rightmost in visual, so index = len-1)
    x_top_last = offset_top + (len(top_nodes) - 1) * (box_w + gap_x) + box_w/2
    x_bot_last = offset_bot + (len(bottom_nodes) - 1) * (box_w + gap_x) + box_w/2
    y_start = box_h
    y_end = box_h + gap_y - 4
    
    # Path that goes down, then connects horizontally if needed, then goes down.
    # Actually, they are aligned perfectly above each other, so it's a straight line!
    svg.append(f'<line x1="{x_top_last}" y1="{y_start}" x2="{x_bot_last}" y2="{y_end}" class="arrow" marker-end="url(#arrowHead)" />')
    
    svg.append('</svg>')
    
    with open(filename, 'w') as f:
        f.write('\n'.join(svg))

# Paths
base_path = '/Users/nikhilmundhra/.gemini/antigravity/brain/dd0e9248-3248-48a2-9636-76e6ef9a9643/'
os.makedirs(base_path, exist_ok=True)

# L1
t1 = ["Input", "Conv1", "MaxPool", "ResBlock 1", "ResBlock 2"]
b1 = ["Output", "FC Head", "GAP", "ResBlock 4", "ResBlock 3"]
generate_svg(base_path + 'l1_arch.svg', t1, b1, align_bottom='left')

# L2
t2 = ["Input", "Stem", "MBConv 1", "MBConv 2"]
b2 = ["Output", "FC Head", "Head", "MBConv 3"]
generate_svg(base_path + 'l2_arch.svg', t2, b2, align_bottom='left')

# L3
t3 = ["High-Res Input", "Stem", "MBConv 1", "MBConv 2"]
b3 = ["Output", "FC Head", "Head"]
generate_svg(base_path + 'l3_arch.svg', t3, b3, align_bottom='right')

