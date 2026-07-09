def generate_flexbox(top, bot, is_l3=False):
    html = []
    
    container_style = "display: flex; flex-direction: column; align-items: center; font-family: sans-serif; margin: 20px 0;"
    row_style = "display: flex; flex-direction: row; align-items: center; justify-content: center; width: 100%; gap: 10px;"
    
    # We will use simple boxes and text elements for arrows
    box_style = "display: flex; align-items: center; justify-content: center; width: 130px; height: 50px; background: rgba(128,128,128,0.1); border-radius: 8px; font-weight: bold; text-align: center;"
    arrow_style = "font-size: 24px; font-weight: bold; width: 30px; text-align: center; display: flex; align-items: center; justify-content: center;"
    
    html.append(f'<div style="{container_style}">')
    
    # ROW 1
    html.append(f'  <div style="{row_style}">')
    for i, name in enumerate(top):
        html.append(f'    <div style="{box_style}">{name}</div>')
        if i < len(top) - 1:
            html.append(f'    <div style="{arrow_style}">→</div>')
    html.append('  </div>')
    
    # ARROW DOWN
    # The down arrow should be positioned under the last element of the top row for L1, L2.
    # For L3, the down arrow is under the last element too.
    # We can use a flex row that mimics the spacing, with visibility: hidden elements to pad it out.
    html.append(f'  <div style="{row_style}">')
    for i in range(len(top) - 1):
        html.append(f'    <div style="{box_style}; visibility: hidden;"></div>')
        html.append(f'    <div style="{arrow_style}; visibility: hidden;"></div>')
    html.append(f'    <div style="width: 130px; {arrow_style}">↓</div>')
    html.append('  </div>')
    
    # ROW 2
    html.append(f'  <div style="{row_style}">')
    
    if is_l3:
        # Pad the first items to shift them right
        # L3 top has 4 items. L3 bot has 3 items.
        # We need the last item of bot to align with the last item of top.
        # So pad 1 item.
        html.append(f'    <div style="{box_style}; visibility: hidden;"></div>')
        html.append(f'    <div style="{arrow_style}; visibility: hidden;"></div>')
    
    # Reversing the arrow logic: the text is left-to-right, but arrows are pointing left
    for i, name in enumerate(bot):
        html.append(f'    <div style="{box_style}">{name}</div>')
        if i < len(bot) - 1:
            html.append(f'    <div style="{arrow_style}">←</div>')
    
    html.append('  </div>')
    html.append('</div>')
    
    return '\n'.join(html)

t1 = ["Input", "Conv1", "MaxPool", "ResBlock 1", "ResBlock 2"]
b1 = ["Output", "FC Head", "GAP", "ResBlock 4", "ResBlock 3"]

t2 = ["Input", "Stem", "MBConv 1", "MBConv 2"]
b2 = ["Output", "FC Head", "Head", "MBConv 3"]

t3 = ["High-Res Input", "Stem", "MBConv 1", "MBConv 2"]
b3 = ["Output", "FC Head", "Head"]

with open('flex_test.txt', 'w') as f:
    f.write(generate_flexbox(t1, b1, False) + "\n\n")
    f.write(generate_flexbox(t2, b2, False) + "\n\n")
    f.write(generate_flexbox(t3, b3, True) + "\n\n")
