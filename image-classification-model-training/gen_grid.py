def gen_grid(top, bot, is_l3=False):
    html = []
    
    cols = len(top) * 2 - 1
    
    html.append(f'<div style="width: 100%; overflow-x: auto; margin: 20px 0;">')
    # Use explicit min-width or fit-content
    html.append(f'  <div style="display: grid; grid-template-columns: repeat({cols}, auto); row-gap: 5px; align-items: center; justify-items: center; width: max-content; margin: 0 auto; font-family: sans-serif;">')
    
    # ROW 1
    for i, name in enumerate(top):
        html.append(f'    <div style="width: 120px; padding: 8px 0; border: 2px solid currentColor; border-radius: 8px; background: transparent; color: currentColor; font-weight: bold; text-align: center;">{name}</div>')
        if i < len(top) - 1:
            html.append(f'    <div style="font-size: 24px; font-weight: bold; padding: 0 10px;">→</div>')
            
    # BRIDGE ROW
    html.append(f'    <div style="grid-column: {cols}; font-size: 24px; font-weight: bold; padding: 5px 0;">↓</div>')
    
    # ROW 2
    bot_cols = len(bot) * 2 - 1
    start_col = cols - bot_cols + 1
    
    for i, name in enumerate(bot):
        grid_col_str = f' grid-column: {start_col + i*2};' if (is_l3 and i==0) else ''
        html.append(f'    <div style="width: 120px; padding: 8px 0; border: 2px solid currentColor; border-radius: 8px; background: transparent; color: currentColor; font-weight: bold; text-align: center;{grid_col_str}">{name}</div>')
        if i < len(bot) - 1:
            html.append(f'    <div style="font-size: 24px; font-weight: bold; padding: 0 10px;">←</div>')
            
    html.append('  </div>')
    html.append('</div>')
    return '\n'.join(html)

t1 = ["Input", "Conv1", "MaxPool", "ResBlock 1", "ResBlock 2"]
b1 = ["Output", "FC Head", "GAP", "ResBlock 4", "ResBlock 3"]

t2 = ["Input", "Stem", "MBConv 1", "MBConv 2"]
b2 = ["Output", "FC Head", "Head", "MBConv 3"]

t3 = ["High-Res Input", "Stem", "MBConv 1", "MBConv 2"]
b3 = ["Output", "FC Head", "Head"]

with open('grids.txt', 'w') as f:
    f.write("=== L1 ===\n")
    f.write(gen_grid(t1, b1, False) + "\n\n")
    f.write("=== L2 ===\n")
    f.write(gen_grid(t2, b2, False) + "\n\n")
    f.write("=== L3 ===\n")
    f.write(gen_grid(t3, b3, True) + "\n\n")

