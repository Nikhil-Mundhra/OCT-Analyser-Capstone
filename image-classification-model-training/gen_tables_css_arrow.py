def gen_table(top, bot, is_l3=False):
    html = []
    html.append('<div style="width: 100%; overflow-x: auto; margin: 20px 0;">')
    html.append('  <table style="margin: 0 auto; border-collapse: separate; border-spacing: 0; background: transparent; border: none; font-family: sans-serif;">')
    
    # ROW 1
    html.append('    <tr>')
    for i, name in enumerate(top):
        html.append(f'      <td style="width: 120px;"><div style="padding: 8px 0; border: 2px solid currentColor; border-radius: 8px; background: transparent; color: currentColor; font-weight: bold; text-align: center;">{name}</div></td>')
        if i < len(top) - 1:
            html.append(f'      <td style="padding: 0 10px; color: currentColor; font-size: 18px; text-align: center;">▶</td>')
    html.append('    </tr>')
    
    # BRIDGE
    cols = len(top) * 2 - 1
    html.append('    <tr>')
    html.append(f'      <td colspan="{cols - 1}"></td>')
    html.append('      <td style="text-align: center; padding: 8px 0;">')
    html.append('        <div style="width: 6px; height: 16px; background-color: currentColor; margin: 0 auto;"></div>')
    html.append('        <div style="width: 0; height: 0; border-left: 8px solid transparent; border-right: 8px solid transparent; border-top: 10px solid currentColor; margin: 0 auto;"></div>')
    html.append('      </td>')
    html.append('    </tr>')
    
    # ROW 2
    html.append('    <tr>')
    if is_l3:
        # We need to skip the first 2 columns (1 box, 1 gap)
        html.append('      <td colspan="2"></td>')
    
    for i, name in enumerate(bot):
        html.append(f'      <td style="width: 120px;"><div style="padding: 8px 0; border: 2px solid currentColor; border-radius: 8px; background: transparent; color: currentColor; font-weight: bold; text-align: center;">{name}</div></td>')
        if i < len(bot) - 1:
            html.append(f'      <td style="padding: 0 10px; color: currentColor; font-size: 18px; text-align: center;">◀</td>')
    html.append('    </tr>')
    
    html.append('  </table>')
    html.append('</div>')
    return '\n'.join(html)

t1 = ["Input", "Conv1", "MaxPool", "ResBlock 1", "ResBlock 2"]
b1 = ["Output", "FC Head", "GAP", "ResBlock 4", "ResBlock 3"]

t2 = ["Input", "Stem", "MBConv 1", "MBConv 2"]
b2 = ["Output", "FC Head", "Head", "MBConv 3"]

t3 = ["High-Res Input", "Stem", "MBConv 1", "MBConv 2"]
b3 = ["Output", "FC Head", "Head"]

with open('tables_css_arrow.txt', 'w') as f:
    f.write("=== L1 ===\n")
    f.write(gen_table(t1, b1, False) + "\n\n")
    f.write("=== L2 ===\n")
    f.write(gen_table(t2, b2, False) + "\n\n")
    f.write("=== L3 ===\n")
    f.write(gen_table(t3, b3, True) + "\n\n")

