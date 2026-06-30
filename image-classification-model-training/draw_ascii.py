def draw_boxes(names, row):
    l1, l2, l3 = "", "", ""
    for i, name in enumerate(names):
        name = name.center(10)
        l1 += f"╭──────────╮"
        l2 += f"│{name}│"
        if row == 1 and i == len(names) - 1:
            l3 += f"╰─────┬────╯"
        elif row == 2 and i == len(names) - 1:
            l3 += f"╰──────────╯"
        else:
            l3 += f"╰──────────╯"
        
        if i < len(names) - 1:
            if row == 1:
                l1 += "     "
                l2 += " ──▶ "
                l3 += "     "
            else:
                l1 += "     "
                l2 += " ◀── "
                l3 += "     "
    return l1 + "\n" + l2 + "\n" + l3

names1 = ["Input", "Conv1", "MaxPool", "ResBlock1", "ResBlock2"]
names2 = ["Output", "FC Head", "GAP", "ResBlock4", "ResBlock3"]
names2_rev = names2[::-1]  # actually we want them printed from left to right

# We need Output on the far left. So names to pass: Output, FC Head, GAP, ResBlock4, ResBlock3
# Wait, if output is on far left, the arrows go left.

r1 = draw_boxes(names1, 1)
r2 = draw_boxes(names2, 2)

center_idx = len(names1) - 1
offset = center_idx * 17 + 6
bridge = (" " * offset) + "│\n" + (" " * offset) + "▼"

print(r1)
print(bridge)
print(r2)
print("\n" + "="*80 + "\n")

names3 = ["Input", "Stem", "MBConv 1", "MBConv 2"]
names4 = ["Output", "FC Head", "Head", "MBConv 3"]
print(draw_boxes(names3, 1))
offset3 = (len(names3) - 1) * 17 + 6
print((" " * offset3) + "│\n" + (" " * offset3) + "▼")
print(draw_boxes(names4, 2))

print("\n" + "="*80 + "\n")

names5 = ["High-Res", "Stem", "MBConv 1", "MBConv 2"]
names6 = ["Output", "FC Head", "Head"]
# For L3 we have 4 on top, 3 on bottom. We want them right-aligned!
pad = " " * 17
print(draw_boxes(names5, 1))
offset5 = (len(names5) - 1) * 17 + 6
print((" " * offset5) + "│\n" + (" " * offset5) + "▼")
# pad row 2
l1, l2, l3 = draw_boxes(names6, 2).split("\n")
print(pad + l1)
print(pad + l2)
print(pad + l3)

