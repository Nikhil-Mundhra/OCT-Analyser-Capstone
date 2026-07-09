def generate_ascii(top, bot, is_l3=False):
    # Determine the spacing to align everything
    # We'll just separate everything by " -> " and " <- "
    
    top_line = " \u2192 ".join(top)
    
    # The down arrow should align with the center of the last element
    last_word = top[-1]
    last_word_start = top_line.rfind(last_word)
    down_arrow_pos = last_word_start + len(last_word) // 2
    
    down_line = " " * down_arrow_pos + "\u2193"
    
    bot_line_items = bot.copy()
    
    if is_l3:
        # Align Output (first bot item) with MBConv 1 (second to last top item)
        # Because L3 bottom is smaller than L3 top
        target_align = top[-2]
        target_start = top_line.rfind(target_align)
        
        # Build bot line from right to left
        bot_str = " \u2190 ".join(bot)
        
        # We need the first item of bot ("Output") to align with target
        bot_line = " " * target_start + bot_str
        
    else:
        # Align perfectly under
        # We need the last item of bot_line to align with last item of top_line
        bot_str = " \u2190 ".join(bot)
        
        # Find position of last bot item in the bot_str
        last_bot_word = bot[-1]
        last_bot_start = bot_str.rfind(last_bot_word)
        
        # The center of last_bot_word should align with the down arrow
        # Or just left align last_bot_word with last_word
        
        pad = last_word_start - last_bot_start
        bot_line = " " * pad + bot_str if pad > 0 else bot_str
        
    return f"```text\n{top_line}\n{down_line}\n{bot_line}\n```"

t1 = ["Input", "Conv1", "MaxPool", "ResBlock 1", "ResBlock 2"]
b1 = ["Output", "FC Head", "GAP", "ResBlock 4", "ResBlock 3"]

t2 = ["Input", "Stem", "MBConv 1", "MBConv 2"]
b2 = ["Output", "FC Head", "Head", "MBConv 3"]

t3 = ["High-Res Input", "Stem", "MBConv 1", "MBConv 2"]
b3 = ["Output", "FC Head", "Head"]

with open('ascii_test.txt', 'w') as f:
    f.write(generate_ascii(t1, b1, False) + "\n\n")
    f.write(generate_ascii(t2, b2, False) + "\n\n")
    f.write(generate_ascii(t3, b3, True) + "\n\n")

