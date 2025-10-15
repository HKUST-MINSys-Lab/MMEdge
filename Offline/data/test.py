with open ('label_sorted.txt', 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]

with open ('all_words.txt', 'w') as f:
    for idx, line in enumerate(lines):
        f.write(line + '\t' + str(idx) + '\n')    