import csv
import hanja

concat_w = []
with open('./concat_words.tsv', 'r', encoding='utf-8') as f:
    tr = csv.reader(f, delimiter='\t')
    for _ in tr:
        concat_w.append(_)

concat_w = sorted(concat_w, key=lambda x: len(x[1]), reverse=True)


def blank_and_hanja_and_concat(str1, concat_w):
    s = hanja.translate(str1, 'substitution')
    s = s.replace("(", " (").replace(")", ") ").replace("  (", " (").replace(")  ", ") ") \
        .replace("{", " {").replace("}", "} ").replace("[", " [").replace("]", "] ")\
        .replace(":", " :").replace("  :",  " :")
    s = s.replace("%uF98E", "연").replace("%uF9B5", "예")

    for c in concat_w:
        s = s.replace(c[0], c[1])
    s = s.strip()

    return s
