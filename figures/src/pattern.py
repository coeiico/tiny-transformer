"""
Draw the critical head's attention pattern for pattern.tex: build/pattern.tikz.

One cell per (query, key) pair of the first test list, shaded by the attention weight; the
keys a query cannot see (later positions, removed by the causal mask) are back-hatched; and a
spot ring marks, for each output query, the input positions holding the value just written,
where the README says the head looks. Layout only; the numbers are data/pattern-*.json.
"""

import json

d = json.load(open('data/pattern-default-seed0.json'))
tokens, pattern, n = d['tokens'], d['pattern'], d['input_length']
size = len(tokens)
out = []
# Rows top to bottom are queries 0..size-1; columns left to right are keys 0..size-1.
for q in range(size):
  y = size - 1 - q
  for k in range(size):
    if k > q:
      out.append(f'\\fill[hatch back=ink!16] ({k},{y}) rectangle ++(1,1);')
      continue
    shade = round(100 * min(1.0, pattern[q][k]))
    if shade >= 3:
      out.append(f'\\fill[ink!{shade}] ({k},{y}) rectangle ++(1,1);')
out.append(f'\\mgrid{{(0,0)}}{{({size},{size})}}')
for q, k in d['expected']:
  out.append(f'\\draw[mring] ({k + 0.1},{size - 1 - q + 0.1}) rectangle ++(0.8,0.8);')
# The separator splits the list into the input and the sorted output.
out.append(f'\\draw[ink, line width=0.6pt] ({n},0) -- ({n},{size}) ({n + 1},0) -- ({n + 1},{size});')
out.append(f'\\draw[ink, line width=0.6pt] (0,{size - n}) -- ({size},{size - n}) (0,{size - n - 1}) -- ({size},{size - n - 1});')
for i, tok in enumerate(tokens):
  label = r'\textsc{s}' if tok == 'SEP' else tok
  out.append(f'\\node[font=\\tiny, anchor=east] at (-0.15,{size - 1 - i + 0.5}) {{{label}}};')
  out.append(f'\\node[font=\\tiny, anchor=north] at ({i + 0.5},-0.15) {{{label}}};')
with open('build/pattern.tikz', 'w') as f:
  f.write('\n'.join(out) + '\n')
with open('build/pattern-values.tex', 'w') as f:
  f.write(f'\\def\\patternlayer{{{d["layer"]}}}\\def\\patternhead{{{d["head"]}}}\\def\\patternsize{{{size}}}\\def\\patternn{{{n}}}\n')
