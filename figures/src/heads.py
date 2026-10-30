"""
Draw the head-ablation grids for heads.tex: build/heads-<seed>.tikz, one per seed.

Each cell is one head (rows are layers, columns heads), shaded by how much mean-ablating it
raises test loss, on a square-root scale shared by all seeds so the grids compare. The
critical head gets a spot ring. Beside each row, the token accuracy lost when that layer's
attention is ablated whole. Layout only; the numbers are data/heads-*.csv and layers-*.csv.
"""

import csv
import math

RUNS = ('default-seed0', 'default-seed1', 'default-seed2')


def rows(path: str) -> list[dict[str, str]]:
  with open(path) as f:
    return list(csv.DictReader(f))


heads = {run: rows(f'data/heads-{run}.csv') for run in RUNS}
largest = max(float(h['loss_rise']) for hs in heads.values() for h in hs)

for run, hs in heads.items():
  n_layers = 1 + max(int(h['layer']) for h in hs)
  worst = max(hs, key=lambda h: float(h['loss_rise']))
  out = []
  for h in hs:
    layer, head, rise = int(h['layer']), int(h['head']), max(float(h['loss_rise']), 0.0)
    # Grid coordinates: column = head, row 0 (layer 0) at the top.
    x, y = head, n_layers - 1 - layer
    shade = round(100 * math.sqrt(rise / largest))
    if shade >= 2:
      out.append(f'\\fill[ink!{shade}] ({x},{y}) rectangle ++(1,1);')
    if rise >= 0.095:
      colour = 'white' if shade > 50 else 'ink'
      out.append(f'\\node[font=\\tiny, text={colour}] at ({x + 0.5},{y + 0.5}) {{{rise:.1f}}};')
  out.append(f'\\mgrid{{(0,0)}}{{(8,{n_layers})}}')
  wx, wy = int(worst['head']), n_layers - 1 - int(worst['layer'])
  out.append(f'\\draw[mring] ({wx - 0.08},{wy - 0.08}) rectangle ++(1.16,1.16);')
  for layer in rows(f'data/layers-{run}.csv'):
    y = n_layers - 1 - int(layer['layer'])
    out.append(f'\\node[font=\\scriptsize, anchor=west] at (8.25,{y + 0.5}) {{{100 * float(layer["token_accuracy_drop"]):.1f}}};')
  with open(f'build/heads-{run}.tikz', 'w') as f:
    f.write('\n'.join(out) + '\n')
