# Figure Design

How every figure in this repository's README is made and how it looks. The owner chose the
**engraving** style: the look of a figure in a classic journal paper or a well-printed
textbook. Black ink on white, one vermilion spot colour, hatching instead of filled tints,
shape instead of colour, framed axes with inward ticks, and Libertinus type throughout.
Quiet and restrained, but technically dense: every mark carries data or structure.

A figure in this style must still read when printed in black and white. The spot colour
only adds emphasis; nothing depends on it alone.

Change a design value only when the owner asks. The values live in `figures/tex/latentfig.sty`,
which is identical in every repo of the project (a master copy is in `ml-instructions-draft/`
in the project folder), so a change goes into all of them at once.

## Two kinds of plot

- **Diagnostic plots** are the PNGs `train.py` and `analyze.py` write into `runs/<run>/` with
  matplotlib's default backend, for looking at a run. They are quick and plain, their titles
  state the finding, and they never go into the README.
- **README figures** are everything the README shows. They are drawn in LaTeX as described
  here, from data files, and committed as PNGs in `figures/`. Never put a diagnostic PNG, or
  any other matplotlib PNG, in the README.

## Layout and pipeline

```
figures/
├── tex/latentfig.sty    the house style: fonts, colours, notation, styles, canvas
├── tex/figstyle.py      matplotlib through the PGF backend, in the same style
├── tex/roles.tex        this repo's roles: which series, class or object gets which mark or line
├── src/<name>.tex       one standalone document per figure
├── src/<name>.py        optional: prepares build/ files for the figure (contours, sorted tables, .pgf)
├── data/<name>.csv      the figure's numbers, committed
├── build/               LaTeX and PGF intermediates, gitignored
└── <name>.png           the built figure the README shows, committed
```

- **Numbers come from data, never from the figure.** `analyze.py` writes each figure's numbers
  (CSV, `.npz` for grids, JSON for graphs) into its analysis folder. To show a result in the
  README, copy that file into `figures/data/` and name the run it came from in the README
  caption. Figure code never trains, evaluates or analyses anything. A `src/<name>.py` may only
  rearrange data for drawing (extract contours, sort a table, write a values file of macros),
  and labels that quote a result read it from the data, never from a typed-in number. Only an
  illustrative diagram (a mechanism, not a measurement) uses made-up numbers, and its caption
  says "illustrative".
- **`just figures` builds everything.** From `figures/`, it runs each `src/*.py`, then compiles
  each `src/*.tex` with LuaLaTeX (`TEXINPUTS=tex//:`) and turns the PDF into `<name>.png` at
  300 dpi with `pdftoppm`. Paths inside figure sources are relative to `figures/`.
- **Requirements** (not Python packages, so not in `uv.lock`): TeX Live with LuaLaTeX, latexmk,
  pgfplots, siunitx and derivative; the Libertinus fonts (TeX Live's `libertinus-fonts`);
  JuliaMono for code; `pdftoppm` (`poppler-utils`). CI does not build figures: the PNGs are
  committed.

A figure source:

```latex
\documentclass[border=10pt]{standalone}
\usepackage{latentfig}
\input{roles}
\begin{document}
\begin{figurecanvas}
\begin{tikzpicture}
\begin{axis}[engraved, name=a, width=0.39\figwidth, height=0.27\figwidth, xlabel={step}]
  \addplot[series spot] table[x=step, y=val_accuracy, col sep=comma] {data/curves.csv};
\end{axis}
\node[head] at ($(a.north west)+(0,0.25)$) {(a)\enspace Accuracy on 200 held-out points.};
\end{tikzpicture}
\end{figurecanvas}
\end{document}
```

## The canvas

- Every figure is drawn on `figurecanvas`, a white canvas `\figwidth` (6.5 in) wide, with the
  picture centred on it. One width means GitHub scales every PNG by the same factor, so text is
  the same size in every figure.
- White, not transparent, so GitHub's dark mode doesn't put black ink on a dark page.
- Design at the size the README shows: about 880 px across. Text a reader must read is at least
  `\footnotesize`; `\scriptsize` only for small labels on edges and grids.

## Type

| Role | Font |
|---|---|
| Text, numbers and maths | Libertinus Serif and Libertinus Math |
| Panel titles, notes, axis labels | Libertinus Serif *italic* |
| Code, token strings, file names | JuliaMono (`\texttt`) |

Set by `latentfig.sty`; never set fonts in a figure.

## Ink and the spot colour

| Name | Hex | Used for |
|---|---|---|
| `ink` | `#111111` | everything by default: lines, marks, frames, text |
| `soft` | `#555555` | notes, secondary labels |
| `faint` | `#8A8A8A` | hatching, guides, inactive things (dead units, masked entries) |
| `spot` | `#C8372D` | the one thing the figure is about: the model being shown, the errors, the result, the backward pass |

- **One spot colour, used for one meaning per figure.** In a comparison of a model with
  baselines, the model is spot and the baselines are ink. In a figure about errors, the errors
  are spot. If everything is important, nothing is spot.
- **Grey is lighter ink, not a colour.** Use it for what recedes: hatching, inactive units,
  guides. Never use grey to tell two series apart; use line style.
- **No other colours.** No tinted fills, no gradients, no colour maps for categories.

## Regions: hatching

- A region (a class's decision region, a gap between two curves, a masked band) is hatched:
  `hatch` (45°, 3.2 pt), `hatch dense`, `hatch back` (−45°) and `crosshatch`. Hatching is light
  (`ink!45`), so marks on top read first.
- Two regions that must be told apart get different hatch directions, or one hatched and one
  blank. Never two greys.
- A label sitting on hatching gets a white knockout behind it (`knockout`).
- A heatmap whose values matter (weights, activations, attention) uses greys
  (`figstyle.GREYS`, white to ink) or, for signed values, `figstyle.SIGNED` (ink, white, spot),
  with a thin colour bar in ink. Draw it as a matrix grid when it is small enough to rule.

## Marks and lines

- **Classes by shape:** `class open` (open circle), `class filled` (filled dot), `class triangle`,
  `class square`. Never by colour.
- **Series by line style:** `series ink` (solid), `series dashed`, `series dotted`,
  `series dashdot`; the focal series is `series spot`. Several seeds of one run are several thin
  lines of the same style.
- **The points that matter** (errors, a chosen example) are `marked`: a spot ring around the
  point, drawn on top of its class mark, so it shows both the class and the emphasis, and still
  reads in greyscale.
- **Level lines** (where a model is 90% sure, a threshold, chance) are `level line`: fine dotted ink.
- Weights: data lines 0.6 to 0.8 pt, frames 0.5 pt, ticks 0.4 pt, hatching 0.3 pt. Arrow tips
  `Latex`.

## Axes

- `engraved`: framed on all four sides, ticks inside on every side, no grid, ink tick labels in
  Libertinus, axis labels in italic.
- Few ticks, at round values. Percentages as plain numbers with the unit in the axis label
  (`validation accuracy (\%)`). Large counts through siunitx (`\num{13002}`).
- **Label directly.** A series' name sits at the end of its line in its own ink or spot, a
  class's name sits beside it with its mark (`upper moon, $\circ$`). Use a legend only when
  direct labels would collide.

## Titles, notes and keys

- **Panel titles are italic run-ins** ending with a full stop: `(a)\enspace Linear model.`
  (`head`). When a panel has a formula, it goes on its own line under the title (`formula`).
- **Notes under a panel** say what the panel's result is, in italic (`note`): `98.3\% of
  1\,000 test points correct; the 17 errors in red.`
- **A key shared by all panels** goes on one line under the whole figure, centred: what the
  hatching means, what the dotted lines mean.
- No title stating the finding inside a README figure: the finding is the first sentence of the
  README caption, in bold.

## Diagrams

- **Boxes:** a value or a component is a square-cornered `box` in thin ink; an input or a
  parameter is a double-ruled `inputbox`. A box holds its definition, its value and, when the
  figure is about gradients, its gradient in spot.
- **Flow:** the forward pass and data flow are thin ink arrows (`fwd`). Gradients and the
  backward pass are dashed spot arrows (`back`), running parallel beside the forward arrow they
  reverse, offset perpendicular to it, with their label (`backlabel`) upright on the far side.
- **Converging arrows** enter a box at separate points, so arrows never cross at a single point.
- **Networks:** `nnode` neurons, `nnode off` for inactive ones, `nedge` thin edges, `hedge` for
  the highlighted path, `wlab` for an edge's weight written along it.
- **Generalise drawings of general objects:** the first few nodes, cells or bars, then `\vdots` or
  `\cdots`, then the last one labelled with its index ($x_d$, $n_{d_{\text{ff}}}$).
- **Matrix grids:** shading first (greys for magnitude, `mband` hatching for the row or column
  discussed, `mpick` spot for the one entry, `mmask` back-hatching for masked entries), then
  `\mgrid` for faint rules and an ink frame. Highlights land on whole cells.
- **Bars** (`\pbar`): an ink outline over light hatching, the focal bar in spot, on an ink
  baseline (`pbase`).
- Put the matching formula beside a diagram of a computation, in ink, and read every number in
  it from the same data as the diagram.

## Notation in figures

Labels are mathematics in the notation of the owner's ai-math notes: vectors `\vb{x}` (bold
italic), matrices `\mb{W}` (bold upright), transpose `\tp`, never `^T`; operators upright
(`\softmax`, `\ReLU`, `\E`); word subscripts upright (`\mathcal{L}_{\text{CE}}`); a tensor's shape
small and soft, `\shp{T \times d}`; index ranges `i \in [T]`, never `[1, T]`.

## Which tool for which figure

| Figure | Tool |
|---|---|
| Curves, scatters up to a few thousand points, line charts | pgfplots with `engraved`, data from `data/` |
| Decision regions and level lines | pgfplots: hatched region polygons and contour lines that `src/<name>.py` extracts into `build/` |
| Heatmaps, images, large scatters, many small maps | matplotlib through `figstyle.py` (PGF backend), `\import`ed into the canvas |
| Computation graphs, pipelines, networks | TikZ with `box`, `inputbox`, `fwd`, `back`, `nnode` |
| Matrices, attention patterns, weight grids | TikZ fills, then `\mgrid` |
| An object passing through several steps | tikz-cd: each object with its shape underneath, each step a labelled arrow |
| Graphical models | `tikz-bayesnet` (load the `bayesnet` library in the figure) |

## TikZ pitfalls

- A role or style name must not be a TikZ key: `baseline`, `left`, `right`, `above`, `below`,
  `scale`, `label`, `node` and `level` are taken. `\addplot[baseline]` silently sets picture alignment
  instead of a style.
- `fill between` replaces the picture's layer list. A figure that uses it and draws on the
  background layer declares a layer set that includes `background` (see any figure using both).
- Position a second axis from the first one's plot area, `at={($(a.east -| a.outer east)+(…)$)}`
  with `anchor=west`; `anchor=outer west` is not supported with layers.
- xcolor mixes alternate colour and percentage: `ink!45` works, `ink!45!60` does not.

## Checking

After adding or changing a figure, build it and look at the PNG at the size the README shows it.

- No overlapping labels, no label crossed by a line or sitting on unknocked-out hatching, nothing
  clipped at the canvas edge, panel titles on one baseline.
- It reads in greyscale: convert the PNG to grey and check that nothing becomes ambiguous.
- The build log (`figures/build/<name>.log`) has no errors, no `Overfull` boxes and no
  `Missing character` warnings; `pdffonts figures/build/<name>.pdf` lists Libertinus and no Latin
  Modern or Computer Modern.
- Every number in the figure traces to a file in `figures/data/`, and the caption names the run
  that file came from.

## In the README

```md
![The linear model's and the MLP's decision boundaries over the test points](figures/boundary.png)

*__Two hidden layers bend the boundary round both moons.__ Runs `linear-seed0` and
`default-seed0`. …*
```

The alt text says what the figure shows. The caption opens with the finding in bold, names the
runs, and then explains how to read the figure.
