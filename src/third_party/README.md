# Third-party code

## relmodpy

`relmodpy/` is a verbatim copy of **RelModPy** by Fabian Gittins
(https://github.com/fgittins/RelModPy), vendored here so that the f-mode
computation reproduces without a separate installation step.

RelModPy is released under the MIT License; its licence text is in
`relmodpy/LICENSE` and applies to every file in that directory. It is **not**
covered by this repository's own licence.

The code is used unmodified. Everything specific to this work — the equation-of-state
adapters and the replacement of the diverging Muller root finder by an |A_in| scan
plus a two-dimensional Nelder-Mead minimisation — lives in `src/polar_sectors/`,
not here.

Method references implemented by RelModPy: Lindblom & Detweiler (1985) and
Andersson, Kokkotas & Schutz (1995).
