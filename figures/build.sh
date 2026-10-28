#!/bin/sh
# Rebuild every figure: <name>.tex -> <name>.pdf (print), .svg (README), .png (slides).
# Needs pdflatex (TeX Live / MacTeX, with TikZ) and pdftocairo (poppler).
set -e
cd "$(dirname "$0")"
for tex in *.tex; do
  name="${tex%.tex}"
  pdflatex -interaction=nonstopmode -halt-on-error "$tex" > /dev/null
  pdftocairo -svg "$name.pdf" "$name.svg"
  pdftocairo -png -r 200 -singlefile "$name.pdf" "$name"
  rm -f "$name.aux" "$name.log"
done
