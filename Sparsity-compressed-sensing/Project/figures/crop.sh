#!/bin/bash
for f in *.pdf
do
  #mv $f $f.pdf
  echo "Processing $f file..."
  #pdfcrop --pdftexcmd=pdftex "$f.pdf" "$f.pdf"
  pdfcrop --pdftexcmd=pdftex "$f" "$f"
done
