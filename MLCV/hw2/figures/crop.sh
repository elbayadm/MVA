#!/bin/bash
for f in *.pdf
do
  echo "Processing $f file..."
  pdfcrop --pdftexcmd=pdftex "$f" "$f"
done
#mv -f *pdf ..