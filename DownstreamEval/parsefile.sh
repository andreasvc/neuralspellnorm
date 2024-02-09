#!/bin/bash
set -e
PAR="${1%.tok}/"
TGZ="${PAR%/}.tar.gz"

if [ -f "parses/$TGZ" ]; then
	echo "already done: $1"
else
	echo parsing $1

	# Parse tokenized file
	mkdir -p "parses/$PAR"
	cd parses
	$ALPINO_HOME/bin/Alpino -veryfast -flag treebank "$PAR" end_hook=xml -parse < "../$1"
	tar czf "$TGZ" "$PAR"

	echo done with $1
	echo
fi
