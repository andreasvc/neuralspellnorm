#!/bin/bash
# Run coreference resolution on a directory with Alpino parse trees for a single book
set -e
ROOT="/mnt/local/tmp/andreas/clin2023spelling/"
DUTCHCOREF="/mnt/local/tmp/andreas/code/dutchcoref"
OUT="$ROOT/coref/$1"
TGZ="$(basename "$1").tar.gz"

if [ -f "$(dirname $OUT)/$TGZ" ]; then
	echo "already done: $1"
else
	echo "resolving coref for $1"
	mkdir -p "$OUT"
	cd "$DUTCHCOREF"

	# redirect stderr to stdout, so that we have all messages in the log
	time python3 coref.py "$ROOT/parses/$1" \
		--outputprefix "$OUT/coref" \
		--fmt conll2012 \
		2>&1
	# --neural=span,feat,pron,quote \

	# time python3 coref.py "$ROOT/parses/$1" \
	# 	--outputprefix "$OUT/frag" \
	# 	--fmt booknlp \
	# 	--slice=500:1000 \
	# 	2>&1

	cd "$(dirname "$OUT")"
	tar -czf "$TGZ" "$(basename "$1")"

	echo "done with $1"
fi

echo
