#!/bin/bash
set -e
MAXPROC=30

source /mnt/local/tmp/andreas/code/dutchcoref/venv/bin/activate

cd parses
# restrict find output to leaf directories https://stackoverflow.com/a/4269862
# e.g,. 'corpus_1stpers_clean/Akyol_Eus_clean' but not 'corpus_1stpers_clean'
# use GNU parallel
find . -type d -links 2 -print0 \
    | parallel \
		--null \
		--quote \
		--max-args 1 \
		--max-procs $MAXPROC \
		--progress \
		bash ../corefdir.sh \
	| tee >(gzip > ../coref.log.gz)
# write to stdout AND to compressed file https://unix.stackexchange.com/a/86864
