#!/bin/bash
set -e
MAXPROC=50
# use GNU parallel
find tokenized/ByT5_pre_books_pred10k -name '*.tok' -print0 \
    | parallel \
		--null \
		--quote \
		--max-args 1 \
		--max-procs $MAXPROC \
		--progress \
		bash parsefile.sh \
	| tee >(gzip > parsing.log.gz)
