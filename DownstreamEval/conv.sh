#!/bin/bash
set -e

for a in $HOME/src/AWolters-Master-Thesis/Evaluation/*/
do
	if [[ $(basename $a) == "Gold" || $(basename $a) == "RuleBased_pred" ]]; then
		continue
	fi
	OUT=tokenized/$(basename $a)
	mkdir -p $OUT
	python3 merge.py \
		$HOME/code/openboek/tokenized/Nescio_Titaantjes.tok \
		$a/Nescio_Titaantjes_pred.txt \
		>$OUT/Nescio_Titaantjes.tok

	python3 merge.py \
		$HOME/code/openboek/tokenized/Multatuli_MaxHavelaar.tok \
		$a/Multatuli_MaxHavelaar_pred.txt \
		>$OUT/Multatuli_MaxHavelaar.tok

	python3 merge.py \
		$HOME/code/openboek/tokenized/ConanDoyle_SherlockHolmesDeAgraSchat.tok \
		$a/ConanDoyle_SherlockHolmesDeAgraSchat_pred.txt \
		>$OUT/ConanDoyle_SherlockHolmesDeAgraSchat.tok
done
