#!/bin/bash
set -e
cd $HOME/code/coval

for a in /mnt/local/tmp/andreas/clin2023spelling/coref/*
do
	echo $a
	echo Nescio_Titaantjes
	python3 scorer.py \
		$HOME/code/openboek/coref/Nescio_Titaantjes.conll \
		$a/Nescio_Titaantjes/coref.conll

	echo Multatuli_MaxHavelaar
	python3 scorer.py \
		$HOME/code/openboek/coref/Multatuli_MaxHavelaar.conll \
		$a/Multatuli_MaxHavelaar/coref.conll

	echo ConanDoyle_SherlockHolmesDeAgraSchat
	python3 scorer.py \
		$HOME/code/openboek/coref/ConanDoyle_SherlockHolmesDeAgraSchat.conll \
		$a/ConanDoyle_SherlockHolmesDeAgraSchat/coref.conll
	echo ================================================
done

