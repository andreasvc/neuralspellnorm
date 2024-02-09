#!/bin/bash
set -e

echo Original
paste -d " " <(ls $HOME/code/openboek/parses/Nescio_Titaantjes/*.xml) <(ls parses/Original/Nescio_Titaantjes/*.xml) | Alpino -compare_xml_files | tail

echo RuleBased
paste -d " " <(ls $HOME/code/openboek/parses/Nescio_Titaantjes/*.xml) <(ls parses/RuleBased_pred/Nescio_Titaantjes/*.xml) | Alpino -compare_xml_files | tail

echo Neural
paste -d " " <(ls $HOME/code/openboek/parses/Nescio_Titaantjes/*.xml) <(ls parses/ByT5_pre_books_pred10k/Nescio_Titaantjes/*.xml) | Alpino -compare_xml_files | tail

echo GoldSpelling
paste -d " " <(ls $HOME/code/openboek/parses/Nescio_Titaantjes/*.xml) <(ls parses/Gold/Nescio_Titaantjes/*.xml) | Alpino -compare_xml_files | tail
