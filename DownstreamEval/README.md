
For the downstream evaluation, the original sentences and the normalized sentences have to be aligned and merged (see `merge.py`).
The resulting files are written to the directory `tokenized`. These files are parsed by Alpino (results in directory `parses`), after which the parse trees are used by dutchcoref for coreference resolution (results in directory `coref`).

To reproduce the results, issue the following commands:

```bash
bash conv.sh
bash parsefiles.sh
bash corefdirs.sh
bash parseeval.sh > parseresults.txt
bash corefeval.sh > corefresults.txt
```

Requirements:

- Alpino: http://www.let.rug.nl/vannoord/alp/Alpino/
- dutchcoref: https://github.com/andreasvc/dutchcoref
- coval: https://github.com/andreasvc/coval
- openboek corpus: https://github.com/andreasvc/openboek
