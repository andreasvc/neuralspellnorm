import re
import sys
import difflib


def fix_gold(gold):
	"""Fix gold sentence."""
	gold = ' '.join(gold).strip()
	combin_words_gold = [('aan te spreken', 'aan@te@spreken'), ('op aan', 'op@aan'),
			('op te merken', 'op@te@merken'), ('om mijnentwille', 'om@mijnentwille'),
			('God weet waarheen', 'God@weet@waarheen'), ('van wie het', 'van@wie@het'),
			('van mijn', 'van@mijn'), ('in plaats', 'in@plaats'), ('rozenbomen hout', 'rozenbomen@hout'),
			(",'s", ", 's"), (" krant ' estaan ", " krant'estaan "), ('XIII .', 'XIII.'),
			('XII .', 'XII.'), ('XI .', 'XI.'), (' toe juichen ', ' toe@juichen '), (' zo -iets ', ' zo@-iets ')]
	for word_pair1 in combin_words_gold:
		if word_pair1[0] in gold:
			gold = gold.replace(word_pair1[0], word_pair1[1])
	adj_gold = []
	for g in gold.split():
		g = g.replace('@', ' ')
		adj_gold.append(g)
	return adj_gold


def fix_pred(pred):
	"""Align prediction sentences."""
	pred = ' '.join(pred).strip()
	combin_words_pred = [(' der ', ' der - '), (' des ', ' des - '), ('S .', 'S.'), ('A .', 'A.'),
			('D .', 'D.'), ('P .', 'P.'), ('Z .', 'Z.'), ('V .', 'V.'), ('I .', 'I.'),
			('b . v .', 'b.v.'), ('W .', 'W.'), ('N .', 'N.'), ('J .', 'J.'),
			('Mrs .', 'Mrs.'), ('H .', 'H.'), ('Dr .', 'Dr.'), ('St .', 'St.'),
			('3 , 37', '3,37'), ('X .', 'X.'), ('G .', 'G.'), (' zooiemand ', ' zoo iemand '),
			('enz .', 'enz.'), ('No .', 'No.'), ('Mr .', 'Mr.')]
	for word_pair2 in combin_words_pred:
		if word_pair2[0] in pred:
			pred = pred.replace(word_pair2[0], word_pair2[1])
	adj_pred = []
	for p in pred.split():
		p = p.replace('@', ' ')
		adj_pred.append(p)
	return adj_pred


def align(orig, norm):
	"""Given an original sentence and a spelling normalized version,
	combine them into a single sentence with Alpino-tags for normalized tokens.

	- @alt: 1-to-1 token replacement
	- @mwu_alt: n-to-1: 2 or more tokens in orig replaced by 1 token in normalized
	- @phantom: 1 new token in normalized, not in orig.
		NB: used when 1 token in orig is split into multiple tokens in normalized.

	>>> align('Dat is waer .', 'Dat is waar .')
	'Dat is [ @alt waar waer ] .'
	>>> align('Dat is van te vooren bepaald .', 'Dat is van tevoren bepaald .')
	'Dat is van [ @mwu_alt tevoren te vooren ] bepaald .'
	>>> align('En dan kon je ervan opaan .', 'En dan kon je ervan op aan .')
	'En dan kon je ervan [ @phantom op ] [ @alt aan opaan ] .'
	"""
	if norm.count('.') > 30:
		return orig
	norm = re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', norm)
	norm = norm.replace('. . .', '...').replace('... .', '....').replace('. .', '..')
	orig = orig.replace(" 's", "'s").replace(" 'm", "'m").replace('... .', '....')
	orig, norm = orig.split(' '), norm.split(' ')
	if len(orig) > len(norm):
		orig = fix_gold(orig)
	if len(orig) < len(norm):
		norm = fix_pred(norm)
	result = ''
	for op, a, b, c, d in difflib.SequenceMatcher(a=orig, b=norm).get_opcodes():
		if op == 'equal':
			result += ' '.join(orig[a:b]) + ' '
		elif op == 'replace':
			if b - a == d - c:
				for n, m in zip(range(a, b), range(c, d)):
					result += '[ @alt %s %s ] ' % (norm[m], orig[n])
			elif b - a < d - c:
				for m in range(c, d - 1):
					result += '[ @phantom %s ] ' % (norm[m])
				result += '[ @alt %s %s ] ' % (' '.join(norm[d - 1:d]), ' '.join(orig[a:b]))
			elif b - a > d - c and d - c == 1:
				result += '[ @mwu_alt %s %s ] ' % (' '.join(norm[c:d]), ' '.join(orig[a:b]))
			elif b - a > d - c and d - c == 2:
				result += '[ @mwu_alt %s %s ] ' % (' '.join(norm[c:d - 1]), ' '.join(orig[a:b - 1]))
				result += '[ @alt %s %s ] ' % (' '.join(norm[d - 1:d]), ' '.join(orig[b - 1:b]))
			else:
				print(orig, file=sys.stderr)
				print(norm, file=sys.stderr)
				print(orig[a:b], file=sys.stderr)
				print(norm[c:d], file=sys.stderr)
				raise ValueError
		elif op == 'insert' and norm[c].strip() and norm[c] != '-':
			result += '[ @phantom %s ] ' % (norm[c])
	return result.rstrip()


def main():
	_, orig, norm = sys.argv
	with open(orig, encoding='utf8') as originp:
		with open(norm, encoding='utf8') as norminp:
			origlines = originp.read().splitlines()
			normlines = norminp.read().splitlines()
			if len(origlines) != len(normlines):
				print(orig, file=sys.stderr)
				print(norm, file=sys.stderr)
				print(len(origlines), len(normlines), file=sys.stderr)
				raise ValueError('different number of lines')
			for origline, normline in zip(origlines, normlines):
				lineid, origline = origline.split('|', 1)
				print(lineid + '|' + align(origline, normline))


if __name__ == '__main__':
	main()
