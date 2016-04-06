#!/usr/bin/python
#coding: utf-8

import os
import codecs
import operator
import pandas as pd

__all__ = ['TextProcessor', 'TextClassifier']

class TextProcessor:

	processed_dir = 'processed'

	vocabulary = {'rus': [u'а', u'б', u'в', u'г', u'д', u'е', u'ё', u'ж', u'з', u'и', u'й', u'к',
           u'л', u'м', u'н', u'о', u'п', u'р', u'с', u'т', u'у', u'ф', u'х', u'ц',
           u'ч', u'ш', u'щ', u'ъ', u'ы', u'ь', u'э', u'ю', u'я']}

	@staticmethod
	def in_voc(sym, voc='rus'):
		alphabet = TextProcessor.vocabulary[voc]
		return sym in alphabet

	@staticmethod
	def generate_empty_dict(size, lang='rus'):
		alphabet = TextProcessor.vocabulary[lang]

		gen_n_grams = []
		if (size == 1): gen_n_grams = alphabet[:]

		if (size == 2):
			for i in alphabet:
				for j in alphabet:
					res_str = i + j
					gen_n_grams.append(res_str)

		if(size == 3):
			for i in alphabet:
				for j in alphabet:
					for k in alphabet:
						res_str = i + j + k
						gen_n_grams.append(res_str)

		return dict((k,0) for k in gen_n_grams)

	@staticmethod
	def preprocess(root_path, lang='rus'):
		print "Try to preprocess documents in %s" % root_path

		for dr in os.listdir(root_path):
			author_path = root_path + '/' + dr
			if os.path.isdir(author_path):
				# print author_path
				processed_path = author_path + '/' + TextProcessor.processed_dir

				if not os.path.exists(processed_path):
					os.makedirs(processed_path)

				for text in os.listdir(author_path):
					if text.endswith(".txt"):
						TextProcessor.process_document(author_path, text, lang)

	@staticmethod
	def process_document(author_path, text, lang):
		# print ('\t%s/%s' % (author_path, text))
		origin_text_path = author_path + '/' + text
		content_origin = codecs.open(origin_text_path, encoding='utf-8')
		content_processed = ''.join(e for e in content_origin.read().lower() if TextProcessor.in_voc(e, voc=lang))
		content_origin.close()

		processed_path = author_path + '/' + TextProcessor.processed_dir + '/' + text
		processed_file = codecs.open(processed_path, 'w', encoding='utf-8')
		processed_file.write(content_processed)
		processed_file.close()

	@staticmethod
	def get_processed_data(path):
		df = pd.DataFrame(columns=['Category', 'Item', 'Text'])

		for author in os.listdir(path):
			author_path = path + '/' + author

			if os.path.isdir(author_path):
				processed_path = author_path + '/' + TextProcessor.processed_dir

				if not os.path.exists(processed_path):
					TextProcessor.preprocess_train(path)

				for text in os.listdir(processed_path):
					if text.endswith(".txt"):
						text_path = processed_path + '/' + text
						content = open(text_path, "r")
						df = df.append(
							{
								"Category": author, 
								"Item": text[:-4], 
								"Text": content.read()
							}, ignore_index=True )
						content.close()

		return df

class TextClassifier: 
	n_gram = 0
	train = pd.DataFrame()
	metrics = ['l1']

	def __init__(self, n_gram = 2):
		self.n_gram = 3 if n_gram > 3 else 1 if n_gram < 1 else n_gram

	@staticmethod
	def get_n_gram_dict(text, n):
		ngc = TextProcessor.generate_empty_dict(n)

		text = text.decode('utf-8')
		for i in range(len(text) - n + 1):
			ng_curr = text[i:i+n]
			ngc[ng_curr] += 1

		return ngc

	@staticmethod
	def combine_dicts(a, b, op=operator.add):
		return dict(a.items() + b.items() +
			[(k, op(a[k], b[k])) for k in set(b) & set(a)])

	@staticmethod
	def l1_distance(corpus, item):
		dist = 0
		for sym in item:
			dist += abs(corpus[sym] - item[sym])
		return dist

	@staticmethod
	def calculate_distance(corpus, item_row, metric='l1'):
		corpus_dict = {}
		sum_len = 0
		item_len = len(item_row['Text'].decode('utf-8'))
		item_dict = item_row['voc'].copy()

		for i, row in corpus.iterrows():
			corpus_dict = TextClassifier.combine_dicts(corpus_dict, row['voc'])
			sum_len += len(row['Text'].decode('utf-8'))

		for key in corpus_dict:
			corpus_dict[key] = corpus_dict[key] / float(sum_len)

		for key in item_dict:
			item_dict[key] = item_dict[key] / float(item_len)

		if (metric not in TextClassifier.metrics):
			raise ValueError("metric must be `l1`")

		return TextClassifier.l1_distance(corpus_dict, item_dict)

	def fit_transform(self, data):
		self.train = data.copy()

		self.train['voc'] = self.train.apply(lambda row: TextClassifier.get_n_gram_dict(row['Text'], self.n_gram), axis=1)

		for cat in self.train['Category'].unique():
			self.train['dist_' + cat] = self.train.apply(
				lambda row: TextClassifier.calculate_distance(
					self.train[(self.train['Category'] == cat) & (self.train['Item'] != row['Item'])],
					row
				), axis=1
			)

		return self.train

	def calculate_error_rate(self):
		num_errors = 0
		cols = [name for name in self.train.columns if name.startswith('dist_')]
		for i, row in self.train.iterrows():
			if (row['dist_' + row['Category']] != min(row[cols])): num_errors += 1

		return num_errors / float(len(self.train))