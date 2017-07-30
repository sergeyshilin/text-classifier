#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
import operator
import numpy as np

__all__ = ['TextProcessor', 'TextUtils']

class TextProcessor:
    """Text processor class
    Make sure you have all your documents in UTF-8 encoding
    
    Attributes:
        processed_dir (str): Proccessed files folder
        vocabulary (dict): All languages vocabularies (available: 'rus', 'eng', 'eng_con', 'eng_vow')
    
    Functions:
        \"\"\" Remove all unneccessary symbols from .txt file
        preprocess_document(document_path, document_name, lang='eng')

        \"\"\" Remove all unneccessary symbols from every .txt file in :folder_path
        preprocess_folder(folder_path, lang='eng', join=False)
    """
    processed_dir = 'processed'

    vocabulary = {
        'ru': [u'а', u'б', u'в', u'г', u'д', u'е', u'ё', u'ж', u'з', u'и', u'й', u'к',
                u'л', u'м', u'н', u'о', u'п', u'р', u'с', u'т', u'у', u'ф', u'х', u'ц',
                u'ч', u'ш', u'щ', u'ъ', u'ы', u'ь', u'э', u'ю', u'я'],
        'ru_con': [u'б', u'в', u'г', u'д', u'ж', u'з', u'й', u'к', u'л', u'м', u'н', u'п',
                u'р', u'с', u'т', u'ф', u'х', u'ц', u'ч', u'ш', u'щ', u'ъ', u'ь'],

        'ru_vow': [u'а', u'е', u'ё', u'и', u'о', u'у', u'ы', u'э', u'ю', u'я'],

        'en': [u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l',
                u'm', u'n', u'o', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', 
                u'y', u'z'],

        'en_con': [u'b', u'c', u'd', u'f', u'g', u'h', u'j', u'k', u'l', u'm', u'n', 
                    u'p', u'q', u'r', u's', u't', u'v', u'w', u'x', u'z'],

        'en_vow': [u'a', u'e', u'i', u'o', u'u', u'y'],

        'it': [u'a', u'à', u'b', u'c', u'd', u'e', u'è', u'é', u'f', u'g', u'h', u'i',
                u'ì', u'í', u'î', u'l', u'm', u'n', u'o', u'ò', u'ó', u'p', u'q', u'r', 
                u's', u't', u'u', u'ù', u'ú', u'v', u'z'],

        'it_con': [u'b', u'c', u'd', u'f', u'g', u'h', u'l', u'm', u'n', u'p', u'q', u'r', 
                    u's', u't', u'v', u'z'],

        'it_vow': [u'a', u'à', u'e', u'è', u'é', u'i', u'ì', u'í', u'î', u'o', u'ò', u'ó',
                    u'u', u'ù', u'ú'],

        'de': [u'a', u'ä', u'b', u'ß', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', 
                u'l', u'm', u'n', u'o', u'ö', u'p', u'q', u'r', u's', u't', u'u', u'ü', u'v', 
                u'w', u'x', u'y', u'z'],

        'de_con': [u'b', u'ß', u'c', u'd', u'f', u'g', u'h', u'j', u'k', u'l', u'm', u'n', 
                    u'p', u'q', u'r', u's', u't', u'v', u'w', u'x', u'z'],

        'de_vow': [u'a', u'ä', u'e', u'i', u'o', u'ö', u'u', u'ü', u'y'],

        'fr': [u'a', u'à', u'â', u'æ', u'b', u'c', u'ç', u'd', u'e', u'é', u'è', u'ê', u'ë', 
                u'f', u'g', u'h', u'i', u'î', u'ï', u'j', u'k', u'l', u'm', u'n', u'o', u'ô', 
                u'œ', u'p', u'q', u'r', u's', u't', u'u', u'ù', u'û', u'ü', u'v', u'w', u'x', 
                u'y', u'ÿ', u'z'],

        'fr_con': [u'b', u'c', u'ç', u'd', u'f', u'g', u'h', u'j', u'k', u'l', u'm', u'n', 
                    u'p', u'q', u'r', u's', u't', u'v', u'w', u'x', u'z'],

        'fr_vow': [u'a', u'à', u'â', u'æ', u'e', u'é', u'è', u'ê', u'ë', u'i', u'î', u'ï', u'o',
                    u'ô', u'œ', u'u', u'ù', u'û', u'ü', u'y', u'ÿ'],

        'hr': [u'a', u'b', u'c', u'č', u'ć', u'd', u'ǆ', u'đ', u'e', u'f', u'g', u'h', u'i', 
                u'j', u'k', u'l', u'ǉ', u'm', u'n', u'ǌ', u'o', u'p', u'r', u's', u'š', u't',
                u'u', u'v', u'z', u'ž'],

        'hr_con': [u'b', u'c', u'č', u'ć', u'd', u'ǆ', u'đ', u'f', u'g', u'h', u'j', u'k', u'l',
                    u'ǉ', u'm', u'n', u'ǌ', u'p', u'r', u's', u'š', u't', u'v', u'z', u'ž'],

        'hr_vow': [u'a', u'e', u'i', u'o', u'u'],

        'cs': [u'a', u'á', u'b', u'c', u'č', u'd', u'ď', u'e', u'é', u'ě', u's', u'f', u'g',
                u'h', u'i', u'í', u'j', u'k', u'l', u'm', u'n', u'ň', u'o', u'ó', u'p', u'q',
                u'r', u'ř', u's', u'š', u't', u'ť', u'u', u'ú', u's', u'ů', u'v', u'w', u'x',
                u'y', u'ý', u'z', u'ž'],

        'cs_con': [u'b', u'c', u'č', u'd', u'ď', u'f', u'g', u'h', u'j', u'k', u'l', u'm', u'n',
                    u'ň', u'p', u'r', u'ř', u's', u'š', u't', u'ť', u'v', u'x', u'z', u'ž'],

        'cs_vow': [u'a', u'á', u'e', u'é', u'ě', u'i', u'í', u'o', u'ó', u'u', u'ú', u'ů', u'y', u'ý'],

        'da': [u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm',
                u'n', u'o', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y', u'z',
                u'æ', u'ø', u'å'],

        'da_con': [u'b', u'c', u'd', u'f', u'g', u'h', u'j', u'k', u'l', u'm', u'n', u'p', u'q',
                    u'r', u's', u't', u'v', u'w', u'x', u'z'],

        'da_vow': [u'a', u'e', u'i', u'o', u'u', u'y', u'æ', u'ø', u'å'],

        'pl': [u'a', u'ą', u'b', u'c', u'ć', u'd', u'e', u'ę', u'f', u'g', u'h', u'i', u'j',
                u'k', u'l', u'ł', u'm', u'n', u'ń', u'o', u'ó', u'p', u'r', u's', u'ś', u't',
                u'u', u'w', u'y', u'z', u'ź', u'ż'],

        'pl_con': [u'b', u'c', u'ć', u'd', u'f', u'g', u'h', u'j', u'k', u'l', u'ł', u'm', u'n', u'ń',
                u'p', u'r', u's', u'ś', u't', u'w', u'z', u'ź', u'ż'],

        'pl_vow': [u'a', u'ą', u'e', u'ę', u'i', u'o', u'ó', u'u', u'y'],

        'ro': [u'a', u'ă', u'â', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'î', u'j', u'k',
                u'l', u'm', u'n', u'o', u'p', u'q', u'r', u's', u'ș', u't', u'ț', u'u', u'v', u'w',
                u'x', u'y', u'z'],

        'ro_con': [u'b', u'c', u'd', u'f', u'g', u'h', u'j', u'k', u'l', u'm', u'n', u'p', u'q',
                    u'r', u's', u'ș', u't', u'ț', u'v', u'w', u'x', u'z'],

        'ro_vow': [u'a', u'ă', u'â', u'e', u'i', u'î', u'o', u'u', u'y'],

        'sr': [u'а', u'б', u'в', u'г', u'д', u'ђ', u'е', u'ж', u'з', u'и', u'ј', u'к', u'л',
                u'љ', u'м', u'н', u'њ', u'о', u'п', u'р', u'с', u'т', u'ћ', u'у', u'ф', u'х',
                u'ц', u'ч', u'џ', u'ш'],

        'sr_con': [u'б', u'в', u'г', u'д', u'ђ', u'ж', u'з', u'ј', u'к', u'л', u'љ', u'м', u'н',
                    u'њ', u'п', u'р', u'с', u'т', u'ћ', u'ф', u'х', u'ц', u'ч', u'џ', u'ш'],

        'sr_vow': [u'а', u'е', u'и', u'о', u'у'],

        'es': [u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm',
                u'n', u'ñ', u'o', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y', u'z'],

        'es_con': [u'b', u'c', u'd', u'f', u'g', u'h', u'j', u'k', u'l', u'm', u'n', u'ñ', u'p', 
                    u'q', u'r', u's', u't', u'v', u'w', u'x', u'z'],

        'es_vow': [u'a', u'e', u'i', u'o', u'u', u'y'],

        'sv': [u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm', u'n',
                u'o', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y', u'z', u'å', u'ä', u'ö'],

        'sv_con': [u'b', u'c', u'd', u'f', u'g', u'h', u'j', u'k', u'l', u'm', u'n', u'p', u'q', u'r',
                    u's', u't', u'v', u'w', u'x', u'z'],

        'sv_vow': [u'a', u'e', u'i', u'o', u'u', u'y', u'å', u'ä', u'ö'],

        'el': [u'α', u'β', u'γ', u'δ', u'ε', u'ζ', u'η', u'θ', u'ι', u'κ', u'λ', u'μ',
                u'ν', u'ξ', u'ο', u'π', u'ρ', u'σ', u'ς', u'τ', u'υ', u'φ', u'χ', u'ψ', u'ω'],

        'el_ext': [u'α', u'β', u'γ', u'δ', u'ε', u'ζ', u'η', u'θ', u'ι', u'κ', u'λ', u'μ',
                u'ν', u'ξ', u'ο', u'π', u'ρ', u'σ', u'ς', u'τ', u'υ', u'φ', u'χ', u'ψ', u'ω',
                u'ἀ', u'ἁ', u'ἂ', u'ἃ', u'ἄ', u'ἅ', u'ἆ', u'ἇ', u'ἐ', u'ἑ', u'ἒ', u'ἓ', u'ἔ',
                u'ἕ', u'ἠ', u'ἡ', u'ἢ', u'ἣ', u'ἤ', u'ἥ', u'ἦ', u'ἧ', u'ἰ', u'ἱ', u'ἲ', u'ἳ',
                u'ἴ', u'ἵ', u'ἶ', u'ἷ', u'ὀ', u'ὁ', u'ὂ', u'ὃ', u'ὄ', u'ὅ', u'ὐ', u'ὑ', u'ὒ',
                u'ὓ', u'ὔ', u'ὕ', u'ὖ', u'ὗ', u'ὠ', u'ὡ', u'ὢ', u'ὣ', u'ὤ', u'ὥ', u'ὦ', u'ὧ',
                u'ὰ', u'ά', u'ὲ', u'έ', u'ὴ', u'ή', u'ὶ', u'ί', u'ὸ', u'ό', u'ὺ', u'ύ', u'ὼ',
                u'ώ', u'ᾀ', u'ᾁ', u'ᾂ', u'ᾃ', u'ᾄ', u'ᾅ', u'ᾆ', u'ᾇ', u'ᾐ', u'ᾑ', u'ᾒ', u'ᾓ',
                u'ᾔ', u'ᾕ', u'ᾖ', u'ᾗ', u'ᾠ', u'ᾡ', u'ᾢ', u'ᾣ', u'ᾤ', u'ᾥ', u'ᾦ', u'ᾧ', u'ᾰ',
                u'ᾱ', u'ᾲ', u'ᾳ', u'ᾴ', u'ᾶ', u'ᾷ', u'ῂ', u'ῃ', u'ῄ', u'ῆ', u'ῇ', u'ῐ', u'ῑ',
                u'ῒ', u'ΐ', u'ῖ', u'ῗ', u'ῠ', u'ῡ', u'ῢ', u'ΰ', u'ῤ', u'ῥ', u'ῦ', u'ῧ', u'ῲ',
                u'ῳ', u'ῴ', u'ῶ', u'ῷ', u'έ', u'ύ', u'ή', u'ϊ', u'ώ', u'ί', u'ϋ', u'ά', u'ό']
    }

    english_dict = {u'b': u'b', u'c': u'c', u'č': u'c', u'ć': u'c', u'd': u'd', u'ǆ': u'd', u'đ': u'd', 
                    u'f': u'f', u'g': u'g', u'h': u'h', u'j': u'j', u'k': u'k', u'l': u'l', u'ǉ': u'l',
                    u'm': u'm', u'n': u'n', u'ǌ': u'n', u'p': u'p', u'r': u'r', u's': u's', u'š': u's',
                    u't': u't', u'v': u'v', u'z': u'z', u'ž': u'z', u'ß': u's', u'q': u'q', u'w': u'w', 
                    u'x': u'x', u'ñ': u'n', u'ç': u'c', u'ș': u's', u'ț': u't', u'б': u'b', u'в': u'v',
                    u'г': u'g', u'д': u'd', u'ђ': u'd', u'ж': u'z', u'з': u'z', u'ј': u'j', u'к': u'k',
                    u'л': u'l', u'љ': u'l', u'м': u'm', u'н': u'n', u'њ': u'n', u'п': u'p', u'р': u'r',
                    u'с': u's', u'т': u't', u'ћ': u'c', u'ф': u'f', u'х': u'h', u'ц': u'c', u'ч': u'c',
                    u'џ': u'd', u'ш': u's', u'й': u'j', u'щ': u's', u'ъ': u'', u'ь': u'', u'ł': u'l',
                    u'ń': u'n', u'ś': u's', u'ź': u'z', u'ż': u'z', u'ď': u'd', u'ň': u'n', u'ř': u'r',
                    u'ť': u't', u'а': u'a', u'о': u'o', u'э': u'e', u'у': u'u', u'и': u'i', u'е': u'e',
                    u'ё': u'e', u'я': u'a', u'ы': u'y', u'a': u'a', u'o': u'o', u'i': u'i', u'e': u'e',
                    u'u': u'u', u'á': u'a', u'é': u'e', u'ě': u'e', u'í': u'i', u'ó': u'o', u'ú': u'u',
                    u'ů': u'u', u'ý': u'y', u'y': u'y', u'å': u'a', u'æ': u'a', u'ø': u'o', u'à': u'a',
                    u'â': u'a', u'è': u'e', u'ê': u'e', u'ë': u'e', u'ï': u'i', u'î': u'i', u'ô': u'o',
                    u'œ': u'o', u'ù': u'u', u'û': u'u', u'ü': u'u', u'ÿ': u'y', u'ì': u'i', u'ò': u'o',
                    u'ą': u'a', u'ę': u'e', u'ă': u'a', u'ä': u'a', u'ö': u'o', u'ю': u'u'}

    diacritics_dict = {u'ἀ': u'α', u'ἁ': u'α', u'ἂ': u'α', u'ἃ': u'α', u'ἄ': u'α', u'ἅ': u'α',
                       u'ἆ': u'α', u'ἇ': u'α', u'ά': u'α', u'ὰ': u'α', u'ά': u'α', u'ᾀ': u'α',
                       u'ᾁ': u'α', u'ᾂ': u'α', u'ᾃ': u'α', u'ᾄ': u'α', u'ᾅ': u'α', u'ᾆ': u'α',
                       u'ᾇ': u'α', u'ᾰ': u'α', u'ᾱ': u'α', u'ᾲ': u'α', u'ᾳ': u'α', u'ᾴ': u'α',
                       u'ᾶ': u'α', u'ᾷ': u'α',
                       u'ἐ': u'ε', u'ἑ': u'ε', u'ἒ': u'ε', u'ἓ': u'ε', u'ἔ': u'ε', u'ἕ': u'ε',
                       u'ὲ': u'ε', u'έ': u'ε', u'έ': u'ε',
                       u'ἠ': u'η', u'ἡ': u'η', u'ἢ': u'η', u'ἣ': u'η', u'ἤ': u'η', u'ἥ': u'η',
                       u'ἦ': u'η', u'ἧ': u'η', u'ὴ': u'η', u'ή': u'η', u'ᾐ': u'η', u'ᾑ': u'η',
                       u'ᾒ': u'η', u'ᾓ': u'η', u'ᾔ': u'η', u'ᾕ': u'η', u'ᾖ': u'η', u'ᾗ': u'η',
                       u'ῂ': u'η', u'ῃ': u'η', u'ῄ': u'η', u'ῆ': u'η', u'ῇ': u'η', u'ή': u'η',
                       u'ἰ': u'ι', u'ἱ': u'ι', u'ἲ': u'ι', u'ἳ': u'ι', u'ἴ': u'ι', u'ἵ': u'ι',
                       u'ἶ': u'ι', u'ἷ': u'ι', u'ὶ': u'ι', u'ί': u'ι', u'ῐ': u'ι', u'ῑ': u'ι',
                       u'ῒ': u'ι', u'ΐ': u'ι', u'ῖ': u'ι', u'ῗ': u'ι', u'ϊ': u'ι', u'ί': u'ι',
                       u'ὀ': u'ο', u'ὁ': u'ο', u'ὂ': u'ο', u'ὃ': u'ο', u'ὄ': u'ο', u'ὅ': u'ο',
                       u'ὸ': u'ο', u'ό': u'ο', u'ό': u'ο',
                       u'ὐ': u'υ', u'ὑ': u'υ', u'ὒ': u'υ', u'ὓ': u'υ', u'ὔ': u'υ', u'ὕ': u'υ',
                       u'ὖ': u'υ', u'ὗ': u'υ', u'ὺ': u'υ', u'ύ': u'υ', u'ῠ': u'υ', u'ῡ': u'υ',
                       u'ῢ': u'υ', u'ΰ': u'υ', u'ῦ': u'υ', u'ῧ': u'υ', u'ύ': u'υ', u'ϋ': u'υ',
                       u'ὠ': u'ω', u'ὡ': u'ω', u'ὢ': u'ω', u'ὣ': u'ω', u'ὤ': u'ω', u'ὥ': u'ω',
                       u'ὦ': u'ω', u'ὧ': u'ω', u'ὼ': u'ω', u'ώ': u'ω', u'ᾠ': u'ω', u'ᾡ': u'ω',
                       u'ᾢ': u'ω', u'ᾣ': u'ω', u'ᾤ': u'ω', u'ᾥ': u'ω', u'ᾦ': u'ω', u'ᾧ': u'ω',
                       u'ῲ': u'ω', u'ῳ': u'ω', u'ῴ': u'ω', u'ῶ': u'ω', u'ῷ': u'ω', u'ώ': u'ω',
                       u'ῤ': u'ρ', u'ῥ': u'ρ'}


    @staticmethod
    def in_voc(sym, voc='en'):
        alphabet = TextProcessor.vocabulary[voc]
        return sym in alphabet

    @staticmethod
    def get_english_transliteration(text):
        return ''.join([TextProcessor.english_dict[s] for s in text])

    @staticmethod
    def merge_files(folder):
        """Merge all files in :folder to one 'all.txt' file
        
        Args:
            folder (str): Folder with .txt files to merge
        
        Returns:
            void:
        """
        output_name = "all.txt"
        output_file = folder + "/" + output_name
        with open(output_file, 'w') as outfile:
            for fname in os.listdir(folder):
                file_path = folder + "/" + fname
                if fname != output_name:
                    with open(file_path) as infile:
                        for line in infile:
                            outfile.write(line)

        print "Merged file path: " + output_file

    @staticmethod
    def generate_empty_dict(n, lang='en'):
        """Generate empty dictionary for N-grams
        
        Args:
            n (int): N number for N-grams
            lang (str, optional): Detect which language dictionary to use
        
        Returns:
            dict: Zero-counters language dictionary
        """
        alphabet = TextProcessor.vocabulary[lang]

        gen_n_grams = []
        if (n == 1): gen_n_grams = alphabet[:]

        if (n == 2):
            for i in alphabet:
                for j in alphabet:
                    res_str = i + j
                    gen_n_grams.append(res_str)

        if (n == 3):
            for i in alphabet:
                for j in alphabet:
                    for k in alphabet:
                        res_str = i + j + k
                        gen_n_grams.append(res_str)

        return dict((k,0) for k in gen_n_grams)

    @staticmethod
    def preprocess_folder(folder_path, lang='en', join=False, remove_diacritics=False):
        """Remove all unneccessary symbols from every .txt file in :folder_path
        
        Args:
            folder_path (str): Folder with .txt files path
            lang (str, optional): Language of texts in folder
            join (bool, optional): Merge all files after preprocessing to processed/all.txt
            remove_diacritics (bool, optional): Remove all diacritical marks from the text
        
        Returns:
            void: 
        """
        print "Try to preprocess documents in %s" % folder_path
        processed_path = folder_path + '/' + TextProcessor.processed_dir

        if not os.path.exists(processed_path):
            os.makedirs(processed_path)

        for text in os.listdir(folder_path):
            if text.endswith(".txt"):
                print ('\t---> %s/%s' % (folder_path, text))
                TextProcessor.preprocess_document(folder_path, text, lang, remove_diacritics)
        print "Files are successfully processed to " + processed_path

        if (join):
            TextProcessor.merge_files(processed_path)

    @staticmethod
    def preprocess_document(file_path, name, lang='en', remove_diacritics=False):
        """Remove all unneccessary symbols from .txt file
        
        Args:
            file_path (str): File folder
            name (str): File name
            lang (str, optional): File main language
            remove_diacritics (bool, optional): Remove all diacritical marks from the text
        
        Returns:
            void: 
        """
        origin_text_path = file_path + '/' + name
        content_origin = codecs.open(origin_text_path, encoding='utf-8')
        content_processed = ''.join(e for e in content_origin.read().lower() if TextProcessor.in_voc(e, voc=lang))
        content_origin.close()

        if remove_diacritics:
            content_processed = ''.join(e if e not in TextProcessor.diacritics_dict 
                else TextProcessor.diacritics_dict[e] for e in content_processed)

        processed_loc = file_path + '/' + TextProcessor.processed_dir
        processed_path = processed_loc + '/' + name

        if not os.path.exists(processed_loc):
            os.makedirs(processed_loc)
    
        processed_file = codecs.open(processed_path, 'w', encoding='utf-8')
        processed_file.write(content_processed)
        processed_file.close()


class TextUtils:

    @staticmethod
    def get_n_gram_dict(text, n, lang='en'):
        """Count all n-grams in text
        
        Args:
            text (str): Document text previously stored as string
            n (int): N in N-gram
            lang (str, optional): Detect which language dictionary to use
        
        Returns:
            dict: n-gram text counters
        """
        ngc = TextProcessor.generate_empty_dict(n, lang)

        text_dec = text.decode('utf-8')
        for i in xrange(len(text_dec) - n + 1):
            ng_curr = text_dec[i:i+n]
            ngc[ng_curr] += 1

        return ngc

    @staticmethod
    def combine_dicts(a, b, op=operator.add):
        return dict(a.items() + b.items() +
            [(k, op(a[k], b[k])) for k in set(b) & set(a)])

    @staticmethod
    def l1_distance(corpus, item):
        """ Calculate L1 distance between dictionaries
        
        Args:
            corpus (dict): First text (corpus) ordered dictionary
            item (dict): Second text ordered dictionary
        
        Returns:
            dist (float): L1 Distance
        """
        min_len = min(len(corpus), len(item))
        corpus_norm = corpus[:min_len] / np.sum(corpus[:min_len])
        item_norm = item[:min_len] / np.sum(item[:min_len])

        dist = 0
        for x, y in zip(corpus_norm, item_norm):
            dist += abs(x - y)
        return dist

    @staticmethod
    def get_normalized_dict(data):
        norm = data.copy()
        sum_values = sum(data.values())
        for key in norm:
            norm[key] = norm[key] / float(sum_values)

        return norm


    @staticmethod
    def get_ordered_dict(data, by='key', reverse=False):
        """Get ordered dictionary
        
        Args:
            data (dict): Input dictionary
            by (str, optional): 'key' or 'value'
            reverse (bool, optional): False is ascending order
        
        Returns:
            dict: Ordered dictionary
        
        Raises:
            ValueError: \"\by\" parameter must be 'key' or 'value'
        """
        if (by == 'key'):
            return sorted(data.iteritems(), key=operator.itemgetter(0), reverse=reverse)
        elif (by == 'value'):
            return sorted(data.iteritems(), key=operator.itemgetter(1), reverse=reverse)
        else:
            raise ValueError("\"by\" parameter must be 'key' or 'value'")


class HurstExponent:
    """Hurst Exponent values calculator
    
    Attributes:
        distances ([int]): Similar letter pair distances
        hurst_values ([float]): Result Hurst values
        sym (str): The letter which the calculations based on
        text (str): The input text in UTF-8 encoded formatted string (only alphabet letters)

    Functions:
        \"\"\" Claculate Hurst values for specifc window size
        calculate(window_size)
    """
    def __init__(self, text, sym):
        self.text = text[:]
        self.sym = sym
        self.distances = self.calculate_distances()
        self.hurst_values = []

    def calculate_distances(self):
        distances = []
        previous_index = -1
        cur_index = 0

        for i in xrange(len(self.text)):
            cur_sym = self.text[i]
            if (cur_sym == self.sym):
                cur_index = i
                if (previous_index > -1):
                    distances.append(cur_index - previous_index)
                    previous_index = i
                else:
                    previous_index = i
        return distances

    def calculate(self, window_size=10000):
        """Claculate Hurst values for specifc window size
        
        Args:
            window_size (int, optional): The size of text distances fragment. By default 10000
        
        Returns:
            void:
        """
        n_indexes = [i * window_size-1 for i in xrange(1,len(self.distances)) if i * window_size <= len(self.distances)]
        for n in n_indexes:
            self.hurst_values.append(self.calculate_hurst_n(window_size, n))

    def calculate_hurst_n(self, N, n):
        coef_sum = 0
        rescale_coef = np.mean([np.log(k / float(N)) ** 2 for k in range(100, N + 1, 100)])
        mean_ANL = self.mean_amplitude_noise_logarithm(N, n)

        for k in range(100, N + 1, 100):
            regression_coef = self.amplitude_noise_logarithm(n, k) - mean_ANL
            sample_len_log = 1 + np.log(k / float(N))
            coef_sum += regression_coef * sample_len_log
        return coef_sum / (float(N) / 100) / rescale_coef

    def mean_amplitude_noise_logarithm(self, N, n):
        return np.sum([self.amplitude_noise_logarithm(n, k) for k in range(100, N + 1, 100)]) / (float(N) / 100)

    def amplitude_noise_logarithm(self, n, k):
        return np.log(self.accumulated_deviation_mean(n, k) / float(np.sqrt(self.series_variance(n, k))))

    def accumulated_deviation_mean(self, n, k):
        current_sum = 0
        max_n = float(-np.inf)
        min_n = float(np.inf)
        fmi = self.float_mean_increment(n, k)

        for j in range(n - k + 1, n + 1):
            current_sum += self.distances[j] - fmi
            if (current_sum > max_n):
                max_n = current_sum
            if (current_sum < min_n):
                min_n = current_sum

        return max_n - min_n

    def series_variance(self, n, k):
        fmi = self.float_mean_increment(n, k)
        return np.sum([(self.distances[i] - fmi) ** 2 for i in range(n - k + 1, n + 1)]) / float(k)

    def float_mean_increment(self, n, k):
        return np.sum([self.distances[i] for i in range(n - k + 1, n + 1)]) / float(k)
