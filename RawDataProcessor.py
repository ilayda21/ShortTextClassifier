import csv
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from scipy.sparse import coo_matrix
import scipy.sparse as sparse
import numpy as np
import string
import sys


class RawDataProcessor:
    if len(sys.argv) > 1:
        corpus_path = sys.argv[1]
        print("Corpus path: " + corpus_path)
    else:
        print("No corpus path is provided. Default path will be used: ./data/stck_data.csv")
        corpus_path = "./data/stck_data.csv"
    doc_number = 0
    document_term_matrix = None
    document_tag = set([])
    tag_dict = dict()
    vocab_size = 0

    def read_data(self):
        stop_words = set(stopwords.words('english'))
        vocab = dict()
        ps = PorterStemmer()
        with open(self.corpus_path, "r", encoding='utf-8') as corp_f:
            csv_reader = csv.reader(corp_f)
            # document
            row_count = 0
            # word
            column_count = 0
            temp_col = []
            temp_row = []
            for doc in csv_reader:
                tag = doc[0]
                self.document_tag.add(tag)
                self.tag_dict[row_count] = tag
                col_count = 0
                doc_str = doc[2].lower().translate(str.maketrans('', '', string.punctuation))
                doc_str = re.sub(r'\d+', '', doc_str)
                tokenized_words = word_tokenize(doc_str)
                for token in tokenized_words:
                    # stemmed_token = ps.stem(token)
                    stemmed_token = token
                    # if stemmed_token not in stop_words and not stemmed_token.isnumeric():
                    if stemmed_token not in stop_words:
                        # add index of the unique word
                        if stemmed_token not in vocab:
                            vocab[stemmed_token] = column_count
                            temp_col.append(column_count)
                            column_count = column_count + 1
                        else:
                            index = vocab[stemmed_token]
                            temp_col.append(index)
                        col_count = col_count + 1
                temp_row.append((row_count, col_count))
                row_count = row_count + 1
                if row_count % 10000 == 0:
                    print("document count: " + str(row_count))
        vocab_size = len(vocab)
        vocab.clear()

        tf_value = 1 + np.log10(10)
        data = np.full((len(temp_col)), tf_value)
        print("data is ready")
        row_temp = []
        # temp_row => (document_index, how many data exists on that row)
        for row_tuple in temp_row:
            for index in range(0, row_tuple[1]):
                row_temp.append(row_tuple[0])
        row = np.array(row_temp)
        print("row is ready")
        column = np.array(temp_col)
        print("column is ready")

        print("tf-idf")
        sparse_matrix = coo_matrix((data, (row, column)), shape=(row_count, vocab_size)).tocsr()
        transpose_sparse_matrix = sparse_matrix.transpose()

        self.doc_number = row_count
        doc_count = sparse_matrix.shape[0]
        idf = np.log10(doc_count / sparse.csr_matrix.getnnz(transpose_sparse_matrix, 1))
        self.document_term_matrix = sparse.csr_matrix.multiply(transpose_sparse_matrix,
                                                               np.array([idf]).transpose()).tocsr().transpose()
        print("sparse created")
