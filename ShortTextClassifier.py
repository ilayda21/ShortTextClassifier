from RawDataProcessor import RawDataProcessor
import random
from scipy.sparse import vstack
import scipy.sparse as sparse
import numpy as np

import sklearn
class ShortTextClassifier:

    def find_similarity_matrix(self, centroid_matrix_param, doc_matrix):
        return sparse.csr_matrix.dot(doc_matrix, centroid_matrix_param.transpose())

    def get_initial_centroid_matrix(self, num_of_clusters, document_number, doc_term_matrix):
        initial_centroids = []  # centroid id's
        init_cluster = dict()
        centroid_list = []
        centroid = None
        i = 0
        while i != num_of_clusters:
            rand_num = random.randint(0, document_number - 1)

            if rand_num not in initial_centroids:
                partial_centroid_coo_matrix = doc_term_matrix.getrow(rand_num).tocsr()
                initial_centroids.append(rand_num)
                try:
                    centroid = vstack([centroid, partial_centroid_coo_matrix])
                except:
                    centroid = partial_centroid_coo_matrix
                init_cluster[i] = [rand_num]
                centroid_list.append(rand_num)
                i = i + 1
        return centroid, init_cluster, centroid_list

    # find the mean of the clusters with the docs which are in that cluster (cluster_ids)
    def calculate_centroids(self, doc_word_matrix, clusters_):
        centroid_mat = None
        # stack all features of the cluster in a sparse matrix
        for cluster_item in clusters_:
            cluster_mean_matrix = sparse.csr_matrix(doc_word_matrix[clusters_[cluster_item]].mean(0))
            try:
                centroid_mat = vstack([centroid_mat, cluster_mean_matrix])
            except:
                centroid_mat = cluster_mean_matrix
        return centroid_mat

    def set_initial_clusters(self, cluster_dict, centroid_list, similarity_matrix):
        # add them selves for initial clustering
        print("initial clustering")
        a, cols = np.unravel_index(sparse.csr_matrix.argmax(similarity_matrix, -1), similarity_matrix.shape)
        for i in range(cols.shape[0]):
            ind = cols[i][0]
            if i not in centroid_list:
                cluster_dict[ind].append(i)
        print("ok")
        return cluster_dict

    def set_clusters(self, cluster_dict, similarity_matrix):
        # find position and value of the max value in similarity matrix
        rows, cols = np.unravel_index(sparse.csr_matrix.argmax(similarity_matrix, -1), similarity_matrix.shape)
        np_cols = np.array(cols)
        i = 0
        for element in np.nditer(np_cols):
            try:
                cluster_dict[+element].append(i)
            except:
                cluster_dict[+element] = [i]
            i = i + 1
        return cluster_dict

    def check_equality(self, dict1, dict2):
        eq = False
        for elem in dict1:
            try:
                eq = np.array_equal(np.array([dict1[elem]]), np.array([dict2[elem]]))
            except:
                return False
        return eq

    def start_k_means(self):
        processor = RawDataProcessor()
        processor.read_data()
        cluster_number = len(processor.document_tag)
        centroid_matrix, initial_cluster, centroid_li = self.get_initial_centroid_matrix(cluster_number,
                                                                                         processor.doc_number,
                                                                                         processor.document_term_matrix)
        similarity = self.find_similarity_matrix(centroid_matrix, processor.document_term_matrix)
        clusters = self.set_initial_clusters(initial_cluster, centroid_li, similarity)
        print("initial clusters are ready")

        k = 0
        prev_cluster = dict()
        # print("start centroid calculation")
        while self.check_equality(clusters, prev_cluster) is False:
            print("iteration no: " + str(k))
            k = k + 1
            prev_cluster = dict(clusters)
            centroid_matrix = self.calculate_centroids(processor.document_term_matrix, clusters)
            print("centroids found")
            similarity = self.find_similarity_matrix(centroid_matrix, processor.document_term_matrix)
            print("distance calculated")
            # reset cluster
            initial_cluster = dict()
            clusters = self.set_clusters(initial_cluster, similarity)

        # check purity
        print("clustering is done purity will be calculated:")
        doc_tag = processor.tag_dict
        avg = 0
        for cluster in clusters:
            dict_for_tags = dict()
            # create count dict for the given cluster
            for doc in clusters[cluster]:
                tag = doc_tag[doc]
                if tag in dict_for_tags:
                    dict_for_tags[tag] = dict_for_tags[tag] + 1
                else:
                    dict_for_tags[tag] = 1
            max_val = -1
            max_tag = "x"
            total_element = 0
            for cluster_tag in dict_for_tags:
                count = dict_for_tags[cluster_tag]
                if max_val < count:
                    max_val = count
                    max_tag = cluster_tag
                total_element = total_element + count
            print("-----PURITY-------")
            print(max_tag)
            print(max_val / total_element)
            print("--------------")
            avg = avg + (max_val / total_element)

        print("********AVERGE PURITY*********")
        print(avg / len(clusters))


short_text_classifier = ShortTextClassifier()
short_text_classifier.start_k_means()
