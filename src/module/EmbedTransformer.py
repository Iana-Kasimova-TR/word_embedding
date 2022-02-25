import pandas as pd
import numpy as np
import Utils
from nltk.tokenize import word_tokenize


class EmbedTransformer:
    def __init__(self, dataframe, token_dict, dim):
        self.dataframe = dataframe
        self.token_dict = token_dict
        self.dim = dim
        self.matrix_embed = self.init_embed()
        self.matrix_gradient = self.init_gradient()

    def init_embed(self):
        # Declaring rows
        N = len(self.token_dict.keys()) + 1
        # Declaring columns
        M = self.dim
        return pd.DataFrame([np.random.normal(0, 1, M)] * N)

    def init_gradient(self):
        N_G = len(self.token_dict.keys()) + 1
        M_G = self.dim
        return pd.DataFrame([np.array([1e-6 for i in range(M_G)])] * N_G)

    def get_average_vector(self, matrix, index_list):
        batch_vector = []
        for index in index_list:
            batch_vector.append(matrix.iloc[index])

        avg_vec = []
        for i in range(self.dim):
            sum_of_component = 0
            for j in range(len(index_list) - 1):
                sum_of_component = sum_of_component + batch_vector[j][i]
            avg_vec.append(sum_of_component / len(index_list))
        return avg_vec

    def calculate_triplet_loss(anchor, correct, wrong):
        loss = 1 - np.dot(np.asarray(anchor), np.asarray(correct)) + np.dot(np.asarray(correct), np.asarray(wrong))
        return np.maximum(0, loss)

    def calculate_gradient(anchor, correct, wrong):
        return ([item for item in [-c + w for (w, c) in zip(wrong, correct)]],
                [item for item in [z - a for (z, a) in zip(np.zeros(len(anchor)), anchor)]],
                [item for item in anchor])

    def update_gradient_matrix_adagrad(self, index_list, grad_vector):
        self.matrix_gradient.iloc[index_list] += np.square(grad_vector)

    def update_gradient_matrix_rmsprop(self, index_list, grad_vector, update_gamma):
        update_gamma * self.matrix_gradient.iloc[index_list] + (1 - update_gamma) * np.square(grad_vector)

    def update_embed_matrix(self, index_list, grad_vector, matrix_grad, learning_rate):
        self.matrix_embed.iloc[index_list] -= (learning_rate * np.asarray(grad_vector)) / np.sqrt(matrix_grad.iloc[index_list])


    def calculate_triplet(self, df_t, matrix_embed):
        matrix_title_tags = []
        indexes = []

        for i in range(df_t.shape[0] - 1):
            batch_anc_cor_uncor = []
            batch_indexes = []
            token_title = word_tokenize(df_t['clean_title'][i])
            index_list_anchor = Utils.get_indexes_of_tokens(token_title, self.token_dict)
            index_list_correct = Utils.get_indexes_of_tokens(df_t['clean_tags'][i], self.token_dict)
            # exclude title and tags, which doesn't contain frequent tokens at all
            if len(index_list_anchor) > 0 and len(index_list_correct) > 0:
                batch_anc_cor_uncor.append(self.get_average_vector(matrix_embed, index_list_anchor))
                batch_anc_cor_uncor.append(self.get_average_vector(matrix_embed, index_list_correct))
                batch_indexes.append(index_list_anchor)
                batch_indexes.append(index_list_correct)
                index_list_wrong, average_incorrect_vector = self.get_incorrect(i, df_t)
                batch_anc_cor_uncor.append(average_incorrect_vector)
                batch_indexes.append(index_list_wrong)

                indexes.append(np.asarray(batch_indexes))
                matrix_title_tags.append(batch_anc_cor_uncor)
        return matrix_title_tags, indexes



    def get_incorrect(self, i, df_t):
        k = self.get_inconsistient_index(i, df_t)
        index_list_wrong = Utils.get_indexes_of_tokens(df_t['clean_tags'][k])
        return zip(index_list_wrong, self.get_average_vector(self.matrix_embed, index_list_wrong))

    def get_inconsistient_index(self, i, df_t):
        k = self.get_index_not_out_of_bounds(i, df_t.shape[0])
        while len(Utils.get_indexes_of_tokens(df_t['clean_tags'][k])) == 0:
            k = self.get_index_not_out_of_bounds(k, df_t.shape[0])
        return k

    @staticmethod
    def get_index_not_out_of_bounds(k, shape):
        k = k + 1
        if k > (shape - 1):
            k = k - shape
        return k

    def calculate_map_metrica(self, matrix_title, matrix_tags, k):
        dot_matrix = matrix_title.dot(matrix_tags.transpose())
        return np.diagonal(dot_matrix)
