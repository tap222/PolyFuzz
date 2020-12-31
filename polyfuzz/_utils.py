import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import List
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity as scikit_cosine_similarity

try:
    from sparse_dot_topn import awesome_cossim_topn
    _HAVE_SPARSE_DOT = True
except ImportError:
    _HAVE_SPARSE_DOT = False


def cosine_similarity(from_vector: np.ndarray,
                      to_vector: np.ndarray,
                      from_list: List[str],
                      to_list: List[str],
                      nbest,
                      min_similarity: float = 0,
                      method: str = "sparse") -> pd.DataFrame:
    """ Calculate similarity between two matrices/vectors and return best matches

    Arguments:
        from_vector: the matrix or vector representing the embedded strings to map from
        to_vector: the matrix or vector representing the embedded strings to map to
        from_list: The list from which you want mappings
        to_list: The list where you want to map to
        min_similarity: The minimum similarity between strings, otherwise return 0 similarity
        method: The method/package for calculating the cosine similarity.
                Options: "sparse", "sklearn", "knn".
                Sparse is the fastest and most memory efficient but requires a
                package that might be difficult to install.
                Sklearn is a bit slower than sparse and requires significantly more memory as
                the distance matrix is not sparse
                Knn uses 1-nearest neighbor to extract the most similar strings
                it is significantly slower than both methods but requires little memory

    Returns:
        matches:  The best matches between the lists of strings


    Usage:

    Make sure to fill the `to_vector` and `from_vector` with vector representations
    of `to_list` and `from_list` respectively:

    ```python
    from polyfuzz.models import extract_best_matches
    indices, similarity = extract_best_matches(from_vector, to_vector, method="sparse")
    ```
    """
    if nbest != None:
        if int(nbest) >  len(to_list):
            raise ValueError('best choice must be less than to_list')
    else:
        nbest = int(1)
        
    # Slower but uses less memory
    if method == "knn":

        if from_list == to_list:
            knn = NearestNeighbors(n_neighbors=2, n_jobs=-1, metric='cosine').fit(to_vector)
            distances, indices = knn.kneighbors(from_vector)
            distances = distances[:, 1]
            indices = indices[:, 1]

        else:
            knn = NearestNeighbors(n_neighbors=1, n_jobs=-1, metric='cosine').fit(to_vector)
            distances, indices = knn.kneighbors(from_vector)

        similarity = [round(1 - distance, 3) for distance in distances.flatten()]

    # Fast, but does has some installation issues
    elif _HAVE_SPARSE_DOT and method == "sparse":
        if isinstance(to_vector, np.ndarray):
            to_vector = csr_matrix(to_vector)
        if isinstance(from_vector, np.ndarray):
            from_vector = csr_matrix(from_vector)

        # There is a bug with awesome_cossim_topn that when to_vector and from_vector
        # have the same shape, setting topn to 1 does not work. Apparently, you need
        # to it at least to 2 for it to work
        if int(nbest) <= 1:
            similarity_matrix = awesome_cossim_topn(from_vector, to_vector.T, 2, min_similarity)
        elif int(nbest) > 1:
            similarity_matrix = awesome_cossim_topn(from_vector, to_vector.T, nbest, min_similarity)
            
        if from_list == to_list:
            similarity_matrix = similarity_matrix.tolil()
            similarity_matrix.setdiag(0.)
            similarity_matrix = similarity_matrix.tocsr()
        
        if int(nbest) <= 1 and method == "sparse":
            indices = np.array(similarity_matrix.argmax(axis=1).T).flatten()
            similarity = similarity_matrix.max(axis=1).toarray().T.flatten()
        elif int(nbest) > 1 and method == "sparse":
            similarity = np.flip(np.take_along_axis(similarity_matrix.toarray(), np.argsort(similarity_matrix.toarray(), axis =1), axis=1) [:,-nbest:], axis=1)
            indices = np.flip(np.argsort(np.array(similarity_matrix.toarray()), axis =1)[:,-nbest:], axis=1)
            
    # Faster than knn and slower than sparse but uses more memory
    else:
        similarity_matrix = scikit_cosine_similarity(from_vector, to_vector)

        if from_list == to_list:
            np.fill_diagonal(similarity_matrix, 0)

        indices = similarity_matrix.argmax(axis=1)
        similarity = similarity_matrix.max(axis=1)

    # Convert results to df
    if int(nbest) <= 1:
        matches = [to_list[idx] for idx in indices.flatten()]
        matches = pd.DataFrame(np.vstack((from_list, matches, similarity)).T, columns=["From", "To", "Similarity"])
        matches.Similarity = matches.Similarity.astype(float)
        matches.loc[matches.Similarity < 0.001, "To"] = None
    else:
        matches = [np.array([to_list[idx] for idx in l]) for l in indices] ##In progress
        column = ["To"]
        for i in range(nbest - 1):
            column.append("BestMatch" + "__" + str(i+1))
        column.append("Similarity")
        for j in range(nbest - 1):
            column.append("Similarity" + "__" + str(j+1))
        matches = pd.concat([pd.DataFrame({'From' : from_list}), pd.DataFrame(np.hstack((matches, similarity)), columns= column)], axis =1)
        matches.Similarity = matches.Similarity.astype(float)
        matches.loc[matches.Similarity < 0.001, "To"] = None
        for i in range(nbest - 1):
            matches.loc[matches.Similarity < 0.001, "BestMatch" + "__" + str(i+1)] = None
        
    return matches
