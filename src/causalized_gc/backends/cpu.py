"""Conventional CPU implementation of causalised Granger causality."""

import numpy as np
from sklearn.linear_model import LinearRegression
from numba import jit, njit
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import concurrent.futures
import logging
import multiprocessing
from itertools import permutations
from dataclasses import dataclass

@jit(nopython=True)
def perm_test(x, y, n_perm):
    """
    Computes p_value of correlation of two iid generated variables.
    Args:
        x: (array like, vector): Realisation of a variable x
        y: (array like, vector): Realisation of a variable y
    Returns:
        p_value
    """
    count = 0
    corr_1 = np.corrcoef(x, y)[1, 0]
    x_copy = x.copy()
    for j in range(n_perm):
        # x_ = np.roll(x_copy, np.random.randint(30, x_copy.size), 0)
        l = np.random.randint(30, x.size)
        x_ = np.hstack((x_copy[l:], x_copy[:l]))
        corr_2 = np.corrcoef(x_, y)[1, 0]
        if np.abs(corr_2) >= np.abs(corr_1):
            count += 1
    return count / n_perm


#@jit(nopython=True)
def prep_data(X, nn):
    if nn == None or nn == 0:
        X_ = X
        return X_
    else:
        X_ = X[:,(nn):]
        for i in range(nn):
            idx1, idx2 = nn-1-i,-i-1
            X_ = np.r_[X_, X[:,idx1:idx2]]
        return X_

# @jit(nopython=True)
def residual(x: np.ndarray, z: np.ndarray) -> np.ndarray:  # private
    """
    Computes the residuals of a variable x by regressing a
    conditioning set z out of it

    Args:
        x: Variable in question to regress out the conditioning set
        z: Conditioning set based on whether c-GC or fc-GC was used.

    return:
        the residual of x conditioned on z
    """

    model = LinearRegression(fit_intercept=True)
    model.fit(z.T, x)

    coefs = model.coef_
    intercept = model.intercept_

    return x - np.dot(coefs, z) - intercept

@dataclass
class GcStar:
    """
    Implementation of Granger causality from causal Bayesian network perspective.
    Methods:
        fit(self, data: np.ndarray)
        get_connectivity_matrix(self)
        plot_conn_mat_on_topography(self)
    """

    corr_: Optional[np.ndarray]
    pVal_corr_: Optional[np.ndarray]
    inv_corr_: Optional[np.ndarray]
    pVal_inv_corr_: Optional[np.ndarray]

    def __init__(self, n_perm: int, n_pasts: int, n_lags: int, temporal: bool, method: str):
        """

        Args:
            n_perm: number of permutations (default = 1000)
            n_pasts: number of past states
            n_lags: maximum allowable lags
            temporal: defines if data is time series or iid
        """
        # logging.basicConfig(level=logging.INFO)
        self.logger = None  # Initialize logger when needed
        # self.logger = logging.getLogger(__name__)

        self.n_neur = None
        self.n_perm = n_perm
        self.n_pasts = n_pasts
        self.n_lags = n_lags
        self.method = method
        self.temporal = temporal

        self.f_s = None
        self.f_c = None
        self.M = None

        self.data = None
        self.shifted_data = None
        self.topography = None

        self.corr_ = None
        self.pVal_corr_ = None
        self.inv_corr_ = None
        self.pVal_inv_corr_ = None

    def is_time_series(self) -> bool:
        """
            Collects a boolean (True or False)
            True: specifying that data is time series e.g. calcium imaging data
            False: independent and identically distributed (iid) data
        """
        return self.temporal

    def get_number_of_lags(self) -> int:
        """
        Collects an integer valued maximum lag desired for use in the analysis
        """
        return self.n_lags

    def get_number_of_perms(self) -> int:
        """
            Collects an integer valued number of permutations to be used
            for all p-value computations
        """
        return self.n_perm

    def get_number_of_pasts(self) -> int:
        """
        Collects an integer valued maximum lag desired for use in the analysis
        """
        return self.n_pasts

    def get_method(self) -> str:
        """
        Defines which conditioning set method to use
        """
        return self.method

    def _configure_logging(self, verbose):
        log_level = logging.WARNING  # Default to show only warnings and errors

        if verbose == 1:
            log_level = logging.INFO  # Show informational logs

        elif verbose == 2:
            log_level = logging.DEBUG  # Show detailed debug logs

        logging.basicConfig(level=log_level, force=True)

    def shift_data(self, arr: np.ndarray) -> np.ndarray:
        """
        Creates shifted versions of the original data based on the number of pasts (n_past) required.
        This is done by stacking sliding windows of columns from data based on n_past.
        Concatenate the shifted arrays and return the new data.

        Examples: Given original data as X,
            trimmed_arr = __shifted_data(X, n_past = n)
            with trimmed_arr = {X_{t}, X_{t-1}, ..., X_{t-n}}^T
        Args:
            arr: original data recorded or obtained from experiments
            n_pasts: number of pasts defining the number of shifts

        Returns:
            Shifted data.
        """

        # self.data = arr.copy()
        self.n_neur = arr.shape[0]
        if self.n_pasts == 0:
            return arr

        trimmed_arr = arr[:, self.n_pasts:]
        for i in range(self.n_pasts):
            idx1 = self.n_pasts - 1 - i
            idx2 = -i - 1

            trimmed_arr = np.r_[trimmed_arr, arr[:, idx1:idx2]]
        return trimmed_arr

    def get_number_of_neurons(self) -> int:  # private
        """
        Computes the number of neurons/variables from the shape
        of the shifted data obtained from calling
        self.__shift_data() on the original data

        Args:
            trimmed_arr: shifted data from self.__shift_data()

        Returns:
            number of variables/neurons in data
        """
        self.n_neur = self.shifted_data.shape[0] // (self.n_pasts + 1)
        return self.n_neur

    def correlation_func(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the unconditional dependence of any variable pair
        using Pearson's correlation as a dependence metric

        Args:
            X: data from __shift_data()

        Returns:
            corr[:, :n]: np.ndarray shape [trimmed_arr.shape[0], self.n_neur]
                Correlation matrix of the data, but done on the shifted data.
                However, we select the portion that is most relevant for us
                by slicing the matrix into the required shape.

            pVal_corr: np.ndarray with shape as the correlation matrix.
                contains the p-values of individual elements in
                the correlation matrix
        """
        X_copy = X.copy()
        self.n_neur = X_copy.shape[0]
        data = self.shift_data(X_copy)
        n = data.shape[0]

        # compute correlation and p-values matrices of shifted data
        corr = np.abs(np.corrcoef(self.shifted_data))

        # Initialize p-value matrix of correlation
        pVal_corr = np.zeros((n, self.n_neur))

        total_steps = n * self.n_neur
        current_step = 0

        # compute and populate p-value matrix
        for i in range(n):
            for j in range(self.n_neur):
                pVal_corr[i, j] = perm_test(data[i, :], data[j, :], self.n_perm)

                current_step += 1
                completion_percentage = (current_step / total_steps) * 100
                self.logger.info(f"Step {current_step}/{total_steps} "
                                 f"({completion_percentage:.2f}% complete)")

        return corr[:, :self.n_neur], pVal_corr

    def get_conditioning_set(self, X, i, j) -> np.ndarray:
        """
        Identifies the variables in the conditioning set for Granger causality implementation.

        Args:
            i: (int) index of the cause variable
            j: (int) index of the effect variable

        Returns:
            np.ndarray: The conditioning set containing relevant variables.
        """
        data = X.copy()
        self.shifted_data = self.shift_data(data)
        i_ = i % self.n_neur

        # Exclude variables 'i', 'j' and futures of 'i' from the shifted_data
        x_idx = [i_ + a * self.n_neur for a in range(i // self.n_neur)]

        # Exclude variables 'i' and 'k * n' from the shifted_data
        z = np.delete(self.shifted_data, np.r_[np.array(x_idx).astype(int), [i, j]], axis=0)

        return z



    def inv_correlation_func(self, X: np.ndarray,
                            method: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the conditional dependency of any variable pair with Pearson's
        correlation as the dependence metric.
        It computes the residual of any pair in question with the
        conditioning set of choice (c-GC or fc-GC style).

        Args:
            X (np.ndarray): Time series data obtained with the
            shape [n_neur x Time]

            method: str. Defines which conditioning set to use.
                Takes arguments "cgc" or "fcgc"

                cgc: when conventional Granger causality conditioning set
                is preferred

                fcgc: when full conditioning set including future states
                is preferred for use.


        Returns:
            inv_corr: np.ndarray shape [trimmed_arr.shape[0], self.n_neur]
                Correlation matrix of the data but done on the shifted data.
                However, we select the portion that is most relevant for us
                by slicing the matrix into the required shape.

            pVal_inv_corr: np.ndarray with shape as the correlation matrix
                contains the p-values of individual elements in the
                correlation matrix
        """
        self.n_neur = X.copy().shape[0]
        data = self.shift_data(X.copy())
        n = data.shape[0]

        # Initialize matrices for inverse correlation and their p-values
        inv_corr = np.zeros((n, self.n_neur))
        pVal_inv_corr = np.zeros((n, self.n_neur))

        total_steps = n * self.n_neur
        current_step = 0

        # Compute inverse correlation and p-values
        for i in range(n):
            for j in range(self.n_neur):
                x = data[i]
                y = data[j]
                self.data = X.copy()

                # Obtain conditioning set via the 'method' argument
                if self.method == 'fcgc':
                    z = np.delete(data.copy(), [i, j], axis=0)
                else:
                    z = self.get_conditioning_set(self.data, i, j)

                # compute residuals for both cause and effect
                x_res, y_res  = residual(x, z), residual(y, z)

                # Check the dependence of the two residuals and the p-value
                inv_corr[i, j] = np.abs(np.corrcoef(x_res, y_res)[1, 0])
                pVal_inv_corr[i, j] = perm_test(x_res, y_res, self.n_perm)

                current_step += 1
                completion_percentage = (current_step / total_steps) * 100
                self.logger.info(f"Step {current_step}/{total_steps} "
                                f"({completion_percentage:.2f}% complete)")

        return inv_corr, pVal_inv_corr

    def fit(self, X: np.ndarray, verbose=1):
        """
        Fits c-gc to data

        Args:
            X: array-like of shape (n_neur, T)
                where `n_neur` is the number of neurons or variables
                and `T` is the time or number of samples

            method: str. Defines which conditioning set to use.
                Takes arguments "cgc" or "fcgc"

                cgc: when conventional Granger causality conditioning set
                is preferred

                fcgc: when full conditioning set with future states,
                is preferred for use.


        Returns:
            self: object
                Returns the instance itself
        """
        # Make a copy of the input data
        self.data = X.copy()

        # Shift the data
        self.shifted_data = self.shift_data(self.data)

        # Set up logging
        self._configure_logging(verbose)
        self.logger = logging.getLogger(__name__)

        # Create a ThreadPoolExecutor to parallelize the tasks
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the correlation_func task
            self.logger.info("Starting correlation_func")
            corr_future = executor.submit(self.correlation_func, X)

            # Submit the inv_correlation_func task
            self.logger.info("Starting inv_correlation_func")
            inv_corr_future = executor.submit(self.inv_correlation_func,
                                            X, method=self.method)

            # Wait for both tasks to complete and get their results
            self.logger.info("Waiting for correlation_func and "
                            "inv_correlation_func to complete")
            corr_, pVal_corr_ = corr_future.result()
            inv_corr_, pVal_inv_corr_ = inv_corr_future.result()

        # Assign the results
        self.corr_, self.pVal_corr_ = corr_, pVal_corr_
        self.inv_corr_, self.pVal_inv_corr_ = inv_corr_, pVal_inv_corr_

        return self

    def get_connectivity_matrix(self,
                                simulation: bool,
                                alpha: float, beta: float)  -> np.ndarray:
        """
        Computes the weighted connectivity matrix from significant
        conditional and unconditional links based on the maximum
        lag allowed for the analysis.
            >If the connections strength are not useful for
            analysis, the connectivity matrix can be binarized

        Args:
            alpha: float, Significance level for unconditional
                    dependence (default value 0.01)

            beta: float, Significance level for conditional
                    dependence (default value 0.001)

            simulation: bool, Determines if analysis is simulation or not

        Returns:
            Weighted Connectivity matrix of shape [n_neur, n_neur]
        """
        # compute significant correlation and inverse correlation matrices
        sig_corr = np.multiply(self.corr_, self.pVal_corr_ <= alpha)
        sig_inv = np.multiply(self.inv_corr_, self.pVal_inv_corr_ <= beta)

        # extended connectivity matrix
        inferred = np.logical_and(sig_corr, sig_inv)

        b = 0
        all_ = []
        n_neur = inferred.shape[1]

        # merging results for the GC order used
        for a in range(self.n_pasts + 1):
            all_.append(inferred[a * n_neur:(a + 1) * n_neur,
                        b * n_neur:(b + 1) * n_neur])

        if simulation:
            self.conn_mat = all_[1]
            if self.n_lags > 1:
                for i in range(2, self.n_lags + 1):
                    self.conn_mat = np.logical_or(self.conn_mat, all_[i])
        elif self.n_lags == 1:
            self.conn_mat = np.logical_or(all_[0], all_[1])
        elif self.n_lags > 1:
            self.conn_mat = all_[0]
            for i in range(1, self.n_lags + 1):
                self.conn_mat = np.logical_or(self.conn_mat, all_[i])

        # multiplied with correlation matrix to return weighted connectivity
        # matrix (weights depict the connection strengths)
        self.conn_mat = np.multiply(self.corr_[:n_neur, :], self.conn_mat)



        return self.conn_mat

###############################################################
### computing metrics
###############################################################
    def compute_confusion_matrix(self, A: np.ndarray, simulation: bool):
        """
        Function to compute the performance of the algorithm.
        Computes the confusion matrix by comparing the ground truth
        to the inferred connectivity matrix.
        Args:
            A: Ground truth connectivity matrix
            simulation (bool): Takes "True" if data is simulated with
                ground truth A and "False" otherwise.

        Returns:
            Confusion matrix with the form np.array([[TP, FP],
                                                    [FN, TN]])
        """
        if simulation:
            A = A.T
        TP = np.sum(np.logical_and(A != 0, self.conn_mat != 0))
        FN = np.sum(np.logical_and(A != 0, self.conn_mat == 0))
        FP = np.sum(np.logical_and(A == 0, self.conn_mat != 0))
        TN = np.sum(np.logical_and(A == 0, self.conn_mat == 0))
        self.confusion_matrix = np.array([[TP, FP],
                                        [FN, TN]])
        return self.confusion_matrix

    def all_metrics(self):
        confusion_matrix = self.confusion_matrix.flatten()
        accuracy = (confusion_matrix[0] + confusion_matrix[3])/(np.sum(confusion_matrix))
        precision = confusion_matrix[0]/(confusion_matrix[0]+confusion_matrix[1])
        recall = confusion_matrix[0]/(confusion_matrix[0]+confusion_matrix[2])
        FPR = confusion_matrix[1]/(confusion_matrix[1]+confusion_matrix[3])

        # others
        specificity = confusion_matrix[3]/(confusion_matrix[3]+confusion_matrix[1])
        BA = (specificity + recall)/2
        F1 = 2 * (precision * recall) / (precision + recall)
        return np.array([accuracy, precision, recall, FPR, BA, F1])

    def compute_shd_sid(self, A: np.ndarray,
                        inf: np.ndarray, simulation: bool) -> np.ndarray:
        try:
            from cdt.metrics import SHD
        except ImportError as exc:
            raise ImportError(
                "cdt.metrics is required for SHD computations."
            ) from exc

        if simulation:
            A = A.T
        self.shd_ = SHD(target=A, pred=inf, double_for_anticausal=False)
        # TODO: To fix R-package which won't allow SID computations.
        # self.sid_ = SID(target=A.T, pred=inf)
        return self.shd_ # , self.sid_

    ###########################################################################

    ### RISING FLANKS STARTS HERE!!!
    ###########################################################################

    def __perm_test_(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes p_value of correlation of two iid generated variables.
        Args:
        x: (array like, vector): Realisation of a variable x
        y: (array like, vector): Realisation of a variable y

        Returns:
            p_value
        """
        count, corr_1 = 0, np.corrcoef(x, y)[1, 0]
        val = np.random.randint(5, len(x) - 8, self.n_perm)

        for j in range(self.n_perm):
            x_copy = np.copy(x)
            x_ = np.roll(x_copy, val[j])
            corr_2 = np.corrcoef(x_, y)[1, 0]

            if np.abs(corr_2) >= np.abs(corr_1):
                count += 1

        return count/self.n_perm

    def lp_filt(self) -> np.ndarray:
        amp = np.ones(self.M)
        amp[int(self.f_c * self.M / self.f_s):-int(self.f_c *
                                                self.M / self.f_s)] = 0
        phase = np.zeros(self.M)
        H_f = amp * np.exp(1j * phase)
        h_n = np.fft.fftshift(np.real(np.fft.ifft(H_f)))
        return h_n

    def get_rising_flanks(self, arr, f_c: float, f_s: float, M: int) -> Tuple[np.ndarray, np.ndarray]:
        self.f_c = f_c
        self.f_s = f_s
        self.M = M
        h_n = self.lp_filt()
        conv = np.convolve(arr, h_n, 'same')
        j = np.diff(conv) > np.mean(np.diff(conv) > 0.5) / 10  # /5
        idx = np.where(j > 0)[0]
        return arr[idx], idx  # conv[idx] is the squashed vector  # arr ==> conv

    def fit_rising(self, X: np.ndarray, idx: np.ndarray, verbose=1):
        """
        Fits c-gc for rising flanks to data

        Args:
            data: array-like of shape (n_neur, T)
                where `n_neur` is the number of neurons or variables
                and `T` is the time or number of samples

        Returns:
            self: object
                Returns the instance itself
        """
        self.n_neur = X.shape[0]
        corr = np.zeros((len(idx) * self.n_pasts, len(idx)))
        pVal_corr = np.zeros((len(idx) * self.n_pasts, len(idx)))
        inv_corr = np.zeros_like(corr)
        pVal_inv_corr = np.zeros_like(corr)
        for i in tqdm(range(len(idx) * self.n_pasts)):
            for j in range(len(idx)):
                if i < X.shape[0]:
                    same_idx = sorted(set(idx[i]).intersection(idx[j]))  # combine()
                else:
                    i_ = i % len(idx)
                    same_idx = sorted(set(idx[i_]).intersection(idx[j]))  # combine()
                if len(same_idx) > 0.25 * len(same_idx):
                    dat = self.shift_data(X[:, same_idx])
                    corr[i, j] = np.abs(np.corrcoef(dat[i], dat[j])[1, 0])
                    pVal_corr[i, j] = self.__perm_test_(dat[i], dat[j])

                    x = dat[i]
                    y = dat[j]
                    self.data = X.copy()[:, same_idx]
                    z = self.get_conditioning_set(self.data, i, j)

                    x_ = residual(x, z)
                    y_ = residual(y, z)

                    inv_corr[i, j] = np.abs(np.corrcoef(x_, y_)[1, 0])
                    pVal_inv_corr[i, j] = self.__perm_test_(x_, y_)

                else:
                    corr[i, j], pVal_corr[i, j] = 0, 1
                    inv_corr[i, j], pVal_inv_corr[i, j] = 0, 1

        self.corr_, self.pVal_corr_ = corr, pVal_corr
        self.inv_corr_, self.pVal_inv_corr_ = inv_corr, pVal_inv_corr

        return self

    ###############################################################################

    def plot_extended_connectivity_matrix(self,
                                        alpha: float = 0.01,
                                        beta: float = 0.001):
        """
        Plots connectivity matrix inferred into different matrices
        of corresponding pasts

        Args:
            inferred: (array-like: matrix [# of variables * n_pasts X number
                        of variables]) connectivity matrix inferred
            n_past: (int) number of pasts used in analysis

        Returns: Plotted connectivity matrices corresponding to
                number of pasts
        """
        nn = self.n_pasts + 1
        fig, axs = plt.subplots(1, nn, figsize=(3.5 * nn, 3.5))
        b = 0
        sig_corr = np.multiply(self.corr_, self.pVal_corr_ <= alpha)
        sig_inv = np.multiply(self.inv_corr_, self.pVal_inv_corr_ <= beta)

        # extended connectivity matrix
        inferred = np.logical_and(sig_corr, sig_inv)

        for a in range(nn):
            jj = inferred[a * self.n_neur:(a + 1) * self.n_neur,
                 b * self.n_neur:(b + 1) * self.n_neur]
            axs[a].imshow(jj)
            axs[a].axis('off')
        plt.tight_layout()

def compare_with_GT(A: np.ndarray,
                        inf: np.ndarray,
                        simulation: bool) -> np.ndarray:
    """
    This function helps to create the color coding for the
    confusion matrix. TP (yellow), FP (green), and so on.
    Args:
        A (np.ndarray):
        inf (np.ndarray):
        simulation (bool):

    Returns:
        Color-coded inferred connectivity matrix with respect
        TP, FP, TN, and FN.
    """
    if simulation:
        A = A.T
    return (40 * np.logical_and(A != 0, inf != 0) +
            30 * np.logical_and(A == 0, inf != 0) +
            20 * np.logical_and(A != 0, inf == 0) +
            10 * np.logical_and(A == 0, inf == 0))
