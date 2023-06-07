import numpy as np
from sklearn.neural_network import MLPClassifier


def good_log(x):
    x_log = np.log(x, where=(x != 0))
    x_log[np.where(x == 0)] = -(10.0**8)
    return x_log


def lse(x, axis=None, keepdims=False):
    x_max = np.max(x, axis=axis, keepdims=keepdims)
    x_diff = x - x_max
    sumexp = np.exp(x_diff).sum(axis=axis, keepdims=keepdims)
    return (x_max + np.log(sumexp))


def log_gaussian(o, mu, r):
    compute = (- 0.5 * good_log(r) - np.divide(
        np.square(o - mu), 2 * r) - 0.5 * np.log(2 * np.pi)).sum()
    return compute


def forward(pi, a, o, mu, r):
    # pi is initial probability over states, a is transition matrix
    T = o.shape[0]
    J = mu.shape[0]
    log_alpha = np.zeros((T, J))

    for j in range(J):
        log_alpha[0][j] = good_log(pi)[j] + log_gaussian(o[0], mu[j], r[j])

    for t in range(1, T):
        for j in range(J):
            log_alpha[t, j] = log_gaussian(
                o[t], mu[j], r[j]) + lse(good_log(a[:, j].T) + log_alpha[t - 1])

    return log_alpha


def context_expand(data):
    T = data.shape[0]
    data_1 = np.copy(data[0])
    data_T = np.copy(data[-1])
    for i in range(3):
        data = np.insert(data, 0, data_1, axis=0)
        data = np.insert(data, -1, data_T, axis=0)
    expand_data = np.zeros((T, 7 * data.shape[1]))
    for t in range(3, T + 3):
        np.concatenate((data[t - 3], data[t - 2], data[t - 1], data[t], data[t + 1], data[t + 2], data[t + 3]),
                       out=expand_data[t - 3])
    return expand_data


class SingleGaussian():
    def __init__(self):
        # Basic class variable initialized, feel free to add more (Use from project1)
        self.dim = None
        self.mu = None
        self.r = None

    def train(self, data):
        data = np.vstack(data)
        self.mu = np.mean(data, axis=0)
        self.r = np.mean(np.square(np.subtract(data, self.mu)), axis=0)

    # Computes the log-likelihood probability for the Gaussian model fitted to the data
    def loglike(self, data_mat):
        ll = 0
        for each_line in data_mat:
            ll += log_gaussian(each_line, self.mu, self.r)
        return ll


class HMM():
    def __init__(self, sg_model, nstate):
        # Basic class variable initialized, feel free to add more
        # nstate

        self.pi = np.zeros(nstate)
        self.pi[0] = 1
        self.mu = np.tile(sg_model.mu, (nstate, 1))
        self.r = np.tile(sg_model.r, (nstate, 1))
        self.nstate = nstate

    # Initialize the HMM for each voice, and equally divide and align the HMM state.
    def initStates(self, data):
        self.states = []
        for data_s in data:
            T = data_s.shape[0]
            state_seq = np.array(
                [self.nstate * t / T for t in range(T)], dtype=int)
            self.states.append(state_seq)

    # The HMM aligns the input sequence
    def get_state_seq(self, data):
        T = data.shape[0]
        J = self.nstate
        s_hat = np.zeros(T, dtype=int)
        log_a = good_log(self.a)
        log_delta = np.zeros((T, J))
        log_delta[0] = good_log(self.pi)
        psi = np.zeros((T, J))

        # initialize
        for j in range(J):
            log_delta[0, j] += log_gaussian(data[0], self.mu[j], self.r[j])

        # forward algorithm
        for t in range(1, T):
            for j in range(J):
                temp = np.zeros(J)
                for i in range(J):
                    temp[i] = log_delta[t - 1, i] + log_a[i, j] + \
                        log_gaussian(data[t], self.mu[j], self.r[j])
                log_delta[t, j] = np.max(temp)
                psi[t, j] = np.argmax(log_delta[t - 1] + log_a[:, j])

        s_hat[T - 1] = np.argmax(log_delta[T - 1])

        # backtracking state sequence
        for t in reversed(range(T - 1)):
            s_hat[t] = psi[t + 1, s_hat[t + 1]]

        return s_hat

    def viterbi(self, data):
        for u, data_u in enumerate(data):
            s_hat = self.get_state_seq(data_u)
            self.states[u] = s_hat

    # M_step update parameters
    def m_step(self, data):
        self.a = np.zeros((self.nstate, self.nstate))
        gamma_0 = np.zeros(self.nstate)

        gamma_1 = np.zeros((self.nstate, data[0].shape[1]))
        gamma_2 = np.zeros((self.nstate, data[0].shape[1]))

        for s in range(len(data)):
            T = data[s].shape[0]

            # state_seq is a list of states with length t
            state_seq = self.states[s]
            # gamma is emission_matrix
            gamma = np.zeros((T, self.nstate))

            # Statistical state transitions in the state sequence, gamma records the state path
            for t, j in enumerate(state_seq[:-1]):
                self.a[j, state_seq[t + 1]] += 1

            for t, j in enumerate(state_seq):
                gamma[t, j] = 1

            # Update the number of states gamma_0
            gamma_0 += np.sum(gamma, axis=0)

            # gGamma_1 is the sum of feature data in each hidden state, and gamma_2 is the sum of squares of feature data in each hidden state. For subsequent update of Gaussian parameters
            for t, j in enumerate(state_seq):
                gamma_1[j] += data[s][t]
                gamma_2[j] += np.square(data[s][t])
        # Update the transition matrix and the Gaussian mean variance of each hidden state
        for j in range(self.nstate):
            self.a[j] /= np.sum(self.a[j])
            self.mu[j] = gamma_1[j] / gamma_0[j]
            self.r[j] = (gamma_2[j] - np.multiply(gamma_0[j],
                         self.mu[j] ** 2)) / gamma_0[j]

    # Perform iter iterations of the EM algorithm

    def train(self, data, iter):
        if iter == 0:
            self.initStates(data)
        self.m_step(data)
        # renew states
        self.viterbi(data)

    # The forward algorithm calculates the likelihood probability of the HMM fitting data. The digit parameter is not used in this method in order to be unified with the loglike method of the DNNHMM model
    def loglike(self, data, digit=None):
        T = data.shape[0]
        log_alpha_t = forward(self.pi, self.a, data, self.mu, self.r)[-1]
        # log_sum所有状态概率
        ll = lse(log_alpha_t)
        return ll

# MLP+HMM


class HMMMLP():
    def __init__(self, mlp, hmm_model, S, uniq_state_dict):
        """
         mlp: multilayer perceptron as dnn
         hmm_model: trained hmm model
         S: The state sequence of all data is spliced on the x-axis
         uniq_state_dict: key: (digit, state) value: state_index
        """
        self.mlp = mlp
        self.hmm = hmm_model
        self.log_prior = self.com_log_prior(S)
        self.uniq_state_dict = uniq_state_dict

    # Calculate HMM hidden state prior probability based on alignment
    def com_log_prior(self, s):
        states, counts = np.unique(s, return_counts=True)
        p = np.zeros(len(states))
        for s, c in zip(states, counts):
            p[s] = c
        p_dis = p / np.sum(p)
        return good_log(p_dis)

    def forward_dnn(self, data, digit):
        T = data.shape[0]
        J = self.hmm.nstate

        # After expanding the left and right three frames of the voice, perform -3 to 3 7-frame context feature splicing
        o_expand = context_expand(data)

        # Neural Network Forward Propagation
        mlp_ll = self.mlp.predict_log_proba(o_expand)
        log_alpha = np.zeros((T, J))
        log_alpha[0] = good_log(self.hmm.pi)

        """
         HMM forward algorithm, note that it is different from the traditional Gaussian model to get P(observation|state),
         dnn gets P(state|observation), Bayesian conversion P(o|s) = P(s|o)*p(o)|p(s), where p(o) can be ignored.
        """
        for j in range(J):
            log_alpha[0] += np.array(mlp_ll[0][self.uniq_state_dict[(digit, j)]] - self.log_prior[
                self.uniq_state_dict[(digit, j)]])
        for t in range(1, T):
            for j in range(J):
                tmp = mlp_ll[t][self.uniq_state_dict[(
                    digit, j)]] - self.log_prior[self.uniq_state_dict[(digit, j)]]
                log_alpha[t, j] = tmp + \
                    lse(good_log(self.hmm.a[:, j].T) + log_alpha[t - 1])

        return log_alpha

    def loglike(self, data, digit):
        log_alpha_t = self.forward_dnn(data, digit)[-1]
        ll = lse(log_alpha_t)

        return ll


def sg_train(digits, train_data):
    model = {}
    for digit in digits:
        model[digit] = SingleGaussian()

    for digit in digits:
        data = train_data[digit]
        print("process %d data for digit %s" % (len(data), digit))
        model[digit].train(data)

    return model


def hmm_train(digits, train_data, sg_model, nstate, niter):
    print("hidden Markov model training, %d states, %d iterations" %
          (nstate, niter))

    hmm_model = {}

    # Initialize an HMM model for each digit
    for digit in digits:
        hmm_model[digit] = HMM(sg_model[digit], nstate=nstate)

    i = 0

    # training iterations
    while i < niter:
        print("iteration: %d" % i)
        total_log_like = 0.0
        for digit in digits:
            data = train_data[digit]
            print("process %d data for digit %s" % (len(data), digit))

            hmm_model[digit].train(data, i)

            for data_u in data:
                total_log_like += hmm_model[digit].loglike(data_u)

        print("log likelihood: %f" % (total_log_like))
        i += 1

    return hmm_model


def mlp_train(digits, train_data, hmm_model, uniq_state_dict, nunits=(128, 128), lr=0.01):
    """
     MLP-HMM model training, that is, using all data to train the neural network to classify the states of all HMMs,
     Then the Gaussian part in the original Gaussian HMM is replaced by a neural network to calculate the hidden state observation probability.
     train_data: all training data
     hmm_model: trained hmm model
     uniq_state_dict: hidden state dictionary
    """
    data_dict = {}
    seq_dict = {}
    for digit in digits:
        def uniq(t): return uniq_state_dict[(digit, t)]
        vfunc = np.vectorize(uniq)  # Vector function definition

        sequences = []
        data = train_data[digit]
        data_dict[digit] = data

        # Align the speech according to the incoming hmm model, and use the vector function for each alignment sequence to perform a state map to obtain a hidden state sequence
        for data_u in data:
            seq = hmm_model[digit].get_state_seq(data_u)
            sequences.append(vfunc(seq))

        seq_dict[digit] = sequences

    O = []
    S = []

    for digit in digits:
        data = data_dict[digit]
        sequences = seq_dict[digit]
        for data_u, seq in zip(data, sequences):
            data_u_expanded = context_expand(data_u)
            O.append(data_u_expanded)
            S.append(seq)

    # O is all data, S is the alignment status of all data and spliced ​​by the x-axis
    O = np.vstack(O)
    S = np.concatenate(S, axis=0)

    # MLP network definition
    mlp = MLPClassifier(hidden_layer_sizes=nunits, random_state=1, early_stopping=True, verbose=True,
                        validation_fraction=0.1)
    # MLP network training, the number of output nodes is the number of hidden states of all HMMs nstate*num_digits
    mlp.fit(O, S)

    # Here mlp_model is the HMM_MLP model corresponding to each number
    mlp_model = {}
    for digit in digits:
        # variables to initialize HMMMLP are incomplete below, pass additional variables that are required
        mlp_model[digit] = HMMMLP(mlp, hmm_model[digit], S, uniq_state_dict)

    return mlp_model
