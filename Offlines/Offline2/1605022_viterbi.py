# %%
import numpy as np
import math
from scipy.stats import norm
from sklearn.metrics import accuracy_score


# %%
obs = np.loadtxt("data.txt", dtype=float)

# %%
obs = obs.reshape(-1, 1)

# obs[2][0]

# %%
paramfile = "parameters.txt"
states = 0
with open(paramfile) as f:
    states = int(f.readline().strip())


trans_p = np.loadtxt(paramfile, dtype=float, skiprows=1, max_rows=states)

means = np.loadtxt(paramfile, dtype=float, skiprows=states +
                   1, max_rows=1).reshape(-1, 1)

sd = np.loadtxt(paramfile, dtype=float, skiprows=states +
                2, max_rows=1).reshape(-1, 1)
# trans_p
# means[1][0]

# %%
b = np.zeros([states, 1], float)
b[states-1][0] = 1

trans_p_copy = np.copy(trans_p)
a = trans_p_copy[:, :states-1]


for i in range(0, states-1):
    a[i][i] = a[i][i] - 1

a = np.vstack((a, np.ones([states, 1]))).reshape(states, states)

start_p = np.linalg.solve(a, b)


# %%
def initial_probability(transition_matrix):
    d = np.zeros([states, 1], float)
    d[states-1][0] = 1

    transition_matrix_copy = np.copy(transition_matrix)
    c = transition_matrix_copy[:, :states-1]

    for i in range(0, states-1):
        c[i][i] = c[i][i] - 1

    c = np.vstack((c, np.ones([states, 1]))).reshape(states, states)

    init_prob = np.linalg.solve(c, d)

    return init_prob


# %%
def normal_dist(x, mean, sd):
    prob_density = norm.pdf(x, loc=mean, scale=np.sqrt(sd))
    if prob_density == 0:
        prob_density = 1e-250
    return prob_density

# %%


def viterbi():

    problist = np.zeros([len(obs), states])
    prevlist = np.zeros([len(obs), states])

    for state in range(states):
        problist[0][state] = np.log(
            start_p[state] * normal_dist(obs[0][0], means[0][0], sd[0][0]))
        prevlist[0][state] = -1

    for t in range(1, len(obs)):

        for state in range(states):
            max_tr_prob = problist[t-1][0] + np.log(trans_p[0][state])
            prev_state_selected = 0

            for prev_state in range(1, states):
                tr_prob = problist[t - 1][prev_state] + \
                    np.log(trans_p[prev_state][state])
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_state_selected = prev_state

            max_prob = max_tr_prob + \
                np.log(normal_dist(obs[t][0], means[state][0], sd[state][0]))
            problist[t][state] = max_prob
            prevlist[t][state] = prev_state_selected

    opt = []
    max_prob = float('-inf')
    best_state = -1
    # Get most probable state and its backtrack
    for state in range(states):
        if problist[len(obs)-1][state] > max_prob:
            max_prob = problist[len(obs)-1][state]
            best_state = state

    opt.append(best_state)
    previous = best_state

    for t in range(len(problist) - 2, -1, -1):
        opt.insert(0, prevlist[t + 1][previous])
        previous = int(prevlist[t + 1][previous])

    opt = [int(x) for x in opt]

    return opt
    # print(opt)


# %%
output = viterbi()
# print(output.shape)

output_list = ['"El Nino"' if output[i] ==
               0 else '"La Nina"' for i in range(len(output))]

with open('states_Viterbi_wo_learning.txt', 'w') as f:
    for item in output_list:
        f.write("%s\n" % item)


# with open('sir2.txt') as f:
#     sirer_output_list = [line.rstrip() for line in f]

# accuracy_score(sirer_output_list , output_list)


# %%
def forward(initial_distribution, transition_matrix, means, sd):
    alpha = np.zeros((obs.shape[0], transition_matrix.shape[0]))
    for i in range(states):
        alpha[0][i] = initial_distribution[i][0] * \
            normal_dist(obs[0][0], means[i][0], sd[i][0])

    for i in range(1, obs.shape[0]):
        for j in range(trans_p.shape[0]):
            alpha[i][j] = alpha[i - 1].dot(transition_matrix[:, j]) * \
                normal_dist(obs[i][0], means[j][0], sd[j][0])

        alpha[i] = alpha[i] / (np.sum(alpha[i]))

    return alpha


# %%
def backward(transition_matrix, means, sd):
    beta = np.zeros((obs.shape[0], transition_matrix.shape[0]))

    beta[obs.shape[0] - 1] = np.ones((transition_matrix.shape[0]))

    mult = np.zeros((states))

    for i in range(obs.shape[0] - 2, -1, -1):
        for j in range(transition_matrix.shape[0]):
            for k in range(states):
                mult[k] = beta[i+1][k] * \
                    normal_dist(obs[i+1][0], means[k][0], sd[k][0])

            beta[i][j] = mult .dot(trans_p[j, :])

        beta[i] = beta[i] / np.sum(beta[i])

    return beta


# %%
def baum_welch(transition_matrix, means, sd, itr):

    for _ in range(itr):

        initial_distribution = initial_probability(transition_matrix)

        alpha = forward(initial_distribution, transition_matrix, means, sd)
        beta = backward(transition_matrix, means, sd)

        pi_star = alpha * beta

        for i in range(len(obs)):
            pi_star[i] = pi_star[i] / (np.sum(pi_star[i]))

        pi_star_star = np.zeros((len(obs)-1, states, states))

        for i in range(len(obs)-1):
            for j in range(states):
                for k in range(states):
                    pi_star_star[i][j][k] = alpha[i][j] * transition_matrix[j][k] * \
                        normal_dist(obs[i+1][0],  means[k][0],
                                    sd[k][0]) * beta[i+1][k]

        # print(pi_star_star)

        for i in range(len(obs)-1):
            pi_star_star[i] = pi_star_star[i] / (np.sum(pi_star_star[i]))

        transition_matrix = np.zeros((states, states))

        for j in range(states):
            for k in range(states):
                for i in range(len(obs)-1):
                    transition_matrix[j][k] = transition_matrix[j][k] + \
                        pi_star_star[i][j][k]

        for i in range(states):
            transition_matrix[i] = transition_matrix[i] / \
                (np.sum(transition_matrix[i]))

        for i in range(states):
            means[i][0] = np.sum(
                (pi_star[:, i].reshape(-1, 1) * obs)) / np.sum(pi_star[:, i])
            data_minus_mean_sq = (obs - means[i][0]) ** 2
            sd[i][0] = np.sqrt(np.sum(
                pi_star[:, i].reshape(-1, 1) * data_minus_mean_sq) / np.sum(pi_star[:, i]))

    initial_distribution = initial_probability(transition_matrix)

    # print(initial_distribution)

    return means, sd, transition_matrix, initial_distribution


# %%
means, sd, transition_matrix, distribution = baum_welch(trans_p, means, sd, 10)

with open('parameters_learned.txt', 'w') as f:
    f.write("%s\n" % str(states))
    for i in range(states):
        for j in range(states):
            f.write("%s " % str(transition_matrix[i][j]))

        f.write("\n")

    for i in range(states):
        f.write("%s " % str(means[i][0]))

    f.write("\n")

    for i in range(states):
        f.write("%s " % str(sd[i][0] ** 2))

    f.write("\n")

    for i in range(states):
        f.write("%s " % str(distribution[i][0]))

    f.write("\n")


# %%
trans_p = transition_matrix
start_p = distribution

output = viterbi()

output_list = ['"El Nino"' if output[i] ==
               0 else '"La Nina"' for i in range(len(output))]

with open('states_Viterbi_after_learning.txt', 'w') as f:
    for item in output_list:
        f.write("%s\n" % item)

with open('sir2.txt') as f:
    sirer_output_list = [line.rstrip() for line in f]

accuracy_score(sirer_output_list, output_list)
