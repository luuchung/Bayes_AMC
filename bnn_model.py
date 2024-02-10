import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

class BayesianModule(nn.Module):
    def init(self):
        super().__init__()
class TrainableRandomDistribution(nn.Module):
    #Samples weights for variational inference as in Weights Uncertainity on Neural Networks (Bayes by backprop paper)
    #Calculates the variational posterior part of the complexity part of the loss
    def __init__(self, mu, rho):
        super().__init__()

        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)
        self.register_buffer('eps_w', torch.Tensor(self.mu.shape))
        self.sigma = None
        self.w = None
        self.pi = np.pi
        #self.normal = torch.distributions.Normal(0, 1)

    def sample(self):

        self.eps_w.data.normal_()
        self.sigma = torch.log1p(torch.exp(self.rho))
        self.w = self.mu + self.sigma * self.eps_w
        return self.w

    def log_posterior(self, w=None):


        assert (self.w is not None), "You can only have a log posterior for W if you've already sampled it"
        if w is None:
            w = self.w

        log_sqrt2pi = np.log(np.sqrt(2*self.pi))
        log_posteriors =  -log_sqrt2pi - torch.log(self.sigma) - (((w - self.mu) ** 2)/(2 * self.sigma ** 2)) - 0.5
        return log_posteriors.sum()

class PriorWeightDistribution(nn.Module):
    #Calculates a Scale Mixture Prior distribution for the prior part of the complexity cost on Bayes by Backprop paper
    def __init__(self,
                 pi=1,
                 sigma1=0.1,
                 sigma2=0.001,
                 dist=None):
        super().__init__()


        if (dist is None):
            self.pi = pi
            self.sigma1 = sigma1
            self.sigma2 = sigma2
            self.dist1 = torch.distributions.Normal(0, sigma1)
            self.dist2 = torch.distributions.Normal(0, sigma2)

        if (dist is not None):
            self.pi = 1
            self.dist1 = dist
            self.dist2 = None



    def log_prior(self, w):

        prob_n1 = torch.exp(self.dist1.log_prob(w))

        if self.dist2 is not None:
            prob_n2 = torch.exp(self.dist2.log_prob(w))
        if self.dist2 is None:
            prob_n2 = 0

        # Prior of the mixture distribution, adding 1e-6 prevents numeric problems with log(p) for small p
        prior_pdf = (self.pi * prob_n1 + (1 - self.pi) * prob_n2) + 1e-6

        return (torch.log(prior_pdf) - 0.5).sum()
class BayesianLinear(BayesianModule):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 prior_sigma_1 = 0.1,
                 prior_sigma_2 = 0.4,
                 prior_pi = 1,
                 posterior_mu_init = 0,
                 posterior_rho_init = -7.0,
                 freeze = False,
                 prior_dist = None):
        super().__init__()

        #our main parameters
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.freeze = freeze


        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        #parameters for the scale mixture prior
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist

        # Variational weight parameters and sample
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(posterior_rho_init, 0.1))
        self.weight_sampler = TrainableRandomDistribution(self.weight_mu, self.weight_rho)

        # Variational bias parameters and sample
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(posterior_mu_init, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(posterior_rho_init, 0.1))
        self.bias_sampler = TrainableRandomDistribution(self.bias_mu, self.bias_rho)

        # Priors (as BBP paper)
        self.weight_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.bias_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        # Sample the weights and forward it
        w = self.weight_sampler.sample()

        if self.bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)

        else:
            b = torch.zeros((self.out_features), device=x.device)
            b_log_posterior = 0
            b_log_prior = 0

        # Get the complexity cost
        self.log_variational_posterior = self.weight_sampler.log_posterior() + b_log_posterior
        self.log_prior = self.weight_prior_dist.log_prior(w) + b_log_prior

        return F.linear(x, w, b)

def kl_divergence_from_nn(model):

    kl_divergence = 0
    for module in model.modules():
        if isinstance(module, (BayesianModule)):
            kl_divergence += module.log_variational_posterior - module.log_prior
    return kl_divergence

def variational_estimator(nn_class):

    def nn_kl_divergence(self):
        return kl_divergence_from_nn(self)

    setattr(nn_class, "nn_kl_divergence", nn_kl_divergence)

    def sample_elbo(self,
                    inputs,
                    labels,
                    criterion,
                    sample_nbr,
                    complexity_cost_weight=1):

        loss = 0
        for _ in range(sample_nbr):
            outputs = self(inputs)
            for i in range(outputs.shape[1]):
                loss += criterion(outputs[:, -1,:], labels)
            loss += self.nn_kl_divergence() * complexity_cost_weight
        return loss / sample_nbr

    setattr(nn_class, "sample_elbo", sample_elbo)

    return nn_class

@variational_estimator
class VI(nn.Module):
    def __init__(self, features = 2, hidden1 = 128, hidden2 = 64,classes = 23):
        super(VI, self).__init__()
        self.features = features
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.classes = classes


        self.lstm1 = nn.LSTM(input_size=self.features, hidden_size=self.hidden1, num_layers=2, batch_first=True)
        self.Linear1 = BayesianLinear(self.hidden1, self.hidden2, prior_sigma_1=1.0, prior_sigma_2=0.0025, prior_pi=0.5)
        self.Linear2 = BayesianLinear(self.hidden2, self.classes, prior_sigma_1=1.0, prior_sigma_2=0.0025, prior_pi=0.5) # model 23072023


    def forward(self, x):
        x_, _ = self.lstm1(x)
        x_ = self.Linear1(x_)
        x_ = self.Linear2(x_)
        return x_