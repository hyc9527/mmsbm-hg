#!/usr/env python3
# pyright:  reportGeneralTypeIssues=false, reportUnboundedVariable=false

import numpy as np
from scipy.special import psi, polygamma, loggamma, binom
import itertools
import utils_mmsbm_uniform
from formatted_logger import formatted_logger

#import warnings
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.model_selection import train_test_split

log = formatted_logger('MMSB-online-vem', 'info')

class MMSB_hg_online(object):
    """
        Class:  MMSB for h0-uniform hypergraphs
        Global parameters
            h0-dim array Y, of shape (N,N,...,N), adjacency tensor
            int H, cardinality of hyperedges
            int K, number of latent clusters
            int N, number of nodes
            float rho, sparsity parameter
            float alpha0, initial value for symmetic alpha
        MMSB model parameters:
            1-dim array alpha, shape (K,), model parameters such that  pi_p sim Dirichlet(alpha)
            h0-dim array B, shape (K,,K,...,K),  affinity tensor
        Variational parameters:
            2-dim array gamma, shape (N,K), variational parameters to pi
            3-dim array phi, of shape (1,N,K), mat phi[0] of shape (N,K) stores phi;
    """
    def __init__(self, hg_adj, h0, num_cluster, alpha0 = 0.1, isRandomStart=True):
        """
            Constructor
        """

        # model setup
        if hg_adj is not None:
            self.Y = hg_adj
            self.N = hg_adj.shape[0]
        if h0 is not None:
            self.H = h0
        else:
            self.H = 3
            log.info('defualt h0 = 3, h0 is not specified.')
        if num_cluster is not None:
            self.K = num_cluster
        else:
            self.K = 3
            log.info('defualt K = 3, since K is not specified')
        assert self.H > 1 # length of hyperedge must be 2,3,...
        self.random_state=123
        self.max_iter_alpha = 1
        self.max_iter_vem = 5
        self.max_iter_phi =  5
        self.elbo_conv_tol = 1e-6
        self.tol_phi  = 1e-30
        self.num_minibatch = 1
        """ Initial values for model parameters: randomized or fixed
            isRandomStart = True:
                alpha0 is dist as  unif(0,1)
                b0 is dist as unif(0.7, 1)
                b1 is dist as unif(0, 0.1)
                phi[p:] is dist as dirichlet(alpha)
                gamma[p,k] = 1/K
        """
        if isRandomStart:
            self.alpha = np.ones(self.K)*alpha0
            #self.alpha = np.random.rand(self.K)
            self.b0 = np.random.uniform(0.7,1-1e-4,1) # true connecting prob of two nodes in same cluster, b0 ~ unif(0.7,1)
            self.b1 = np.random.uniform(1e-4,0.1,1) # ture connecting prob otherwise, b1 ~ uniform(0,0.1)
            self.rho = 0
            self.phi = np.array([np.random.dirichlet(self.alpha, size=(self.N)) for _ in range(1)])
            self.gamma =  np.ones((self.N, self.K))/self.K
        else:
            self.alpha = np.ones(self.K)*alpha0
            self.b0 = 0.9 # true connecting prob of two nodes in same cluster
            self.b1 = 0.1 # ture connecting prob otherwise
            self.rho = 0
            self.phi =  np.array([np.ones(self.alpha, size=(self.N)) for _ in range(1)])
            self.gamma =  np.ones((self.N, self.K))/self.K
        #print(f' Init values for alpha are {self.alpha}')
        #print(f' Init values for b0 b1 are {self.b0, self.b1}')
        #print(f'Init values for phi are {self.phi}')
        #print(f'Init values for gamma are {self.gamma}')






    def run_vem_online(self, verbose=True):
        """
           Online variational EM for  uniform mmsbm model
        """
        self.B = utils_mmsbm_uniform.gen_singleton_B(self.b0, self.b1, self.H, self.K) # generate B for a singleton model.
        estep_elbo = self.calc_elbo()
        mstep_elbo = np.copy(estep_elbo)
        res_elbo = estep_elbo
        log.info('========== TRAINING STARTS ==========')
        for epoch in range(self.max_iter_vem):
            old_elbo = mstep_elbo
            nodeset = [x for x in range(self.N)]  # the node set removing node p
            subset_power = np.array(list(itertools.combinations(nodeset, self.H)))
            num_edge = subset_power.shape[0]
            #edge_index_split = np.array_split(np.arange(num_edge), self.num_minibatch)
            edge_index_split = np.array_split(np.random.randint(num_edge, size=num_edge), self.num_minibatch)
            for b in range(self.num_minibatch):
                minibatch_ind = edge_index_split[b]
                edgeset = subset_power[minibatch_ind,]
                self.update_phi_online(edgeset) # online update phi
                self.update_gamma_online(edgeset) # update gamma
                estep_elbo = self.calc_elbo()
                if abs(estep_elbo - old_elbo)< self.elbo_conv_tol:
                    break
                else:
                    res_elbo = np.append(res_elbo, estep_elbo)
                self.optimize_B_singleton()
                self.optimize_alpha()
                mstep_elbo = self.calc_elbo()
                if verbose:
                    print(f'epoch {epoch} minibatch no. {b} of size {len(minibatch_ind)}: estep elbo = {estep_elbo};  mstep elbo = {mstep_elbo}')
                err = abs(mstep_elbo - estep_elbo)
                if err < self.elbo_conv_tol:
                    break
                else:
                    res_elbo = np.append(res_elbo, mstep_elbo)
        else:
            log.warn(f"max_iter reached: VEM is not converging.")
        log.info(" ========== TRAINING FINISHED ==========")
        return {"alpha": self.alpha, "B":self.B, "phi":self.phi, "gamma":self.gamma, "elbo":res_elbo}




    def update_phi_online(self, edge_set):
        '''
           Online  E step update phi using minibatch edge_set
        '''
        K_range = [range(0, self.K)]*(self.H-1) # a list of tuple (0,1,...,K-1), of len h0 - 1.
        K_power = [x for x in itertools.product(*K_range)] # a long list of tuple (k_1, k_2, ..., k_{h0-1}).
        for edge in edge_set:
            for iter in range(self.max_iter_phi):
                phi_old = self.phi
                for p in edge:
                    rem_nodeset = [x for x in range(self.N) if x != p]  # the node set removing node p
                    subset_power_p = list(itertools.combinations(rem_nodeset, (self.H-1))) # all (h0-1)-sets incident at node p
                    for k in range(self.K):
                        phi_gam = psi(self.gamma[p,k]) - psi(np.sum(self.gamma[p,:]))
                        for subset in subset_power_p:
                            sum_phi_lk = 0
                            for i in range(self.K**(self.H-1)):
                                phi_prod = 1
                                for j in range(self.H-1):
                                    phi_prod *= self.phi[0,subset[j],K_power[i][j]]
                                p_s = np.append(p, subset) # appended subset (p, s_1, ...,s_{h0-1})
                                p_s.sort()
                                pos = np.where(p_s==p)[0][0] # position node p in the appneded subset s =(s_1, s_2,....,s_{h0})
                                p_s = tuple(p_s)
                                k_s = list(K_power[i])
                                k_s.insert(pos,k)
                                k_s = tuple(k_s)
                                #assert self.B[k_s] > 0
                                if self.B[k_s] > 0 :
                                    #print(f' Y is {self.Y[p_s]} at edge {p_s}')
                                    if not np.array_equal(edge, p_s):
                                        llh = np.log(1-self.B[k_s])
                                    else:
                                        llh = np.log(self.B[k_s])
                                    sum_phi_lk += phi_prod*llh
                                else:
                                    log.warn(f'B at {k_s} is zero.')
                        self.phi[0,p,k] = np.exp(sum_phi_lk + binom(self.N-1, self.H-1)*phi_gam)  # type: ignore
                    #assert np.sum(self.phi[0,p,:]) > 0 # DEBUG
                    #print(f' edge {edge} iter {iter} node {p} phi_p is {self.phi[0,p,:]}')
                    #self.phi[0,p,:] = self.phi[0,p,:]/np.sum(self.phi[0,p,:])
                    if (np.sum(self.phi[0,p,:]) > 1e-250): # DEBUG
                        self.phi[0,p,:] = self.phi[0,p,:]/np.sum(self.phi[0,p,:])
                phi_error = np.sum(abs(self.phi - phi_old))/(self.N*self.K)
                if phi_error < self.tol_phi:
                    break
            else:
                log.warn('Nested phi upadate on edge {edge }: max iter is reached and phi may not be optimal.')


    def update_phi(self):
        '''
            E step update phi
        '''
        K_range = [range(0, self.K)]*(self.H-1)
        K_power = [x for x in itertools.product(*K_range)]
        multi_term = binom(self.N-1, self.H-1)
        for p in range(self.N):
            rem_nodeset = [x for x in range(self.N) if x != p]
            subset_power = list(itertools.combinations(rem_nodeset, (self.H-1)))
            for k in range(self.K):
                phi_p = np.zeros(self.K)
                for subset in subset_power:
                    sum_phi_lk = 0
                    for i in range(self.K**(self.H-1)):
                        phi_prod = 1
                        for j in range(self.H-1):
                            phi_prod *= self.phi[0,subset[j],K_power[i][j]]
                        p_s = np.append(p, subset)
                        p_s.sort()
                        pos = np.where(p_s==p)[0][0]
                        p_s = tuple(p_s)
                        k_s = list(K_power[i])
                        k_s.insert(pos,k)
                        k_s = tuple(k_s)
                        if self.B[k_s] > 1e-50 and self.B[k_s] < 1-1e-50: #DEBUG
                            llh = self.Y[p_s]*np.log(self.B[k_s]) + (1-self.Y[p_s])*np.log(1-self.B[k_s])
                            sum_phi_lk += phi_prod*llh
                phi_gam = psi(self.gamma[p,k]) - psi(np.sum(self.gamma[p,:]))
                phi_p[k] = np.exp(sum_phi_lk + multi_term*phi_gam)  # type: ignore
            if np.sum(phi_p) > 1e-100: # DEBUG
                self.phi[0,p,:] = phi_p/np.sum(phi_p)


    def update_gamma_online(self, edgeset = None):
        '''
            Nested E step update gamma
        '''
        if edgeset is not None: # online
            for edge in edgeset:
                 for p in edge:
                    self.gamma[p, ] = self.alpha + binom(self.N-1, self.H-1)*self.phi[0,p, :]  # DEBUG
        else: # full batch
            self.gamma = self.alpha + binom(self.N-1, self.H-1) * self.phi[0]




    def run_mstep(self):
        """
            Maximize the hyper parameters alpha and B in M-step
            Arguments:
            Input from global:
                H-dim array Y, of shape (N,N,...,N), adjacency matrix
                3-dim array phi, of shape (H,N,K)
                2-dim array gamma, of shape (N,K)
                1-dim array alpha, of shape (K,)
                H-dim array beta, of shape (K,K,...,K)
        """
        self.optimize_alpha()
        #self.optimize_B_singleton_3unif()
        self.optimize_B_singleton()




    def optimize_alpha(self):
        """
            Find MLE of alpha using Newton method
            Arguments:
                1-dim array param_alpha, of shape (K,)
                2-dim array vi_gamma, of shape (N,K), variational parameter to alpha
                int const, number of samples, N
            Output:
                param_alpha
        """
        for it in range(self.max_iter_alpha):
            alpha_old = self.alpha.copy()
            psi_sum_gam = psi(self.gamma.sum(axis=1)).reshape(-1, 1)
            psi_sum_alp = psi(self.alpha.sum())
            g = self.N * (psi_sum_alp - psi(self.alpha)) + (psi(self.gamma) - psi_sum_gam).sum(axis=0) # g: gradient
            z = self.N * polygamma(1, self.alpha.sum()) # polygamma(1,.) is the first derivative to psi
            h = -self.N * polygamma(1, self.alpha) # H = diag(h) + 1z1'
            if np.all((h > 1e-250) &( h< -1e-250)):
            #if  (1./z  > 1e-20) and (h.sum() > 1e-20): #! Dealing with corner cases where h or z = 0
                 c = (g / h).sum() / (1./z + (1./h).sum())
                 self.alpha -= (g - c) / h
            if np.any(self.alpha < 0): #! Force each alpha_k should be positive in dirichlet distribution
                self.alpha[self.alpha < 0] = 1e-200
            err = np.sqrt(np.mean((self.alpha - alpha_old) ** 2))
            crit = err < self.elbo_conv_tol
            if crit:
                break
        else:
            log.warn(f"max_iter={self.max_iter_alpha} reached: alpha estimation in M-step might not be optimal.")




    def optimize_B_singleton(self):
        """
             BASE model update  on B in m step,  where B is parametrized ONLY by  {b0, b1}.
        """
        nodeset = [x for x in range(self.N)]
        all_subsets = list(itertools.combinations(nodeset, self.H))
        num_0, dem_0, num_1, dem_1 = [0.0]*4
        K_range = [range(0, self.K)]*(self.H)
        K_power = [x for x in itertools.product(*K_range)]
        for edge in all_subsets:
            for k in range(self.K): # estimate for b0
                prod_phi = 1
                for j in range(self.H):
                    prod_phi *= self.phi[0,edge[j],k]
                num_0 += prod_phi * self.Y[edge]
                dem_0 += prod_phi
            for i in range(len(K_power)): # estimate for b1
                prod_phi = 1
                for j in range(self.H):
                    prod_phi *= self.phi[0,edge[j],K_power[i][j]]
                num_1 += prod_phi * self.Y[edge]
                dem_1 += prod_phi
        num_1 -= num_0
        dem_1 -= dem_0
        if dem_0 > 1e-250 and dem_1 > 1e-250: # DEBUG
            b0 = num_0/(dem_0*(1-self.rho)) # connecting prob that node p,q,r in the same cluster
            b1 = num_1/(dem_1*(1-self.rho)) # connecting prob that p,q r not in same cluster
            self.B = utils_mmsbm_uniform.gen_singleton_B(b0,b1,self.H, self.K)
        else:
            log.info(f'B is not updated since dem0 is {dem_0} and dem1 is {dem_1}')

    def calc_elbo(self):
        """
            Compute ELBO, see the expression for Lower Bound for the Likelihood in Appendix B.2  Airoldi(08)
            Arguments:
                (Global)
                1-dim array Y, of shape (K,), adjacency matrix
                4-dim array phi, of shape (2,N,N,K), phi[0] stores phi with right arrows and phi[1] with left arrows
                2-dim array gamma, of shape (N,K)
                1-dim array alpha, of shape (K,)
                2-dim array B, of shape (K,K)
            Output:
                float elbo.

        """
        elbo = 0
        elbo += self._term1()
        elbo += self._term2()
        elbo += self._term3()
        elbo -= self._term4()
        elbo -= self._term5()
        return elbo




    def _term1(self, ajacency_tensor = None, num_clust = None):
        """
            Compute $E^{Z sim q} [log p(Y|Z,B)]$
        """
        if ajacency_tensor is None:
            ajacency_tensor = self.Y
        if num_clust is None:
            num_clust = self.K
        res = 0.0
        nodeset = [x for x in range(self.N)]
        all_subsets = list(itertools.combinations(nodeset, self.H))
        K_range = [range(0, self.K)]*(self.H)
        K_power = [x for x in itertools.product(*K_range)] # each row (k_1, k_2, ..., k_{h0-1})
        res = 0
        for edge in all_subsets: #  subset = (s_1, s_2, ...,s_{H-1})
            for i in range(len(K_power)):
                phi_prod = 1
                for j in range(self.H):
                    phi_prod *= self.phi[0,edge[j],K_power[i][j]]
                edge = tuple(edge)
                k_s = tuple(K_power[i])  #(k, k_1, k_2, ...,k_{H-1})
                # assert self.B[k_s] > 0 #DEBUG
                if self.B[k_s] > 1e-50 and self.B[k_s] < 1 - 1e-50: # DEBUG
                    llh = self.Y[edge]*np.log(self.B[k_s]) + (1-self.Y[edge])*np.log(1-self.B[k_s])
                    res += phi_prod*llh
        return res


    def _term2(self):
        """
            Compute $E^{Z,pi sim q} [p2(Z|pi)]$

        """
        res = 0.0
        psi_gamma = psi(self.gamma) - psi(self.gamma.sum(axis=1).reshape(-1,1)) # of shape (N,K)
        #for v in range(H):
        for v in range(1):
            res += np.sum(self.phi[0]*psi_gamma)
        res *= binom(self.N-1, self.H-1)   # debug
        return res

    def _term3(self):
        """
            Compute $E^{pi sim q}[log p(pi | alpha)]$
            Arguments:
            Output:
        """
        psi_gamma = psi(self.gamma) - psi(self.gamma.sum(axis=1).reshape(-1,1)) # of shape (N,K)
        res = np.sum((self.alpha-1)*psi_gamma)-self.N*np.sum(loggamma(self.alpha)) + self.N*loggamma(np.sum(self.alpha))
        return res

    def _term4(self):
        """
            Compute $E^{pi sim q} [ log q(pi |gamma)]$
        """
        psi_gam = psi(self.gamma) - psi(self.gamma.sum(axis=1).reshape(-1,1)) # of shape (N,K)
        res = np.sum((self.gamma-1)*psi_gam) - np.sum(loggamma(self.gamma)) + np.sum(loggamma(self.gamma.sum(axis=1)))
        return res

    def _term5(self):
        """
            Compute $E^{Z sim q} [log q(Z|phi)]$
        """
        #log.info(f'phi is {self.phi}')
        safe_ind = np.where(self.phi>1e-250)       # DEBUG
        log_Phi = np.log(self.phi[safe_ind])
        return np.sum(self.phi[safe_ind]*log_Phi)



    def calc_bic_singleton(self, Y_test, K_test):
        """
            Compute estimated BIC for a **DIAGONAL** test hg using parameters estimation from run_vem.
            When calculating ELBO on a new hypergraph, only _term1() needs to be computed again.
        """
        elbo = self._term1(Y_test, K_test)
        elbo += self._term2()
        elbo += self._term3()
        elbo -= self._term4()
        elbo -= self._term5()
        bic = 2*elbo - (K_test + 2)*np.log(np.sum(Y_test))
        return bic




if __name__ == "__main__":
    num_node = 10
    hg_adj = utils_mmsbm_uniform.gen_singleton_hg(num_node)
    num_clust = 2
    model = MMSB_hg_online(hg_adj, num_clust)
    res = model.run_vem()



### obsolete
    #def run_estep_online(self, edge):
    #    """
    #        Compute update on nested variational parameters phi and gamma, for a $h0$-unifrom hg
    #    """
    #    self.update_phi_online(edge)
    #    self.update_gamma_online(edge)



