#!/usr/bin/env python
# pyright: reportGeneralTypeIssues=false, reportMissingImports=false, reportMissingModuleSource=falss, reportUnboundVariable=false, reportInvalidStringEscapeSequence=false

import numpy as np
from scipy.special import psi, polygamma, loggamma, binom
import itertools
import utils_realdata_workplace
from formatted_logger import formatted_logger

log = formatted_logger('MMSB-general-hg', 'info')

class MMSB_hg_general(object):
    """
        Class:  MMSB for h_lst-uniform hypergraphs
        Global parameters
            a list h-dim tensor Ys,
            int H, cardinality of hyperedges
            int K, number of latent clusters
            int N, number of nodes
            float rho, sparsity parameter
            float alpha0, initial value for symmetic alpha
        MMSB model parameters:
            1-dim array alpha, shape (K,), model parameters such that  pi_p sim Dirichlet(alpha)
            h_lst-dim array B, shape (K,,K,...,K),  affinity tensor
        Variational parameters:
            2-dim array gamma, shape (N,K), variational parameters to pi
            3-dim array phi, of shape (1,N,K), mat phi[0] of shape (N,K) stores phi;
    """
    def __init__(self, Y_lst, h_lst, num_cluster, alpha0 = 0.1, isRandomStart=True):
        """
            Constructor
                Input:
                    Y_lst, a list of h-dim adjacency tensors
                    h_lst, a list of cardinalities. Default is [2,3]
                    num_cluster, K
        """
        # model setup
        assert Y_lst is not None
        self.Y_lst = Y_lst
        self.N = Y_lst[0].shape[0]
        if h_lst is not None:
            self.H_lst = h_lst
        else:
            self.H_lst = np.array([2,3])
            log.info('h_lst is not specified, defualt h_lst = [2,3] is used.')
        if num_cluster is not None:
            self.K = num_cluster
        else:
            self.K = 2
            log.info('K is not specified, defualt K = 2 is used.')
        self.random_state=123
        self.max_iter_alpha = 10
        self.max_iter_vem = 5
        self.max_iter_phi =  5
        self.elbo_conv_tol = 1e-6
        self.tol_phi  = 1e-30
        self.num_minibatch = 1
        self.out_folder="./output/"

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


    def run_vem_general(self, tol=1e-3, IsDiagonalOptB=False, verbose= False):
        """
            Variational EM for singleton model on general hypergrpah
        """
        self.B_lst = utils_realdata_workplace.gen_B_lst(self.b0, self.b1, self.H_lst, self.K, IsDiag=False)
        estep_elbo = self.calc_elbo()
        mstep_elbo = np.copy(estep_elbo)
        res_elbo = estep_elbo
        for epoch in range(self.max_iter_vem):
            self.run_e_step() # E step
            estep_elbo = self.calc_elbo()
            if abs(estep_elbo - mstep_elbo) < self.elbo_conv_tol:
                break
            else:
                res_elbo = np.append(res_elbo, estep_elbo)
            self.run_m_step(IsDiagonalOptB) # M step
            mstep_elbo = self.calc_elbo()
            if verbose:
                print(f'epoch {epoch}: estep elbo = {estep_elbo};  mstep elbo = {mstep_elbo}')
            err = abs(mstep_elbo - estep_elbo)
            if err < self.elbo_conv_tol:
                break
            else:
                res_elbo = np.append(res_elbo, mstep_elbo)
        else:
            log.warn(f"max_iter reached: VEM is not converging.")
        log.info(" ========== TRAINING FINISHED ==========")
        self.elbo=res_elbo
        return {"alpha": self.alpha, "B":self.B_lst, "phi":self.phi, "gamma":self.gamma, "elbo":res_elbo}

    def run_e_step(self, isCheckEstepELBO=False):
        """ Compute update on variational parameters phi and gamma, for a $h_lst$-unifrom hg
            Arguments:
            Input from global
                H-dim array Y, of shape (N,N,..,N), adjacency matrix
                3-dim array current phi, of shape (1,N,K), phi[v] stores phi associated with vertex v = 1,2,...,H;
                2-dim array current gamma, of shape (N,K)
                1-dim array current alpha, of shape (K,)
                H-dim array current B, of shape (K,K,...,K)
            Output:
                3-dim array updated phi, shape (H,N,K)
                2-dim array updated gamma, shape (N,K)
        """
        if isCheckEstepELBO:
            elbo = self.calc_elbo()
            print(f'before phi update elbo is {elbo}')
            #print(f'before update phi is {self.phi}')
        self.update_phi()
        #self.update_phi_old()
        if isCheckEstepELBO:
            #print(f'after update phi is {self.phi}')
            elbo = self.calc_elbo()
            print(f'after phi update, elbo is {elbo}')
        self.update_gamma()
        if isCheckEstepELBO:
            elbo = self.calc_elbo()
            print(f'after gamma upadate elbo is {elbo}')

    def update_phi_old(self):
        '''
            E step update phi for general hg
        '''
        # precompute the term \sum_h \binom(N-1, h-1)
        multiplicative_term = 0
        for ind_h in range(len(self.H_lst)):
            multiplicative_term +=  binom(self.N-1, self.H_lst[ind_h]-1)
        for p in range(self.N):
            rem_nodeset = [x for x in range(self.N) if x != p]
            for k in range(self.K):
                sum_phi_lk = 0
                for ind_h in range(len(self.H_lst)):
                    K_range = [range(0, self.K)]*(self.H_lst[ind_h]-1)
                    K_power = [x for x in itertools.product(*K_range)]
                    subset_power = list(itertools.combinations(rem_nodeset, (self.H_lst[ind_h]-1))) # \Omega_{(h-1, -p)}
                    phi_p = np.zeros(self.K) # BUG ???
                    for subset in subset_power:
                        for i in range(self.K**(self.H_lst[ind_h]-1)):
                            phi_prod = 1 # term \prod_{s, j, k_j} \phi_{s_j, k_j}
                            for j in range(self.H_lst[ind_h]-1):
                                phi_prod *= self.phi[0,subset[j],K_power[i][j]]
                        p_s = np.append(p, subset)
                        p_s.sort()
                        pos = np.where(p_s==p)[0][0]
                        p_s = tuple(p_s)
                        k_s = list(K_power[i])
                        k_s.insert(pos,k)
                        k_s = tuple(k_s)
                        if self.B_lst[ind_h][k_s] > 1e-10 and self.B_lst[ind_h][k_s] < 1-1e-10:
                            llh = self.Y_lst[ind_h][p_s]*np.log(self.B_lst[ind_h][k_s]) +\
                                    (1-self.Y_lst[ind_h][p_s])*np.log(1-self.B_lst[ind_h][k_s])
                            sum_phi_lk += phi_prod*llh
                phi_gam = psi(self.gamma[p,k]) - psi(np.sum(self.gamma[p,:]))
                phi_p[k] = np.exp(sum_phi_lk + multiplicative_term*phi_gam)
            #assert np.sum(self.phi[0,p,:]) > 0 # DEBUG
            if np.sum(phi_p) > 1e-10: # DEBUG: if phi_p is too small, calculation may not be stable
                #print(f"sum is {np.sum(phi_p)}; phi_p  is {phi_p}")
                self.phi[0,p,:] = phi_p/np.sum(phi_p)


    def update_phi(self):
        '''
            E step update phi for general hg
        '''
        multiplicative_term = 0.0 # precompute the term \sum_h \binom(N-1, h-1)
        for ind_h in range(len(self.H_lst)):
            multiplicative_term +=  binom(self.N-1, self.H_lst[ind_h]-1)
        for p in range(self.N):
            rem_nodeset = [x for x in range(self.N) if x != p]
            phi_p = np.zeros(self.K) # BUG ???
            for k in range(self.K):
                sum_phi_lk = 0
                for ind_h in range(len(self.H_lst)):
                    K_range = [range(0, self.K)]*(self.H_lst[ind_h]-1)
                    K_power = [x for x in itertools.product(*K_range)]
                    subset_power = list(itertools.combinations(rem_nodeset, (self.H_lst[ind_h]-1))) # \Omega_{(h-1, -p)}
                    for subset in subset_power:
                        for i in range(self.K**(self.H_lst[ind_h]-1)):
                            phi_prod = 1 # term \prod_{s, j, k_j} \phi_{s_j, k_j}
                            for j in range(self.H_lst[ind_h]-1):
                                phi_prod *= self.phi[0,subset[j],K_power[i][j]]
                        p_s = np.append(p, subset)
                        p_s.sort()
                        pos = np.where(p_s==p)[0][0]
                        p_s = tuple(p_s)
                        k_s = list(K_power[i])
                        k_s.insert(pos,k)
                        k_s = tuple(k_s)
                        if self.B_lst[ind_h][k_s] > 1e-10 and self.B_lst[ind_h][k_s] < 1-1e-10:
                            llh = self.Y_lst[ind_h][p_s]*np.log(self.B_lst[ind_h][k_s]) +\
                                    (1-self.Y_lst[ind_h][p_s])*np.log(1-self.B_lst[ind_h][k_s])
                            sum_phi_lk += phi_prod*llh
                phi_gam = psi(self.gamma[p,k]) - psi(np.sum(self.gamma[p,:]))
                phi_p[k] = np.exp(sum_phi_lk + multiplicative_term*phi_gam)
            if np.sum(phi_p) > 1e-100: # DEBUG: if phi_p is too small, calculation may not be stable
                self.phi[0,p,:] = phi_p/np.sum(phi_p)




    def update_gamma(self):
        '''
            E step update gamma
        '''
        multiplicative_term = 0
        for h in range(len(self.H_lst)):
            multiplicative_term +=  binom(self.N-1, self.H_lst[h]-1)
        for p in range(self.N):
            #self.gamma[p, ] = self.alpha + self.phi[:,p, :].sum(axis=0)
            self.gamma[p, ] = self.alpha + multiplicative_term*self.phi[0,p, :]  # debug




    def run_m_step(self, IsDiagonalOptB = False):
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
        if not IsDiagonalOptB:
            self.optimize_B_singleton()
        else:
            self.optimize_B_diagonal()



    def optimize_alpha(self,  tol=1e-6):
        """
            Find MLE of alpha using Newton method
            Arguments:
                1-dim array param_alpha, of shape (K,)
                2-dim array vi_gamma, of shape (N,K), variational parameter to alpha
                int const, number of samples, N
            Output:
                param_alpha
        """
        for iter in range(self.max_iter_alpha):
            alpha_old = self.alpha.copy()
            psi_sum_gam = psi(self.gamma.sum(axis=1)).reshape(-1, 1)
            psi_sum_alp = psi(self.alpha.sum())
            g = self.N * (psi_sum_alp - psi(self.alpha)) \
                        + (psi(self.gamma) - psi_sum_gam).sum(axis=0) # g: gradient
            z = self.N * polygamma(1, self.alpha.sum()) # polygamma(1,.) is the first derivative to psi
            h = -self.N * polygamma(1, self.alpha) # H = diag(h) + 1z1'
            h = np.array(h)
            #print(f'h is {h}')
            #print(np.all((h > 1e-300 )|( h< -1e-300)))
            #if np.all((h > 1e-30 )|(h< -1e-30)):
            #if sum(abs(h)) > 1e-10:
            c = (g / h).sum() / (1./z + (1./h).sum())
            #print(f'alpha = {self.alpha}, g = {g}, c = {c}, h = {h}, z = {z}')
            if np.sum(abs((g-c)/h)) < 10:
                self.alpha -= np.exp(-iter)*(g - c) / h
                #print(f'update on alpha on {(g-c)/h}')
            if np.any(self.alpha < 0): #! Force each alpha_k should be positive in dirichlet distribution
                #self.alpha[self.alpha < 0] = 1e-200
                self.alpha = alpha_old
                break
            err = np.sqrt(np.mean((self.alpha - alpha_old) ** 2))
            crit = err < tol
            if crit:
                break
        else:
            log.warn(f"max_iter={self.max_iter_alpha} reached: alpha estimation in M-step might not be optimal.")




    def optimize_B_singleton(self):
        """
             BASE model update  on B in m step,  where B is parametrized ONLY by  {b0, b1}.
        """
        num_0, dem_0, num_1, dem_1 = [0.0]*4
        nodeset = [x for x in range(self.N)]
        for ind_h in range(len(self.H_lst)):
            all_subsets = list(itertools.combinations(nodeset, self.H_lst[ind_h]))
            K_range = [range(0, self.K)]*(self.H_lst[ind_h])
            K_power = [x for x in itertools.product(*K_range)]
            for edge in all_subsets:
                for k in range(self.K): # estimate for b0
                    prod_phi = 1
                    for j in range(self.H_lst[ind_h]):
                        prod_phi *= self.phi[0,edge[j],k]
                    if len(self.Y_lst)>1:  # check gen hg
                        num_0 += prod_phi * self.Y_lst[ind_h][edge]
                    #else: ### BUG!
                        #print(f'Y_lst shape is {self.Y_lst[0][0]}')
                        #print(f'edge is {edge}')
                        #num_0 += prod_phi * self.Y_lst[edge]
                    dem_0 += prod_phi
                for i in range(len(K_power)): # estimate for b1
                    prod_phi = 1
                    for j in range(self.H_lst[ind_h]):
                        prod_phi *= self.phi[0,edge[j],K_power[i][j]]
                    if len(self.Y_lst)>1:  # check gen hg
                        num_1 += prod_phi * self.Y_lst[ind_h][edge]
                    #else:  ### BUG!
                    #    num_1 += prod_phi * self.Y_lst[edge]
                    dem_1 += prod_phi
        num_1 -= num_0
        dem_1 -= dem_0
        #log.info(num_0, dem_0)
        #log.info(num_1, dem_1)
        #assert dem_0 > 0
        #assert dem_1 > 0
        # print(dem_0, dem_1)
        if dem_0 > 0 and dem_1 > 0:
            b0 = num_0/(dem_0*(1-self.rho)) # connecting prob that node p,q,r in the same cluster
            b1 = num_1/(dem_1*(1-self.rho)) # connecting prob that p,q r not in same cluster
            if len(self.Y_lst)>1:  # check gen hg
                self.B_lst = utils_realdata_workplace.gen_B_lst(b0, b1, self.H_lst, self.K, IsDiag=False)
            else:
                self.B_lst = utils_realdata_workplace.gen_singleton_B(b0,b1, self.H_lst, self.K)
            self.b0 = b0
            self.b1 = b1
        #if dem_0 > 0 and dem_1 > 0: # DEBUG
        #    b0 = num_0/(dem_0*(1-self.rho)) # connecting prob that node p,q,r in the same cluster
        #    b1 = num_1/(dem_1*(1-self.rho)) # connecting prob that p,q r not in same cluster
        #    self.B_lst = utils_realdata_workplace.gen_singleton_B(b0,b1,self.H_lst, self.K)
        #else:
        #    log.info(f'B is not updated since dem0 is {dem_0} and dem1 is {dem_1}')



    def optimize_B_diagonal(self):
        """
             BASE model update  on B in m step,  where B is parametrized ONLY by  {b0, b1}.
        """
        nodeset = [x for x in range(self.N)]
        all_subsets = list(itertools.combinations(nodeset, self.H_lst))
        num_0, dem_0 = [[0.0]*self.K]*2
        num_1, dem_1 = [0.0]*2
        K_range = [range(0, self.K)]*(self.H_lst)
        K_power = [x for x in itertools.product(*K_range)]
        for edge in all_subsets:
            for k in range(self.K): # estimate for b0
                prod_phi = 1
                for j in range(self.H_lst):
                    prod_phi *= self.phi[0,edge[j],k]
                num_0[k] += prod_phi * self.Y_lst[edge]
                dem_0[k] += prod_phi
            for i in range(len(K_power)): # estimate for b1
                prod_phi = 1
                for j in range(self.H_lst):
                    prod_phi *= self.phi[0,edge[j],K_power[i][j]]
                num_1 += prod_phi * self.Y_lst[edge]
                dem_1 += prod_phi
        num_1 -= np.sum(num_0)
        dem_1 -= np.sum(dem_0)
        #log.info(num_0, dem_0)
        #log.info(num_1, dem_1)
        b0_diag = np.zeros(self.K)
        for k in range(self.K):
            #assert dem_0[k] > 0
            if dem_0[k] > 0:
                b0_diag[k] = num_0[k]/dem_0[k]
        # assert dem_1 > 0
        if dem_1 > 0:
            b1 = num_1/(dem_1*(1-self.rho)) # connecting prob that p,q r not in same cluster
        else:
            b1 = 0.0
        self.B_lst = utils_realdata_workplace.gen_diagonal_B(b0_diag,b1,self.H_lst, self.K)
        self.b0 = b0_diag
        self.b1 = b1
        #if dem_0 > 0 and dem_1 > 0: # DEBUG
        #    b0 = num_0/(dem_0*(1-self.rho)) # connecting prob that node p,q,r in the same cluster
        #    b1 = num_1/(dem_1*(1-self.rho)) # connecting prob that p,q r not in same cluster
        #    self.B_lst = utils_realdata_workplace.gen_singleton_B(b0,b1,self.H_lst, self.K)
        #else:
        #    log.info(f'B is not updated since dem0 is {dem_0} and dem1 is {dem_1}')



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
        #log.info(f' t1: {self._term1()}')
        elbo += self._term2()
        #log.info(f't2 = {self._term2()}')
        elbo += self._term3()
        #log.info(f't3 = {self._term3()}')
        elbo -= self._term4()
        #log.info(f't4 = {self._term4()}')
        elbo -= self._term5()
        #log.info(f't5: {self._term5()}')
        #log.info(f'term3 - term4 = {self._term3() - self._term4()}')
        return elbo




    def _term1(self, ajacency_tensor = None):
        """
            For general hg: compute $E^{Z sim q} [log p(Y|Z,B)]$
        """
        if ajacency_tensor is None:
            ajacency_tensor = self.Y_lst
        nodeset = [x for x in range(self.N)]
        res = 0.0
        for ind_h in range(len(self.H_lst)):
            h = self.H_lst[ind_h]
            all_subsets = list(itertools.combinations(nodeset, h)) # a list of h_lst-sets as tuples
            K_range = [range(0, self.K)]*h
            K_power = [x for x in itertools.product(*K_range)] # all k**h assignment of the form (k_1, k_2, ..., k_{h_lst}), k_j \in [K]
            k_power_h = self.K ** h
            assert k_power_h == len(K_power)
            for edge in all_subsets:
                for i in range(k_power_h):
                    phi_prod = 1.0
                    for j in range(h):
                        phi_prod *= self.phi[0,edge[j],K_power[i][j]]
                    edge = tuple(edge)
                    edge_assignment = tuple(K_power[i])
                    #assert self.B_lst[edge_assignment] > 0 # DEBUG: h_lst=1, it fails
                    #print(self.B_lst[edge_assignment])
                    #print(f' B elem. is {self.B_lst[ind_h][edge_assignment]}')
                    #print(f' B  shape  is {self.B_lst[ind_h].shape}')
                    #print(f' edge  {edge_assignment}')
                    if  self.B_lst[ind_h][edge_assignment] > 0  and self.B_lst[ind_h][edge_assignment]<1:
                        llh = self.Y_lst[ind_h][edge]*np.log(self.B_lst[ind_h][edge_assignment]) +\
                                (1-self.Y_lst[ind_h][edge])*np.log(1-self.B_lst[ind_h][edge_assignment])
                        res += phi_prod*llh
        return res


    def _term2(self):
        """
            General hypergraph: compute $E^{Z,pi sim q} [log p(Z|pi)]$

        """
        psi_gamma = psi(self.gamma) - psi(self.gamma.sum(axis=1).reshape(-1,1)) # of shape (N,K)
        res = 0
        for j in range(len(self.H_lst)):
            h = self.H_lst[j]
            res +=  binom(self.N-1, h-1) * np.sum(self.phi[0]*psi_gamma)
        return res

    def _term3(self):
        """
            Compute $E^{pi sim q}[log p(pi | alpha)]$
            Arguments:
            Output:
        """
        psi_gamma = psi(self.gamma) - psi(self.gamma.sum(axis=1).reshape(-1,1)) # of shape (N,K)
        return  np.sum((self.alpha-1)*psi_gamma)-self.N*np.sum(loggamma(self.alpha)) + self.N*loggamma(np.sum(self.alpha))

    def _term4(self):
        """
            Compute $E^{pi sim q} [ log q(pi |gamma)]$
        """
        psi_gam = psi(self.gamma) - psi(self.gamma.sum(axis=1).reshape(-1,1)) # of shape (N,K)
        return np.sum((self.gamma-1)*psi_gam) - np.sum(loggamma(self.gamma)) + np.sum(loggamma(self.gamma.sum(axis=1)))

    def _term5(self):
        """
            Compute $E^{Z sim q} [log q(Z|phi)]$
        """
        #log.info(f'phi is {self.phi}')
        safe_ind = np.where(self.phi>0)       # DEBUG
        log_Phi = np.log(self.phi[safe_ind])
        return np.sum(self.phi[safe_ind]*log_Phi)



    def calc_bic_singleton(self, Y_test):
        """
            Compute estimated BIC for a **DIAGONAL** test hg using parameters estimation from run_vem.
            When calculating ELBO on a new hypergraph, only _term1() needs to be computed again.
        """
        elbo = self._term1(Y_test)
        elbo += self._term2()
        elbo += self._term3()
        elbo -= self._term4()
        elbo -= self._term5()
        bic = 2*elbo - (self.K + 2)*np.log(np.sum(Y_test))
        log.info(f'2*elbo is {2*elbo}, panelty is {-(self.K+2)*np.log(np.sum(Y_test))}')
        return bic

    def save_output(self):
        """
         Save result from em
        """
        outfile = self.out_folder + 'test_res_workplace'
        np.savez_compressed(outfile + '.npz', alpha=self.alpha, b={"b0":self.b0, "b1":self.b1}, phi = self.phi, gamma = self.gamma, elbo=self.elbo)
        log.info(f'vem output is  saved in the file: {outfile + ".npz"}')
        log.info('To load output file: res=np.load(filename); check out what are in res: [var for var in res.files];for instance: res["elbo"];')



if __name__ == "__main__":
    num_node=10
    num_clust=2
    h_lst=[2,3]
    Y_lst=utils_realdata_workplace.gen_singleton_hg_general_adj(num_node,h_lst)
    model=MMSB_hg_general(Y_lst, h_lst, num_clust)
    res=model.run_vem_general()



""" Obsolete

"""
