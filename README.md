# cWT
In KS-DFFT the kinetic energy of non-interacting electrons $T_{s}$ is  described by the one-electron KS orbitals $\phi (\mathbf{r})$. 

$T_{s}[n]=\sum_{i}\langle \phi_i|-\frac{1}{2}\nabla^2|\phi_i\rangle$

The Kinetic energy depends on the one-electron orbital that scales the computational cost  like $\mathcal{O}(N^3)$. On the other hand, Orbital free Density Functional Theory (OFDFT) is an alternative to KSDFT. However, not just the Exchange-Correlation term needs to be approximated; it is also necessary to approximate the Kinetic energy as a functional of the electronic density (Kenetic energy density functional).

The KEDFT contains the local, semilocal and non-local contribution. Wang and Teter proposed a non-local approximation for $T_{s}$

$T_{WT}[n]=T_{TF}[n]+T_{vW}[n]+\int d\mathbf{r} \int d\mathbf{r}' n^{5/6}(\mathbf{r}) \omega (k_{F},|\mathbf{r} -\mathbf{r}'|)n^{5/6}(\mathbf{r}')$

where $k_{F} = [3\pi^{2}\bar{n}]^{1/3}$ is the fermi wave vector. However, the WT KEDF does not abey the scaling relations.

This code apply the corrected Wang-Teter (cWT) Kinetic Energy Density Functional (KEDF) that scales properly.

$T_{NL}[n]=\int d\mathbf{r} d{\mathbf{r}^\prime} n^{5/6}(\mathbf{r}) \omega (k_{TF}^{\alpha}|{\mathbf{r} -\mathbf{r}^\prime}|)n^{5/6}(\mathbf{r}^\prime)$

The cWT KEDF depends on a new parameter, $\rho_{0}$, which varies with the size of the system. To learn this dependence, we created a database of four phases of silicon and developed an interface that utilizes machine learning models, such as regression and neural networks, to predict $\rho_{0}$. Additionally, the interface performs full OF DFT calculations using the predicted $\rho_{0}$.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
