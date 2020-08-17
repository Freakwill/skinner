

## Bayes-Q learning

est. of post brob. $p(c|x)$
$$
\{\hat{p}(c|x)\}_c\sim softmax(\{Q(x,c)\}_c)\\
\hat{p}(x,c)=\hat{p}(c|x)\hat{p}(x)
$$



estimate $p(c|x_i)$ by $q_i(x_i, c)$, $p(c|x_i)=\frac{e^{Tq_i(x_i,c)}}{\sum_ce^{Tq_i(x_i,c)}}\sim e^{Tq_i(x_i,c)}$; $q_i(x_i,c):=\sum_{\xi_i=x_i}p(\xi, c)$
$$
\{p(c|x_i)\}_c\sim softmax(\{q_i(x_i,c)\}_c)
$$
For unfamiliar state $x$ we have by nb
$$
\ln p(c|x)\sim \ln p(c|x_i)+(1-n)\ln p(c)\\
= T\sum_iq_i(x_i,c)-\sum_i\ln\sum_ce^{Tq_i(x_i,c)}+(1-n)\ln p(c)\\
\sim T\sum_iq_i(x_i,c)+(1-n)\ln p(c)
$$



$\hat{p}:=\frac{N_i+\lambda}{N+\lambda n}$

