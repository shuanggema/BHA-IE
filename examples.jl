using Pkg
using Plots
using LinearAlgebra
using StatsBase
using Random
using Distributions

include("BayesianMixtures_reg/BayesianMixtures_reg.jl") 
B = BayesianMixtures_reg

################ Generating simulated data ################
p = 100
l = 5
L = 1 + l
n = 200

sigma2 = 1.0
nz_p = 5

p0 = [0.5, 0.5]

Random.seed!(1)

true_eta = rand(Uniform(0.8, 1.2), l)
true_z = wsample(1 : 2, p0, n)
    
true_theta = [[zeros(p * L + 1)]; [zeros(p * L + 1)]]   
true_theta[1][1] = rand(Uniform(-1.2, -0.8))
true_theta[2][1] = rand(Uniform(0.8, 1.2))
    
true_theta[1][2 : nz_p * L + 1] = vcat([[rand(Uniform(-1.2, -0.8), 3); 0; 0; 0] for i = 1 : nz_p]...)
true_theta[2][2 : nz_p * L + 1] = vcat([[rand(Uniform(0.8, 1.2), 3); 0; 0; 0] for i = 1 : nz_p]...)

# block diagonal
bk = zeros(5, 5)
for j = 1 : 5; for i = 1 : 5; bk[i, j] = 0.3 ^ abs(i - j); end; end
mu_x = zeros(p); fill!(mu_x, 1)
S_x = Matrix{Float64}(I, p, p);
s = 1 : 5 : p; s1 = 5 : 5 : p
for i = 1 : round(Int, p / 5); S_x[s[i] : s1[i], s[i] : s1[i]] = bk; end
X_bk = [rand(MvNormal(mu_x, S_x)) for i = 1 : n]
 
# conti W
mu_w = zeros(l)
S_w = Matrix{Float64}(I, l, l)
for i = 1 : l; for j = 1 : l; S_w[i, j] = 0.5 ^ abs(i - j) end; end; 
W_c = [rand(MvNormal(mu_w, S_w)) for i = 1 : n]

Z = [zeros(p * L) for i = 1 : n]
y = zeros(Float64, n)
X = deepcopy(X_bk)
W = deepcopy(W_c)
for i = 1 : n
    for j = 1 : p
        Z[i][(L * (j - 1) + 1) : L * j] = [X[i][j] X[i][j] * W[i]']
    end
end

for i = 1 : n
    for j = 1 : 2
        if true_z[i] == j
            y[i] =  true_theta[j][1] + dot(Z[i], true_theta[j][2 : p * L + 1]) + dot(W[i], true_eta) + rand(Normal(0, sqrt(sigma2)))
        end
    end
end

################ Specify options and run MCMC sampler ################
n_total, n_burn = 10000, 5000
o1 = B.options(y, X, W, Z, n_total, gamma = 1.0, m = 3, log_pk = k -> log(0.1)+(k-1)*log(0.9));
res1 = B.run_sampler(o1) 

Plots.plot(res1.t) 

mode(res1.t[(n_burn + 1) : n_total])

