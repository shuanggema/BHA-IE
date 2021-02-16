module BayesianMixtures_reg

using LinearAlgebra
using Distributions
using SpecialFunctions

include("Lower.jl")
include("MFM.jl")
include("generic.jl")

logsumexp(a,b) = (m = max(a,b); m == -Inf ? -Inf : log(exp(a-m) + exp(b-m)) + m)

function randp(p,k)
    s = 0.; for j = 1:k; s += p[j]; end
    u = rand()*s
    j = 1
    C = p[1]
    while u > C
        j += 1
        C += p[j]
    end
    @assert(j <= k)
    return j
end

function randlogp!(log_p,k)
    log_s = -Inf; for j = 1:k; log_s = logsumexp(log_s,log_p[j]); end
    p = log_p
    for j = 1:k; p[j] = exp(log_p[j]-log_s); end
    return randp(p,k)
end

function ordered_insert!(index,list,t)
    j = t
    while (j>0) && (list[j]>index)
        list[j+1] = list[j]
        j -= 1
    end
    list[j+1] = index
end

function ordered_remove!(index,list,t)
    for j = 1:t
        if list[j]>=index; list[j] = list[j+1]; end
    end
end

function ordered_next(list)
    j = 1
    while list[j] == j; j += 1; end
    return j
end

mutable struct MVN_params
    # read/write
    m::Array{Float64,1} # mean
    L::Array{Float64,2} # lower triangular matrix such that L*L' = covariance
    _L::Array{Float64,2} # _L = inv(L) and _L' * _L = precision
    R::Array{Float64,2} # covariance matrix
    _R::Array{Float64,2} # precision matrix
    logdetR::Float64 # log of the determinant of the covariance matrix
    d::Int64 # dimension
    function MVN_params(m, R)
        p = new(); p.m = copy(m); p.R = copy(R); p.d = d = length(m)
        p.L = zeros(d,d); Lower.Cholesky!(p.L, p.R, d)
        p._L = zeros(d,d); Lower.inverse!(p._L, p.L, d)
        p._R = zeros(d,d); Lower.multiplyMtN!(p._R, p._L, p._L, d)
        p.logdetR = Lower.logdetsq(p.L, d) 
        return p
    end
end


struct Hyperparameters
    n::Int64 # sample size
    p::Int64 # dimension of X
    l::Int64 # dimension of W
    L::Int64 # l + 1
    pL::Int64 # the total number of variables
    
    a0::Float64 # parameters of IG on sigma2
    b0::Float64 # parameters of IG on sigma2
    
    a1::Float64 # parameters of beta on pi0
    b1::Float64 # parameters of beta on pi0
    
    a2::Float64 # parameters of beta on pi1
    b2::Float64 # parameters of beta on pi1
    
    a3::Float64 # parameters of gamma on u
    b3::Float64 # parameters of gamma on u
    
    mb::MVN_params
    me::MVN_params 
    
    mu_m::Float64
    mu_s::Float64 
end

function construct_hyperparameters(X, W)
    n, p, l = length(X), length(X[1]), length(W[1])
    L = l + 1
    pL = p * L
    
    a0, b0 = 1, 1
    
    a1, b1 = 1, 1
    
    a2, b2 = 1, 1
    
    a3, b3 = 1, 1
    
    mb = MVN_params(zeros(L), Matrix{Float64}(I, L, L))
    me = MVN_params(zeros(l), 10 * Matrix{Float64}(I, l, l)) 
    
    mu_m, mu_s = 0.0, 10

    return Hyperparameters(n, p, l, L, pL, a0, b0, a1, b1, a2, b2, a3, b3, mb, me, mu_m, mu_s)
end

mutable struct Group
    n::Int64
    sum_zz::Array{Float64,2}
    sum_ww::Array{Float64,2}
    mu::Float64
    b::Array{Float64,1}
    tau::Array{Float64, 1}
    B::Array{Float64, 1}
    Group(p, l) = (g = new(); g.n = 0; g.sum_zz = zeros(p, p); g.sum_ww = zeros(l, l); 
        g.mu = 0.0; g.b = zeros(p); g.tau = zeros(p); g.B = zeros(p); g)
end

function Group_adjoin!(g, Z, W, eta0, H)
    l, pL = H.l, H.pL
    g.n += 1
    for j = 1 : pL; for i = 1 : pL; g.sum_zz[i, j] += Z[i] * Z[j]; end; end
    for j = 1 : l; for i = 1 : l; g.sum_ww[i, j] += W[i] * W[j]; end; end
end

function Group_remove!(g, Z, W, eta0, H)
    l, pL = H.l, H.pL
    g.n -= 1
    for j = 1 : pL; for i = 1 : pL; g.sum_zz[i, j] -= Z[i] * Z[j]; end; end
    for j = 1 : l; for i = 1 : l; g.sum_ww[i, j] -= W[i] * W[j]; end; end
end

function log_lik(y, Z, W, eta, sigma2, g, H)
    p, l, pL = H.p, H.l, H.pL
    
    B = g.B
    mu = g.mu
    
    BZ, Weta, res, res2 = 0.0, 0.0, 0.0, 0.0
    
    for i = 1 : pL; BZ += B[i] * Z[i]; end
    for i = 1 : l; Weta += W[i] * eta[i]; end
    res = y - mu - Weta - BZ
    res2 = res * res
    
    return -0.5 * (log(2) + log(pi) + log(sigma2) + res2 / sigma2)
end  

function base_sample(H)
    a1, b1, a2, b2, a3, b3 = H.a1, H.b1, H.a2, H.b2, H.a3, H.b3
    
    p0 = rand(Beta(a1, b1))
    p1 = rand(Beta(a2, b2))  

    u = rand(Gamma(a3, b3))
    s2 = rand(InverseGamma(1, u))
    
    return p0, p1, s2
end

function prior_sample!(g, g0, H)
    x = prior_sample(g0, H)
    g.b = x[1]
    g.tau = x[2]
    g.B = x[3]
    g.mu = x[4]
end

function prior_sample(g0, H)
    p, l, L, pL = H.p, H.l, H.L, H.pL
    mu_m, mu_s = H.mu_m, H.mu_s 
    m_b, S_b = H.mb.m, H.mb.R
    
    b, tau = zeros(pL), zeros(pL)
    p0 = g0[1]
    p1 = g0[2]
    s2 = g0[3]
    
    # for an intercept term
    mu = rand(Normal(mu_m, mu_s))
    
    for j = 1 : p
        if !(p0 < rand())
            b[(L * (j - 1) + 1) : L * j] = rand(MvNormal(m_b, S_b))
        end
        for l = 1 : L
            if l == 1; 
                tau[L * (j - 1) + l] = rand(truncated(Normal(0, sqrt(s2)), 0, Inf))
            else
                if !(p1 < rand())
                    tau[L * (j - 1) + l] = rand(truncated(Normal(0, sqrt(s2)), 0, Inf))
                end
            end
        end
    end
    
    B = tau .* b
    
    return b, tau, B, mu
end

function update_mu(y, Z, W, z, g, eta0, sigma2, c, H)
    n, p, l, L, pL = H.n, H.p, H.l, H.L, H.pL
    mu_m, mu_s = H.mu_m, H.mu_s
    mu_s2 = mu_s * mu_s
    
    ZZ = g[c].sum_zz
    B = g[c].B
    nd = g[c].n

    u = 0.0
    for k = 1 : n
        if z[k] == c;
            w_k = W[k]
            Z_k = Z[k]
            weta = 0.0; for i = 1 : l; weta += w_k[i] * eta0[i]; end
            ZB = 0.0; for i = 1 : pL; ZB += Z_k[i] * B[i]; end
            u += (y[k] - ZB - weta)
        end
    end
    
    a = mu_s2 * nd + sigma2
    mu = (mu_s2 * u + sigma2 * mu_m) / a
    S = mu_s2 * sigma2 / a
    
    x = rand(Normal(mu, sqrt(S)))
    return x
end

function update_tau(y, Z, W, z, g, eta0, sigma2, p1, s2, c, id, id_main, H)
    n, p, l, L, pL = H.n, H.p, H.l, H.L, H.pL
    
    ZZ = g[c].sum_zz
    b = g[c].b
    tau = g[c].tau
    mu = g[c].mu
    B = g[c].B
    
    n_tau = 0
    
    for j = 1 : pL
        u = 0.0
        id_rest_b = id[1:end .!= j]  
        
        for k = 1 : n
            if z[k] == c;
                w_k = W[k]
                Z_k = Z[k]
                weta = 0.0; for i = 1 : l; weta += w_k[i] * eta0[i]; end
                ZB = 0.0; for i in id_rest_b; ZB += Z_k[i] * B[i]; end
                u += Z_k[j] * (y[k] - mu - ZB - weta)
            end
        end

        _v = b[j] * b[j] * ZZ[j, j] / sigma2 + 1 / s2
        v = 1 / _v
        u = v * b[j] * u / sigma2
        
        if  j in id_main
            tau[j] = rand(truncated(Normal(u, sqrt(v)), 0, Inf))
        else
            p11 = p1 / (p1 + 2 * (1 - p1) * s2 ^ (-0.5) * v ^ (0.5) * exp(u ^ 2  / (2 * v)) * cdf(Normal(), u / sqrt(v)))
            if rand() < p11
                tau[j] = 0.0
            else
                tau[j] = rand(truncated(Normal(u, sqrt(v)), 0, Inf))
                n_tau += 1
            end
        end
        B[j] = tau[j] * b[j]
    end 

    return tau, n_tau, B
end

function update_b(y, Z, W, z, g, eta0, sigma2, p0, c, id, H)
    n, p, l, L, pL = H.n, H.p, H.l, H.L, H.pL
    m, I_L = H.mb.m, H.mb.R
    
    ZZ = g[c].sum_zz
    b = g[c].b
    tau = g[c].tau
    mu = g[c].mu
    B = g[c].B
    V = diagm(tau)

    n_b = 0
    M = zeros(L, L)
    _M = zeros(L, L)
    A = zeros(L, L)
    
    C = zeros(L)
    S = zeros(L, L)
    
    for j = 1 : p
        id_b = (L * (j - 1) + 1) : L * j
        id_rest_b = setdiff(id, id_b) 
        fill!(C, 0.0)
        fill!(S, 0.0)

        for k = 1 : n
            if z[k] == c;
                w_k = W[k]
                Z_k = Z[k]
                weta = 0.0; for i = 1 : l; weta += w_k[i] * eta0[i]; end
                ZB = 0.0; for i in id_rest_b; ZB += Z_k[i] * B[i]; end
                C += Z_k[id_b] * (y[k] - mu - ZB - weta)
            end
        end

        _A = (V[id_b, id_b] * ZZ[id_b, id_b] * V[id_b, id_b] + sigma2 * I_L)
        
        Lower.Cholesky!(M, _A, L) # M * M' = _A
        Lower.inverse!(_M, M, L) # _M = inv(M)
        Lower.multiplyMtN!(A, _M, _M, L) # _M' * _M = A
        
        log_det_A = Lower.logdetsq(_M, L) 
        det_A = exp(log_det_A)
        
        C = V[id_b, id_b] * C
        mu_b = A * C
        
        S = sigma2 * A
        S = Matrix(Hermitian(S))
        m_Am = Lower.quadratic(mu_b, m, M, L) # mu_b' * M * M' * mu_b = mu' * _A * mu

        p1 = p0 / (p0 + (1 - p0) * sigma2 ^ (0.5 * L) * det_A ^ (0.5) * exp(m_Am / (2 * sigma2)))
        
        if rand() < p1
            b[id_b] = fill(0.0, L)
        else
            b[id_b] = rand(MvNormal(mu_b, S)) 
            n_b += 1
        end
        
        B[id_b] = tau[id_b] .* b[id_b]
    end
    
    return b, n_b, B
end

function update_eta(y, Z, W, g, z, t, list, sigma2, H)
    n, l, pL = H.n, H.l, H.pL
    m, _R = H.me.m, H.me._R
    
    x = zeros(l)
    wres, ww = zeros(l), zeros(l, l)

    Q, _Q = zeros(l, l), zeros(l, l)
    A = zeros(l, l)
    
    for j = 1 : t; c = list[j]
        sum_ww = g[c].sum_ww
        B = g[c].B
        mu = g[c].mu
        ww += sum_ww
        for k = 1 : n
            if z[k] == c; 
                Z_k, W_k = Z[k], W[k]
                ZB = 0.0; for i = 1 : pL; ZB += Z_k[i] * B[i]; end
                wres += W_k * (y[k] - mu - ZB)
            end
        end
    end
    
    _A = ww + _R
    Lower.Cholesky!(Q, _A, l) # Q * Q' = _A
    Lower.inverse!(_Q, Q, l) # _Q = inv(Q)
    Lower.multiplyMtN!(A, _Q, _Q, l) # _Q' * _Q = A

    S = sigma2 * A
    S = Matrix(Hermitian(S))
    mu = A * (wres + _R * m)

    x = rand(MvNormal(mu, S))
    return x
end

function update_p0(n, n0, H)
    a, b = H.a1, H.b1
    x = rand(Beta(a + n0, b + n))
    return x
end

function update_p1(n, n0, H)
    a, b = H.a2, H.b2
    x = rand(Beta(a + n0, b + n))
    return x
end

function update_s2(n, g, r, t, list, H)
    p, pL = H.p, H.pL
    
    tau = zeros(pL)
    sum_tau2 = 0.0
    for j = 1 : t; c = list[j];
        tau = g[c].tau
        for i = 1 : pL
            if tau[i] > 0
                sum_tau2 += tau[i] * tau[i]
            end
        end
    end
    
    x = rand(InverseGamma(0.5 * (t * p + n) + 1, 0.5 * sum_tau2 + r))
    return x
end

function update_sigma2(y, Z, W, eta0, g, z, t, list, H)
    n, p, l, L, pL = H.n, H.p, H.l, H.L, H.pL
    me, R = H.me.m, H.me.R
    a0, b0 = H.a0, H.b0

    M, _M = zeros(l, l), zeros(l, l)
    Lower.reverse_Cholesky!(M, R, l) # M' * M = R
    Lower.inverse!(_M, M, l) # _M = inverse(M)
    
    res, res2 = 0.0, 0.0
    for j = 1 : t; c = list[j]
        B = g[c].B
        mu = g[c].mu
        for k = 1 : n; 
            if z[k] == c;
                Z_k, W_k = Z[k], W[k]
                ZB = 0.0; for i = 1 : pL; ZB += Z_k[i] * B[i]; end
                weta = 0.0; for i = 1 : l; weta += W_k[i] * eta0[i]; end
                res = y[k] - mu - ZB - weta
                res2 += res * res
            end
        end
    end

    m_Rm = Lower.quadratic(eta0, me, _M, l) # (eta - m)' * _M * _M' * (eta - m) 
        
    a = 0.5 * (n + l) + a0
    b = 0.5 * (res2 + m_Rm) + b0
    x = rand(InverseGamma(a, b))
    
    return x
end

function options(
        y, X, W, Z, # data
        n_total;
        n_keep = n_total,
        n_burn = round(Int, n_total / 10),
        
        t_max = 40,  
        m = 5, # number of auxiliary variables introduced in Neal's algorithm 8
        
        # MFM
        gamma = 1.0, # Dirichlet_k(gamma, ..., gamma)
        log_pk = k -> log(0.1)+(k-1)*log(0.9)
    )

    n = length(y)
    log_v = MFM.coefficients(log_pk, gamma, n, t_max + 1)
    a = b = gamma

    n_keep = min(n_keep, n_total)
    
    return Options(
        y, X, W, Z, 
        n_total, 
        n_keep,
        n_burn,
        t_max, 
        m,
        gamma, 
        log_pk, 
        a, b, log_v, n)
end

function run_sampler(options)
    o = options
    n, n_total, n_keep = o.n, o.n_total, o.n_keep
    
    println("n = $n, n_total = $n_total")
    print("Running... ")
    
    # Main run
    elapsed_time = @elapsed t_r, z_r, N_r, beta_r, b_r, tau_r, eta_r, sigma2_r, p0_r, p1_r, s2_r, r_r, keepers = sampler(o, n_total, n_keep)
    time_per_step = elapsed_time / n_total
    
    println("complete.")
    println("Elapsed time = $elapsed_time seconds")
    println("Time per step ~ $time_per_step seconds")
    
    return Result(t_r, z_r, N_r, beta_r, b_r, tau_r, eta_r, sigma2_r, p0_r, p1_r, s2_r, r_r, 
        keepers, elapsed_time, time_per_step)
end

function sampler(options, n_total, n_keep)
   
    y, X, W, Z, n = options.y, options.X, options.W, options.Z, options.n
    n_total, n_keep, n_burn = options.n_total, options.n_keep, options.n_burn
    t_max, m = options.t_max, options.m
    a, b, log_v = options.a, options.b, options.log_v
    
    H = construct_hyperparameters(X, W)
    n, p, l, L, pL = H.n, H.p, H.l, H.L, H.pL
    
    keepers = zeros(Int, n_keep)
    keepers[:] = round.(Int, range(round(Int, n_total / n_keep), stop = n_total, length = n_keep))
    keep_index = 0
    
    @assert(n == length(y))
    
    id = 1 : pL
    id_main = zeros(Int, p)
    for j = 1 : p 
        id_main[j] = L * (j - 1) + 1
    end
    
    # Initial values
    eta0 = rand(l) 
    sigma2 = rand(InverseGamma(1, 1)) 
    p0 = rand(Beta(1, 1))
    p1 = rand(Beta(1, 1))
    s2 = rand(Gamma(1, 1)) 
    r = rand(Gamma(1, 1))
    
    # Inital grouping
    group = [Group(pL, l)::Group for c = 1 : t_max + 3]
    t = 1  # number of clusters
    z = ones(Int, n)  # z[i] = the cluster ID associated with data point i
    list = zeros(Int, t_max + 3); list[1] = 1 
    c_next = 2
    N = zeros(Int, t_max + 3); N[1] = n # N[c] = size of cluster c
    for i = 1 : n; Group_adjoin!(group[1], Z[i], W[i], eta0, H); end
    g0 = base_sample(H) 
    prior_sample!(group[1], g0, H)
    c_prop = zeros(Int, m)   

    group = [Group(pL, l)::Group for c = 1 : t_max + 3]
    t = 2  # number of clusters
    z = zeros(Int, n)
    n1, n2 = 0, 0
    for i = 1 : n
        if rand() < 0.5
            z[i] = 1
            n1 += 1
        else 
            z[i] = 2
            n2 += 1
        end
    end
    list = zeros(Int, t_max + 3); list[1] = 1; list[2] = 2
    c_next = 3
    N = zeros(Int, t_max + 3); N[1] = n1; N[2] = n2 
    for i = 1 : n
        if z[i] == 1
            Group_adjoin!(group[1], Z[i], W[i], eta0, H); 
        else
            Group_adjoin!(group[2], Z[i], W[i], eta0, H); 
        end
    end
    prior_sample!(group[1], g0, H)
    prior_sample!(group[2], g0, H)
    c_prop = zeros(Int, m)  
    
    # hyperparameter for MFM
    log_p = zeros(n + 1)
    log_Nb = log.((1:n) .+ b)
    
    z_r = zeros(Int8, n, n_keep)
    t_r = zeros(Int8, n_keep)
    N_r = zeros(Int16, t_max + 3, n_keep)
    eta_r = zeros(Float64, n_keep, l)
    p0_r = zeros(Float64, n_keep)
    p1_r = zeros(Float64, n_keep)
    sigma2_r = zeros(Float64, n_keep)
    s2_r = zeros(Float64, n_keep)
    b_r = zeros(Float64, t_max + 3, p * L, n_keep)
    tau_r = zeros(Float64, t_max + 3, p * L, n_keep)
    beta_r = zeros(Float64, t_max + 3, p * L + 1, n_keep)
    r_r = zeros(Float64, n_keep)
    
    n_list_b = zeros(Int64, t_max + 3)
    n_list_tau = zeros(Int64, t_max + 3)
    n_b, n_b0 = 0, 0
    n_tau, n_tau0 = 0, 0

    for iteration = 1 : n_total

    # -------------- Resample z's --------------
        for i = 1 : n
            # remove point i from its cluster
            c = z[i]
            N[c] -= 1
            list_c = copy(list)
            
            g0 = base_sample(H)    
            if N[c] > 0
                for j = 1 : m; 
                    cc = ordered_next(list_c)
                    c_prop[j] = cc
                    ordered_insert!(cc, list_c, t + (j - 1))
                    prior_sample!(group[cc], g0, H)
                end
            else
                for j = 2 : m; 
                    cc = ordered_next(list_c)
                    c_prop[j] = cc
                    ordered_insert!(cc, list_c, t + (j - 1))
                    prior_sample!(group[cc], g0, H)
                end
                c_prop[1] = c
                # remove cluster {i}, keeping list in proper order
                ordered_remove!(c, list, t)
                t -= 1
            end

            # compute probabilities for resampling
            for j = 1 : t; cc = list[j]
                log_p[j] = log_Nb[N[cc]] + log_lik(y[i], Z[i], W[i], eta0, sigma2, group[cc], H)
            end
            for j = 1 : m; cc = c_prop[j]
                log_p[t + j] = log_v[t + 1] - log_v[t] + log(a) - log(m) + log_lik(y[i], Z[i], W[i], eta0, sigma2, group[cc], H)
            end
            
            # sample a new cluster 
            j = randlogp!(log_p, t + m)

            # add point i to it's new cluster
            if j <= t
                c = list[j]
            else
                c = c_prop[j - t]
                ordered_insert!(c, list, t)
                t += 1
                @assert(t <= t_max, "Sampled t has exceeded t_max. Increase t_max and retry.")
            end

            if c != z[i]
                Group_remove!(group[z[i]], Z[i], W[i], eta0, H)
                Group_adjoin!(group[c], Z[i], W[i], eta0, H)
            end
            
            for j = 1 : m
                cc = c_prop[j]
                if c != cc
                    fill!(group[cc].b, 0.0)
                    fill!(group[cc].tau, 0.0)
                    fill!(group[cc].B, 0.0)
                    group[cc].mu = 0.0
                end
            end
            z[i] = c 
            N[c] += 1
        end
        
        for j = 1 : t; c = list[j];
            group[c].mu = update_mu(y, Z, W, z, group, eta0, sigma2, c, H)
            b_j = update_b(y, Z, W, z, group, eta0, sigma2, p0, c, id, H)
            group[c].b = deepcopy(b_j[1])
            n_list_b[c] = deepcopy(b_j[2])
            group[c].B = deepcopy(b_j[3])
    
            tau_j = update_tau(y, Z, W, z, group, eta0, sigma2, p1, s2, c, id, id_main, H)
            group[c].tau = deepcopy(tau_j[1])
            n_list_tau[c] = deepcopy(tau_j[2])
            group[c].B = deepcopy(tau_j[3])
        end
        
        for j = 1 : t; cc = list[j]; 
            n_b += n_list_b[cc]  
            n_tau += n_list_tau[cc]  
        end 
        n_b0 = t * p - n_b  
        n_tau0 = t * p * l - n_tau  
    
        eta0 = update_eta(y, Z, W, group, z, t, list, sigma2, H)
        sigma2 = update_sigma2(y, Z, W, eta0, group, z, t, list, H)
        p0 = update_p0(n_b, n_b0, H)
        p1 = update_p1(n_tau, n_tau0, H)
        s2 = update_s2(n_tau, group, r, t, list, H)
        r = rand(Gamma(H.a3 + 1, s2 * H.b3 / (H.b3 + s2)))
    
        n_b, n_b0, n_tau, n_tau0 = 0, 0, 0, 0
        for j = 1 : t; cc = list[j]; n_list_b[cc] = 0; n_list_tau[cc] = 0; end
    
        if iteration == keepers[keep_index + 1]
            keep_index += 1
            
            t_r[keep_index] = t
            eta_r[keep_index, :] = eta0
            sigma2_r[keep_index] = sigma2
            p0_r[keep_index] = p0
            p1_r[keep_index] = p1
            s2_r[keep_index] = s2
            r_r[keep_index] = r
            
            for i = 1 : n; z_r[i, keep_index] = z[i]; end
            for j = 1 : t; cc = list[j]
                N_r[cc, keep_index] = N[cc]
                b_r[cc, :, keep_index] = group[cc].b
                tau_r[cc, :, keep_index] = group[cc].tau
                beta_r[cc, 1, keep_index] = group[cc].mu
                beta_r[cc, 2:end, keep_index] = group[cc].B
            end
        end
    end
    
    return t_r, z_r, N_r, beta_r, b_r, tau_r, eta_r, sigma2_r, p0_r, p1_r, s2_r, r_r, keepers

end

end
