module MFM

using SpecialFunctions

logsumexp(a,b) = (m = max(a,b); m == -Inf ? -Inf : log(exp(a-m) + exp(b-m)) + m)

function coefficients(log_pk::Function,gamma,n,upto)
    tolerance = 1e-12
    log_v = zeros(upto)
    for t = 1:upto
        if t>n; log_v[t] = -Inf; continue; end
        a,c,k,p = 0.0, -Inf, 1, 0.0
        while abs(a-c) > tolerance || p < 1.0-tolerance  # Note: The first condition is false when a = c = -Inf
            if k >= t
                a = c
                b = logabsgamma(k+1)[1]-logabsgamma(k-t+1)[1]-logabsgamma(k*gamma+n)[1]+logabsgamma(k*gamma)[1] + log_pk(k)
                c = logsumexp(a,b)
            end
            p += exp(log_pk(k))
            k = k+1
        end
        log_v[t] = c
    end
    return log_v
end

end