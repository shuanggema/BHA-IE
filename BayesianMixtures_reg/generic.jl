struct Options
    y::Array{Float64,1}
    X::Array{Array{Float64,1},1}
    W::Array{Array{Float64,1},1}
    Z::Array{Array{Float64,1},1}
    n_total::Int64
    n_keep::Int64
    n_burn::Int64
    t_max::Int32
    m::Int8
    
    # MFM options
    gamma::Float64
    log_pk::Function
    
    # Partition distribution values
    a::Float64
    b::Float64
    log_v::Array{Float64,1}
    
    # Other
    n::Int64
end

struct Result
    # options::Options
    t::Array{Int32, 1}
    z::Array{Int32, 2}
    N::Array{Int32, 2}
    beta::Array{Float64, 3}
    b::Array{Float64, 3}
    tau::Array{Float64, 3}
    eta::Array{Float64, 2}
    sigma2::Array{Float64, 1}
    p0::Array{Float64, 1}
    p1::Array{Float64, 1}
    s2::Array{Float64, 1}
    r::Array{Float64, 1}
    keepers::Array{Int64, 1}
    elapsed_time::Float64
    time_per_step::Float64
end