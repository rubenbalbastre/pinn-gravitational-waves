#=
    ODE models for orbital mechanics
=#


function RelativisticOrbitModel(u, model_params, t)
    #=
        Defines system of odes which describes motion of
        point like particle in schwarzschild background, uses

        u[1] = χ
        u[2] = ϕ

        where, p, M, and e are constants
    =#
    χ, ϕ = u
    p, M, e  = model_params

    numer = (p-2-2*e*cos(χ)) * (1+e*cos(χ))^2
    denom = sqrt( (p-2)^2-4*e^2 )

    ϕ̇ = numer / (M*(p^(3/2))*denom) # eq. 11a
    χ̇ = numer * sqrt( p-6-2*e*cos(χ) )/( M*(p^2)*denom ) # eq. 11b

    return [χ̇, ϕ̇,]

end


function f!(F, x)
    # L: x[1]
    # E: x[2]
    F[1] = 1 + x[1]^2 /(p^2 * M^2) * (1 + e)^2 - 2/p * (1 + e) - 2*x[1]^2/(p^3*M^2)*(1 + e)^3 - x[2]^2
    F[2] = 1 + x[1]^2 /(p^2 * M^2) * (1 - e)^2 - 2/p * (1 - e) - 2*x[1]^2/(p^3*M^2)*(1 - e)^3 - x[2]^2
end

function j!(J, x)
    J[1, 1] = 2*x[1] / (p^2 * M^2) * (1 + e)^2 - 4*x[1]/(p^3*M^2)*(1 + e)^3
    J[1, 2] = -2*x[2]
    J[2, 1] = 2*x[1] / (p^2 * M^2) * (1 - e)^2 - 4*x[1]/(p^3*M^2)*(1 - e)^3
    J[2, 2] = -2*x[2]
end

using NLsolve

function E(p::Float64, e::Float64,M::Float64)::Float64
    """
    Energy Schwarzschild time-like geodesic
    """
    res = sqrt(((p-2-2*e)*(p-2+2*e))/(p*(p-3-e^2)))
    return res
end

function L(p,e, M)
    """
    Angular momentum Schwarzschild time-like geodesic
    """
    res = p*M/sqrt(p-3-e^2)
    return res
end

function RelativisticOrbitModel_schwarzschild_numerically(u, model_params, t)
    #=
        Defines system of odes which describes motion of
        point like particle in schwarzschild background, uses

        u[1] = χ
        u[2] = ϕ

        where, p, M, and e are constants
    =#
    χ, ϕ = u
    p, M, e  = model_params

    L₀ = p*M/sqrt(p-3-e^2)
    E₀ = sqrt(((p-2-2*e)*(p-2+2*e))/(p*(p-3-e^2)))

    numerical_solution = nlsolve(f!, j!, [ L₀; E₀])
    E = numerical_solution.zero[2]
    L = numerical_solution.zero[1]

    r = p*M/(1+e*cos(χ))
    drdχ = p*M*e*sin(χ)/(1+e*cos(χ))^2

    dϕdτ = L/r^2
    dtdτ = p/(p-2-2*e*cos(χ))*E
    drdτ = e * sin(χ) * sqrt( (p-6-2*e*cos(χ))/(p*(p-3-e^2)))

    ϕ̇ = dϕdτ / dtdτ #L/r^2 # eq. 11a
    # println(r)
    # println(χ)
    # println(E)
    # println(L)
    # println(drdτ)
    # println(dtdτ)
    # println(drdχ)
    χ̇ = drdτ / (dtdτ * drdχ) # eq. 11b

    return [χ̇, ϕ̇,]

end


function f_kerr!(F, x; p, M, e)
    # L: x[1]
    # E: x[2]
    L, E = x
    F[1] = - a^2 + a^2*E^2 - L^2 + (2 * (1 + e) * (a * E - L)^2)/p + (2 *M^2 *p)/(1 + e) - (M^2* p^2)/(1 + e)^2 + (E^2*M^2*p^2)/(1 + e)^2
    F[2] = - a^2 + a^2*E^2 - L^2 + (2 * (1 - e) * (a * E - L)^2)/p + (2 *M^2 *p)/(1 - e) - (M^2* p^2)/(1 - e)^2 + (E^2*M^2*p^2)/(1 - e)^2
end

function j_kerr!(J, x; p, M, e)
    L, E = x
    J[1, 1] = -2*L - 4*(1+e)*(a*E-L) / p
    J[1, 2] = 2*a^2*E + 4*a*(1+e)*(a*E-L)/p + 2*E*M^2*p^2/(1+e)^2
    J[2, 1] = -2*L - 4*(1-e)*(a*E-L)/p
    J[2, 2] = 2*a^2*E + 4*a*(1-e)*(a*E-L)/p + 2*E*M^2*p^2/(1-e)^2
end

function E_kerr(p::Float64, e::Float64, M::Float64, a::Float64)::Float64
    """
    Energy Schwarzschild time-like geodesic
    """
    res = sqrt((M^4*p^3*(-2 - 2*e + p)*(-2 + 2*e + p)*(-3 - e^2 + p) - 
    a^2*(-1 + e^2)^2*M^2*p^2*(-5 + e^2 + 3*p) - 
    2*sqrt(a^2*(-1 + e^2)^4*M^2*p^3*(a^4*(-1 + e^2)^2 + 
        M^4*(-4*e^2 + (-2 + p)^2)*p^2 + 
        2*a^2*M^2*p*(-2 + p + e^2*(2 + p)))))/(M^2*p^3*(-4*a^2*(-1 + 
         e^2)^2 + M^2*(3 + e^2 - p)^2*p)))
    return res
end

function L_kerr(p::Float64, e::Float64, M::Float64, a::Float64)::Float64
    """
    Angular momentum Schwarzschild time-like geodesic
    """
    res = sqrt((M^4*p^3*(-2 - 2*e + p)*(-2 + 2*e + p)*(-3 - e^2 + p) - 
        a^2*(-1 + e^2)^2*M^2*p^2*(-5 + e^2 + 3*p) - 
        2*sqrt(a^2*(-1 + e^2)^4*M^2*p^3*(a^4*(-1 + e^2)^2 + 
        M^4*(-4*e^2 + (-2 + p)^2)*p^2 + 
        2*a^2*M^2*p*(-2 + p + e^2*(2 + p)))))/(M^2*p^3*(-4*a^2*(-1 + e^2)^2 + 
        M^2*(3 + e^2 - p)^2*p)))*(a^4*(-1 + e^2)^4 + 
        a^2*(-1 + e^2)^2*M^2*p*(-4 + 3*p + e^2*(4 + p)) - sqrt(
        a^2*(-1 + e^2)^4*M^2*p^3*(a^4*(-1 + e^2)^2 + 
        M^4*(-4*e^2 + (-2 + p)^2)*p^2 + 
        2*a^2*M^2*p*(-2 + p + e^2*(2 + p)))))/(a^3*(-1 + e^2)^4 - 
        a*(-1 + e^2)^2*M^2*(-4*e^2 + (-2 + p)^2)*p)
    return res
end


function RelativisticOrbitModel_kerr_numerically(u, model_params, t)
    #=
        Defines system of odes which describes motion of
        point like particle in schwarzschild background, uses

        u[1] = χ
        u[2] = ϕ
    =#
    χ, ϕ = u
    p, M, e, a  = model_params
    # f_kerr(F, x) = f_kerr!(F, x, p=p, M=M, e=e)
    # j_kerr(F, x) = j_kerr!(F, x, p=p, M=M, e=e)

    L₀ =  L_kerr(p, e, M, a)
    E₀ = E_kerr(p, e, M, a)
    # numerical_solution = nlsolve(f_kerr, j_kerr, [ L₀; E₀])
    E = E₀ #numerical_solution.zero[2]
    L = L₀ #numerical_solution.zero[1]

    r = p*M/(1+e*cos(χ))
    drdχ = p*M*e*sin(χ)/(1+e*cos(χ))^2

    Δ = r^2 - 2*M*r + a^2
    dϕdτ = ((1-2*M/r)*L + 2*M*a*E/r) / Δ
    dtdτ = ((r^2 + a^2 + 2*M*a^2/r)*E - 2*M*a*L/r) / Δ
    # x = (r^2*E^2 + 2*M*(a*E-L)^2/r + (a^2*E^2 - L^2) - Δ)/r^2
    x_1 = e * sin(χ) * sqrt( (p-6-2*e*cos(χ))/(p*(p-3-e^2)))
    if x_1 < 0
        x1 = - (x_1)^2
    else
        x1=x_1^2
    end
    x2 = (2*M/r*(2*a*E*L+a^2*E^2) + a^2*(E^2 - 1))/r^2
    x = x1 + x2
    drdτ = real(sqrt(Complex(x)))
    if drdτ == 0.0
        drdτ = -imag(sqrt(Complex(x)))
    end

    ϕ̇ = dϕdτ / dtdτ
    # println(r)
    # println(χ)
    # println(E)
    # println(L)
    # println(x1)
    # println(x2)
    # println(x)
    # println(drdτ)
    # println(dtdτ)
    # println(drdχ)
    χ̇ = drdτ / (dtdτ * drdχ)

    return [χ̇, ϕ̇,]
end


function AbstractNNOrbitModel_kerr(u, model_params, t; NN=nothing, NN_params=nothing)
    #=
        Defines system of odes which describes motion of
        point like particle with Newtonian physics, uses

        u[1] = χ
        u[2] = ϕ

        where, p, M, and e are constants
    =#
    χ, ϕ = u
    p, M, e, a  = model_params

    L =  L_kerr(p, e, M, a)
    E = E_kerr(p, e, M, a)

    r = p*M/(1+e*cos(χ))
    drdχ = p*M*e*sin(χ)/(1+e*cos(χ))^2

    Δ = r^2 - 2*M*r + a^2
    dϕdτ = ((1-2*M/r)*L + 2*M*a*E/r) / Δ
    dtdτ = ((r^2 + a^2 + 2*M*a^2/r)*E - 2*M*a*L/r) / Δ

    x_1 = sin(χ) * sqrt( (p-6-2*e*cos(χ))/(p*(p-3-e^2))) #  e * 

    if x_1 < 0
        x1 = - (x_1)^2
    else
        x1=x_1^2
    end
    x2 = (2*M/r*(2*a*E*L+a^2*E^2) + a^2*(E^2 - 1))/r^2
    x = x1 + x2
    drdτ = real(sqrt(Complex(x)))

    if drdτ == 0.0
        drdτ = -imag(sqrt(Complex(x)))
    end

    ϕ̇ = dϕdτ / dtdτ
    χ̇ = drdτ / (dtdτ * drdχ)

    return [χ̇, ϕ̇]
end



function AbstractNNOrbitModel(u, model_params, t; NN=nothing, NN_params=nothing)
    #=
        Defines system of odes which describes motion of
        point like particle with Newtonian physics, uses

        u[1] = χ
        u[2] = ϕ

        where, p, M, and e are constants
    =#
    χ, ϕ = u
    p, M, e  = model_params

    if isnothing(NN)
        nn = [1,1]
    else
        nn = 1 .+ NN(u, NN_params)
    end

    numer = (1 + e*cos(χ))^2
    denom = M*(p^(3/2))

    χ̇ = (numer / denom) * nn[1]
    ϕ̇ = (numer / denom) * nn[2]

    return [χ̇, ϕ̇]

end


function AbstractNNOrbitModel_schwarzschild_numerically(u, model_params, t; NN=nothing, NN_params=nothing)
    #=
        Defines system of odes which describes motion of
        point like particle with Newtonian physics, uses

        u[1] = χ
        u[2] = ϕ

        where, p, M, and e are constants
    =#
    χ, ϕ = u
    p, M, e  = model_params

    if isnothing(NN)
        nn = [1,1]
    else
        nn = 1 .+ NN(u, NN_params)
    end

    numer = (1 + e*cos(χ))^2
    denom = M*(p^(3/2))

    χ̇ = (numer / denom) * nn[1]
    ϕ̇ = (numer / denom) * nn[2]

    return [χ̇, ϕ̇]

end

#------------------- made by Rubén -------------#


function NR_OrbitModel_Ruben(u, model_params, t;
    NN_chiphi=nothing, NN_chiphi_params=nothing,
    NN_pe=nothing, NN_pe_params=nothing)::Vector
    #=
    Defines system of ODEs which describes motion of
    point like particle with Newtonian physics.
    Equation of motion => u = [χ, ϕ, p, e]

    Arguments:
        - u: solution at instant t
        - model_params:
        - N_chiphi: NN
        - N_pe: NN
        - N_chiphi: vector of parameters of NN_chiphi_params
        . N_pe_params: vector of parameters of NN_pe params

    Returns:
        - new_u: solution at instant t+1
    =#

    # get actual solution at instant t
    χ = u[1]
    ϕ = u[2]
    p = u[3]
    e = u[4]

    q = model_params[1]
    M = model_params[2]

    if p <= 0
        # println("p = ", p)
    end

    if isnothing(NN_chiphi)
        nn_chiphi = [1,1]
    else
        nn_chiphi = 1 .+ NN_chiphi(u, NN_chiphi_params)
    end

    if isnothing(NN_pe)
        nn_pe = [0,0]
    else
        nn_pe = NN_pe(u, NN_pe_params)
    end

    numer = (1+e*cos(χ))^2
    denom = M*(abs(p)^(3/2))

    # get solution in t+1
    χ̇ = (numer / denom) * nn_chiphi[1] # eq. 5a
    ϕ̇ = (numer / denom) * nn_chiphi[2] # eq. 5b
    ṗ = nn_pe[1] # eq. 5c
    ė = nn_pe[2] # eq. 5d
    new_u = [χ̇, ϕ̇, ṗ, ė, q, M]

    return new_u

end


function AbstractNNOrbitModel_Ruben(u, model_params, t; NN=nothing, NN_params=nothing)
    #=
        Defines system of odes which describes motion of
        point like particle with Newtonian physics, uses

        u[1] = χ
        u[2] = ϕ

        where, p, M, and e are constants
    =#
    χ, ϕ = u
    p, M, e  = model_params

    if isnothing(NN)
        nn = [1,1]
    else
        nn = 1 .+ NN(u, NN_params)
    end

    numer = (1 + e*cos(χ))^2
    denom = M*(p^(3/2))

    χ̇ = (numer / denom) * nn[1]
    ϕ̇ = (numer / denom) * nn[2]

    return [χ̇, ϕ̇, p, M, e]

end

function RelativisticOrbitModel_Ruben(u, model_params, t)
    #=
        Defines system of odes which describes motion of
        point like particle in schwarzschild background, uses

        u[1] = χ
        u[2] = ϕ

        where, p, M, and e are constants
    =#
    if length(u) == 2
        χ, ϕ = u
    elseif length(u) == 5
        χ, ϕ, _, _, _ = u
    end
    p, M, e  = model_params

    numer = (p-2-2*e*cos(χ)) * (1+e*cos(χ))^2
    denom = sqrt( (p-2)^2-4*e^2 )

    ϕ̇ = numer / (M*(p^(3/2))*denom) # eq. 11a
    χ̇ = numer * sqrt( p-6-2*e*cos(χ) )/( M*(p^2)*denom ) # eq. 11b

    return [χ̇, ϕ̇, p, M, e]

end
