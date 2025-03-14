using Pkg
Pkg.activate("/localdata/Ray/VerticalMathProject/Code")
using DifferentialEquations, Plots, Sundials, LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)
function setup_fluid_model(; 
    DomainPoints=150, 
    starting_height=1.0, 
    T_max=128.0, 
    λ=1.0, 
    DR=800 * 3600.0, 
    ε=1e-5, 
    c=0.7, 
    R_left=5.0, 
    β=0.05, 
    γ=1.0, 
    μ=0.1, 
    n_hill=1.0, 
    MIC=1.0, 
    DA=1.0, 
    A_left=1.0, 
    k_cat=1.0, 
    k_m=1.0, 
    IB=1.0, 
    DB=1.0, 
    DN=1.0, 
    A_ν="Dirichlet", 
    R_ν="Dirichlet", 
    B_ν="Dirichlet",
    η=0.0)

    # Spatial grid
    z = LinRange(0.0, 1.0, DomainPoints)
    Δz = z[2] - z[1]   # Spatial step size
    Δz_sq = Δz^2       # Square of the spatial step size

    # Interpret boundary condition inputs
    A_ν_value = isa(A_ν, String) ? (A_ν == "Dirichlet" ? 1.0 / Δz : 0.0) : A_ν
    R_ν_value = isa(R_ν, String) ? (R_ν == "Dirichlet" ? 1.0 / Δz : 0.0) : R_ν
    B_ν_value = isa(B_ν, String) ? (B_ν == "Dirichlet" ? 1.0 / Δz : 0.0) : B_ν

    # Allocate workspace arrays
    SumGi = zeros(DomainPoints)
    g1 =     zeros(DomainPoints)
    g2 =     zeros(DomainPoints)
    α =      zeros(DomainPoints)
    β_vec =  zeros(DomainPoints)  # Renamed to avoid collision with scalar β
    μ_vec =  zeros(DomainPoints)  # Renamed to avoid collision with scalar μ
    ψ_vec =  zeros(DomainPoints)
    RHS_p =  zeros(DomainPoints + 1)  # Needs an extra space to enforce Neumann BC
    grad_p = zeros(DomainPoints)
    grad_N1 =    zeros(DomainPoints)
    grad_R =     zeros(DomainPoints)
    term1 =      zeros(DomainPoints)
    LaplacianR = zeros(DomainPoints)

    # Return all parameters and arrays as a named tuple
    return (
        N = DomainPoints,
        starting_height = starting_height,
        T_max = T_max,
        λ = λ,
        DR = DR,
        ε = ε,
        c = c,
        R_left = Float64[R_left],
        β = β,
        z = z,
        Δz = Δz,
        Δz_sq = Δz_sq,
        γ = γ,
        μ = μ,
        n_hill = n_hill,
        MIC = MIC,
        DA = DA,
        A_left = Float64[A_left],
        k_cat = k_cat,
        k_m = k_m,
        IB = IB,
        DB = DB,
        DN = DN,
        # Boundary condition ν values
        A_ν = A_ν_value, # Dirichlet or Neumann for A
        R_ν = R_ν_value, # Dirichlet or Neumann for R
        B_ν = B_ν_value, # Dirichlet or Neumann for B
        # Workspace arrays
        SumGi = SumGi,
        g1 = g1,
        g2 = g2,
        α = α,
        β_vec = β_vec,
        μ_vec = μ_vec,
        ψ_vec = ψ_vec,
        RHS_p = RHS_p,
        grad_p = grad_p,
        grad_N1 = grad_N1,
        grad_R = grad_R,
        term1 = term1,
        LaplacianR = LaplacianR,
        η = η
    )
end
@inline function α!(α, R, c, γ)
    @inbounds for i in eachindex(α)
        α[i] = c * R[i]
    end
end
@inline function β!(β_vec, β)
    @inbounds  for i in eachindex(β_vec)
        β_vec[i] = β
    end
end
@inline function μ!(μ_vec, μ)
    @inbounds  for i in eachindex(μ_vec)
        μ_vec[i] = μ
    end
end
@inline function ψ!(ψ_vec, α, A, n_hill, MIC)
    @inbounds  for i in eachindex(ψ_vec)
        ψ_vec[i] = 2*α[i]*(A[i]^n_hill)/(MIC^n_hill + A[i]^n_hill)
    end
end
@inline function g1!(g1, α, β_vec, ψ_vec, N1)
    @inbounds  for i in eachindex(g1)
        N1_i = N1[i]
        a1 = α[i]
        b1 = β_vec[i]
        ψ1 = ψ_vec[i]
        g1[i] = (a1 - b1 -ψ1) * N1_i
    end
end
@inline function g2!(g2, N1, N2, β_vec, μ_vec, ψ_vec, η)
    @inbounds  for i in eachindex(g2)
        N1_i = N1[i]
        N2_i = N2[i]
        b1 = β_vec[i]
        m1 = μ_vec[i]
        ψ1 = ψ_vec[i]
        g2[i] = N1_i*(η*ψ1 + b1) - m1*N2_i
    end
end
@inline function SumGi!(SumGi, g1, g2)
    @inbounds  for i in eachindex(SumGi)
        g1_i = g1[i]
        g2_i = g2[i]    
        SumGi[i] = g1_i + g2_i
    end
end
@inline function update_pressure!(RHS_p, grad_p, SumGi, H, p)
    # Compute RHS of Poisson equation
    RHS_p[1] = 0.0 # Left boundary (growth of the wall if you will)
    
    @inbounds for i in eachindex(grad_p)
        RHS_p[i+1] = -(H * H * SumGi[i]) / p.λ # The first RHS_p must always be zero for the Neumann BC
    end
    # Compute cumulative sum for the integral
    integral_sum = 0.0
    @inbounds for i in 1:length(grad_p)+1
        integral_sum += RHS_p[i] * p.Δz
        RHS_p[i] = integral_sum
    end
    @inbounds for i in eachindex(grad_p)
        grad_p[i] = RHS_p[i+1]
    end
end
@inline function update_subrate_N!(N, grad_N, g, dN, grad_p, term1, SumGi, H, p, dH, laplacian)
    # Compute ∇N using WENO5
    compute_gradient_central!(grad_N, N, p.Δz)

    # Find laplacian of N
    compute_laplacian_Neumann!(laplacian, N, p.Δz_sq)

    # Compute (λ∇p)·(∇N) / H^2
    @inbounds for i in eachindex(dN)
        term1[i] = p.λ * grad_p[i] * grad_N[i] / H^2
    end

    # Compute ∂N/∂t
    @inbounds for i in eachindex(dN)
        dN[i] = term1[i] + g[i] - (N[i] * SumGi[i]) + (p.z[i] * dH * grad_N[i] / H) + (p.DN * laplacian[i] / H^2)
    end
end
@inline function update_dR!(dR, N1, α, laplacian_R, dH, H, grad_R, p)
    hsq=H^2
    @inbounds for i in eachindex(dR)
        a1 = α[i]
        N1_i = N1[i]
        z_i = p.z[i]
        dR[i] = (z_i * dH * grad_R[i] / H) + (p.DR * laplacian_R[i]/hsq) - (1/p.ε)* a1 * N1_i
    end
    dR[1] = 0 #Dirichlet BC
end
@inline function update_dA!(dA, A, grad_A, laplacian_A, N1, N2, dH, H, B, p)
    h_sq = H^2
    inv_H = 1 / H

    @inbounds for i in eachindex(dA)
        z_i = p.z[i]
        A_i = A[i]
        grad_A_i = grad_A[i]
        laplacian_A_i = laplacian_A[i]
        N1_i = N1[i]
        N2_i = N2[i]
        B_i = B[i]

        # Update equation for A
        dA[i] = (z_i * dH * grad_A_i * inv_H) + (p.DA * laplacian_A_i / h_sq) - ((A_i * p.k_cat)/(p.k_m + A_i))*(B_i + (N1_i + N2_i)*p.IB)
    end
end
@inline function update_dB!(dB, H, grad_b, laplacian_B, N1, N2, μ_vec, dH, ψ_vec, p)
    inv_h_sq = 1 / H^2
    inv_H = 1 / H

    @inbounds for i in eachindex(dB)
        z_i = p.z[i]
        grad_B_i = grad_b[i]
        laplacian_B_i = laplacian_B[i]
        N2_i = N2[i]
        μ = μ_vec[i]
        N1_i = N1[i]
        ψ_i = ψ_vec[i]
        # Update equation for A
        dB[i] = (z_i*inv_H*dH*grad_B_i) + (p.DB*laplacian_B_i*inv_h_sq) + (μ * N2_i * p.IB) + (ψ_i *(1-p.η) * N1_i * p.IB)
    end
end
@inline function compute_gradient_central!(grad, vec, dx)
    N = length(vec)
    idx = 1 / dx

    @inbounds begin
        # Interior points using central difference
        for i in 2:N-1
            grad[i] = (vec[i+1] - vec[i-1]) * (0.5 * idx)
        end

        # Boundary points
        grad[1] = (vec[2] - vec[1]) * idx
        grad[N] = (vec[N] - vec[N-1]) * idx
    end
end
@inline function compute_laplacian!(laplacian, R, Δz_sq, R_left)
    # Laplacian computation for R with mixed BCs (Dirichlet at left, Neumann at right)

    N = length(R)
    inv_Δz_sq = 1 / Δz_sq

    @inbounds begin
        # Interior points (first-order central difference)
        for i in 2:N-1
            laplacian[i] = (R[i+1] - 2*R[i] + R[i-1]) * inv_Δz_sq
        end

        # Left boundary fixed with Dirichlet BC (we enforce this way more rigerously in the update_dR! function)
        laplacian[1] = (R[2] - 2*R[1] + R_left) * inv_Δz_sq

        # Right boundary (Neumann BC: zero gradient)
        laplacian[N] = (R[N-1] - R[N]) * inv_Δz_sq
    end
end
@inline function compute_laplacian_Neumann!(laplacian, R, Δz_sq)
    # Laplacian computation for R with Neumann BCs on both ends

    N = length(R)
    inv_Δz_sq = 1 / Δz_sq

    @inbounds begin
        # Interior points (first-order central difference)
        for i in 2:N-1
            laplacian[i] = (R[i+1] - 2*R[i] + R[i-1]) * inv_Δz_sq
        end

        # Left boundary (Neumann BC: zero gradient)
        # Approximation: R[0] ≈ R[1] (ghost cell for Neumann BC)
        laplacian[1] = (R[2] - R[1]) * inv_Δz_sq

        # Right boundary (Neumann BC: zero gradient)
        # Approximation: R[N+1] ≈ R[N] (ghost cell for Neumann BC)
        laplacian[N] = (R[N-1] - R[N]) * inv_Δz_sq
    end
end
function pde_odes!(du, u, p, t)
    # Unpack basic parameters
    N = length(p.grad_N1)                      # Number of spatial points
    # Unpack state variables from u
    N1 = @view u[1:N]            # Biomass concentration (N1)
    R = @view u[N+1:2N]          # Resource concentration (R)
    N2 = @view u[2N+1:3N]          # Dead Cell Mass 
    A = @view u[3N+1:4N]
    B = @view u[4N+1:5N]
    H = u[end]            # Current Height
    # Unpack derivatives
    dN1 = @view du[1:N]          # Time derivative of biomass concentration
    dR = @view du[N+1:2N]        # Time derivative of resource concentration
    dN2 = @view du[2N+1:3N]        # Time derivative of dead cell mass
    dA = @view du[3N+1:4N]
    dB = @view du[4N+1:5N]
    # Allocate workspace arrays to reduce allocations
    SumGi = p.SumGi              # Temporary array
    g1 = p.g1                    # Temporary array for g1(R)
    g2 = p.g2                    # Temporary array for g2(R)
    α = p.α                      # Temporary array for alpha
    β_vec = p.β_vec              # Temporary array for beta
    μ_vec = p.μ_vec              # Temporary array for mu
    ψ_vec = p.ψ_vec              # Temporary array for psi
    RHS_p = p.RHS_p              # RHS of Poisson equation
    grad_p = p.grad_p            # Gradient of pressure
    grad_N1 = p.grad_N1          # Gradient of N1
    term1 = p.term1              # Temporary array for term1 computations
    laplacian_R = p.LaplacianR   # Laplacian of R

    # Compute alpha
    α!(α, R, p.c, p.γ)
    # Compute beta
    β!(β_vec, p.β)
    # Compute mu
    μ!(μ_vec, p.μ)
    # Compute ψ
    ψ!(ψ_vec, α, A, p.n_hill, p.MIC)
    # Compute g1
    g1!(g1, α, β_vec, ψ_vec, N1)
    # Compute g2
    g2!(g2, N1, N2, β_vec, μ_vec, ψ_vec, p.η)
    # Compute SumGi
    SumGi!(SumGi, g1, g2)

    #Update pressure 
    update_pressure!(RHS_p, grad_p, SumGi, H, p)

    # Compute dh/dt 
    dH = -p.λ * grad_p[end] / H
    du[end] = dH

    #Update N1 (use laplacian_R to save on memory)
    update_subrate_N!(N1, grad_N1, g1, dN1, grad_p, term1, SumGi, H, p, dH, laplacian_R)

    #Update N2 
    grad_N2 = grad_N1
    term2 = term1
    update_subrate_N!(N2, grad_N2, g2, dN2, grad_p, term2, SumGi, H, p, dH, laplacian_R)

    # Compute Laplacian of R with mixed boundary conditions
    #compute_laplacian_robin!(laplacian_R, R, p.Δz, p.Δz_sq, p.R_ν, p.R_left, H)
    compute_laplacian!(laplacian_R, R, p.Δz_sq, p.R_left[1])

    #Also need gradient of R 
    grad_R = grad_p # Reuse grad_p
    compute_gradient_central!(grad_R, R, p.Δz)
    # Compute ∂R/∂t
    update_dR!(dR, N1, α, laplacian_R, dH, H, grad_R, p)
    
    # Compute Laplacian of A (reuse vectors no longer needed)
    laplacian_A = laplacian_R
    grad_A = grad_R
    #compute_laplacian_robin!(laplacian_A, A, p.Δz, p.Δz_sq, p.A_ν, p.A_left, H)
    compute_laplacian!(laplacian_A, A, p.Δz_sq, p.A_left[1])

    # Compute gradient of A
    compute_gradient_central!(grad_A, A, p.Δz)

    # Compute ∂A/∂t
    update_dA!(dA, A, grad_A, laplacian_A, N1, N2, dH, H, B, p)

    # Compute gradient of B
    grad_b = grad_R  # Reuse gradient array
    compute_gradient_central!(grad_b, B, p.Δz)

    # Compute Laplacian of B with mixed boundary conditions
    laplacian_B = laplacian_R
    compute_laplacian!(laplacian_B, B, p.Δz_sq, 0) #the left boundary is a sink
    #compute_laplacian_Neumann!(laplacian_B, B, p.Δz_sq) #Bounce back, enzyme cant go through agar
    # Compute ∂B/∂t
    update_dB!(dB, H, grad_b, laplacian_B, N1, N2, μ_vec, dH, ψ_vec, p)
end
function create_callback(h_for_AB, h_for_R, A_new, R_new)
    # Case 1: Both `h_for_AB` and `h_for_R` are `nothing`
    if isnothing(h_for_AB) && isnothing(h_for_R)
        condition1(u, t, integrator) = u[end] <= 0.1
        affect1!(integrator) = terminate!(integrator)
        return DiscreteCallback(condition1, affect1!)

    # Case 2: `h_for_AB` is defined, but `h_for_R` is `nothing`
    elseif !isnothing(h_for_AB) && isnothing(h_for_R)
        function condition2(out, u, t, integrator)
            out[1] = u[end] - 0.1                  # Terminate when height goes below 0.1
            out[2] = h_for_AB - u[end]            # Add antibiotic when height exceeds `h_for_AB`
        end
        function affect2!(integrator, idx)
            if idx == 1
                terminate!(integrator)
            elseif idx == 2
                integrator.p.A_left[1] = A_new    # Add antibiotic
            end
        end
        return VectorContinuousCallback(condition2, affect2!, 2)

    # Case 3: `h_for_AB` is `nothing`, but `h_for_R` is defined
    elseif isnothing(h_for_AB) && !isnothing(h_for_R)
        function condition3(out, u, t, integrator)
            out[1] = u[end] - 0.1                  # Terminate when height goes below 0.1
            out[2] = h_for_R - u[end]             # Add new nutrients when height exceeds `h_for_R`
        end
        function affect3!(integrator, idx)
            if idx == 1
                terminate!(integrator)
            elseif idx == 2
                integrator.p.R_left[1] = R_new    # Add new nutrients
            end
        end
        return VectorContinuousCallback(condition3, affect3!, 2)

    # Case 4: `h_for_AB` equals `h_for_R`
    elseif h_for_AB == h_for_R
        function condition4(out, u, t, integrator)
            out[1] = u[end] - 0.1                  # Terminate when height goes below 0.1
            out[2] = h_for_AB - u[end]            # Add antibiotic and new nutrients at `h_for_AB`
        end
        function affect4!(integrator, idx)
            if idx == 1
                terminate!(integrator)
            elseif idx == 2
                integrator.p.A_left[1] = A_new    # Add antibiotic
                integrator.p.R_left[1] = R_new    # Add new nutrients
            end
        end
        return VectorContinuousCallback(condition4, affect4!, 2)

    # Case 5: `h_for_AB` and `h_for_R` are both defined but different
    else
        function condition5(out, u, t, integrator)
            out[1] = u[end] - 0.1                  # Terminate when height goes below 0.1
            out[2] = h_for_AB - u[end]            # Add antibiotic when height exceeds `h_for_AB`
            out[3] = h_for_R - u[end]             # Add new nutrients when height exceeds `h_for_R`
        end
        function affect5!(integrator, idx)
            if idx == 1
                terminate!(integrator)
            elseif idx == 2
                integrator.p.A_left[1] = A_new    # Add antibiotic
            elseif idx == 3
                integrator.p.R_left[1] = R_new    # Add new nutrients
            end
        end
        return VectorContinuousCallback(condition5, affect5!, 3)
    end
end
function run_model(
    model; 
    printy=false, 
    solver=CVODE_BDF(linear_solver=:LapackDense, max_error_test_failures=12), 
    u0=nothing, 
    h_for_AB=nothing, 
    A_new=0.0,
    h_for_R=nothing,
    R_new=0.0,
    saveatpropmt=0.1)
    # Pre-allocate vectors
    if isnothing(u0)
        N1_0 = ones(Float64, model.N)
        N2_0 = zeros(Float64, model.N)
        R_0 = zeros(Float64, model.N)
        R_0[1] = model.R_left[1]
        A_0 = zeros(Float64, model.N)
        B_0 = zeros(Float64, model.N)
        H_0 = model.starting_height
        u0 = vcat(N1_0, R_0, N2_0, A_0, B_0, H_0)
    end

    # Initialize parameters
    p = model

    # Time span
    tspan = (0.0, model.T_max)

    # ODE Problem
    prob = ODEProblem(pde_odes!, u0, tspan, p)
    cb = create_callback(h_for_AB, h_for_R, A_new, R_new)

    # Solve the ODE problem
    if printy
        @time sol = solve(prob, solver, saveat=saveatpropmt, callback=cb)
        #TRBDF2(autodiff=false), QNDF(autodiff=false), KenCarp4(autodiff=false) all work but may have multithreading issues
        #CVODE_BDF(linear_solver = :LapackDense) seems robust
    else
        sol = solve(prob, solver, callback=cb, save_everystep = false)
    end

    return p, sol
end
function PlotModel(model, sol)
    # Compute the maximum biomass height (H) across all time points
    max_height = maximum([sol[i][end] for i in 1:length(sol.t)])  # Max of H over time

    # Create an animation of the diffusion process
    total_frames = length(sol.t)
    if total_frames <= 200
        frame_indices = 1:total_frames  # Use all frames if less than 200
    else
        frame_indices = round.(Int, range(1, total_frames, length=200))  # Evenly space 200 frames
    end
    
    anim = @animate for i in frame_indices
        # Extract state variables
        N1 = sol[i][1:model.N]               # Live biomass
        R = sol[i][model.N+1:2*model.N]      # Resources
        N2 = sol[i][2*model.N+1:3*model.N]   # Dead biomass
        A = sol[i][3*model.N+1:4*model.N]    # Concentration of A
        B = sol[i][4*model.N+1:5*model.N]    # Concentration of B
        H = sol[i][end]                      # Current height (biomass boundary)
        t = sol.t[i]                         # Current time

        # Map z (scaled domain) to physical domain 
        x = model.z .* H           # Use current height

        # Plot biomass concentration (live biomass)
        p1 = plot(x, N1, label="Live Biomass", lw=2, 
                  xlabel="Height (x)", ylabel="Concentration", 
                  title="Time = $(round(t, digits=2))", color=:green,
                  legend=:topright, ylim=(0, 1.2), xlims=(0, max_height))
        plot!(x, N2, label="Dead Biomass", lw=2, color=:black)
        # Add biomass boundary as a vertical dashed line
        vline!(p1, [H], linestyle=:dash, color=:blue, label="Biomass Boundary")

        # Plot resources and other concentrations
        p2 = plot(x, R, label="Resources", lw=2, color=:red,
                  xlabel="Height (x)", ylabel="Concentration",
                  legend=:topright, ylim=(0, maximum([maximum(R), maximum(A), maximum(B)]) * 1.2), xlims=(0, max_height))
        plot!(x, A, label="Antibiotics", lw=2, color=:purple)
        plot!(x, B, label="Beta Lactamase", lw=2, color=:blue)
        
        # Add biomass boundary to this plot too
        vline!(p2, [H], linestyle=:dash, color=:blue, label="Biomass Boundary")

        # Combine the two plots
        plot(p1, p2, layout=(1, 2), size=(900, 400))
    end

    # Return the animation
    return anim
end
function trapezoidal_integration(x, y)
    n = length(x)
    integral = 0.0
    for i in 1:(n-1)
        integral += 0.5 * (x[i+1] - x[i]) * (y[i] + y[i+1])
    end
    return integral
end
##