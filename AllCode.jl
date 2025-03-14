using Pkg
Pkg.activate("/localdata/Ray/VerticalMathProject/Code")
using Plots, DifferentialEquations
using CSV, DataFrames, Statistics
using Optimization, OptimizationOptimJL
import Zygote
using SciMLSensitivity
using LinearAlgebra
using Base.Math
using Interpolations
using JLD2
using Smoothers
using Measures
using Random
using Statistics
using Distributions
using GLM        
using DataFrames
using StatsBase
include("FluidCode.jl")
translator = Dict(
    "bgt127" => "A. veronii",
    "jt305" => "E. coli",
    "gob33" => "A. cerevisiae (aa)",
    "y55" => "A. cereviaiae (wt)",
    "bh1514" => "V. colerae (wt)",
    "ea387" => "V. cholerae (EPS-)",
    "cc117" => "K. penumoniae",
    "sw520" => "B. cereus",
    "sw519"=>  "S. aureus")
function window_smooth(v::AbstractVector, window_size::Int)
    n = length(v)

    # Preallocate the smoothed vector with the same element type
    smooth_v = similar(v)
    half_window = div(window_size, 2)

    for i in 1:n
        # Determine the window boundaries
        start_idx = max(1, i - half_window)
        end_idx = min(n, i + half_window)
        # Calculate the average over the window
        smooth_v[i] = sum(v[start_idx:end_idx]) / (end_idx - start_idx + 1)
    end

    return smooth_v
end
function fit_to_model(t_data, h_data, ode_func)
    tspan = (t_data[1], t_data[end])
    model = Dict("name"=>"interface",
                        "equation"=>ode_func, 
                        "u0"=>[h_data[1]],
                        "p_guess"=>[0.7, 0.2, 5.0],
                        "p_low"=>[0.2, 0.01, 1],
                        "p_high"=>[1.5, 0.5, 60])

    # Modify the loss function to accept two arguments
    function loss(p, _)
        prob = ODEProblem(model["equation"], model["u0"], tspan, p)
        sol = solve(prob, Tsit5(), saveat=t_data)
        sum(abs,(sol[1,:] .- h_data))
    end

    # Create the optimization problem with bounds
    optf = OptimizationFunction(loss, Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, model["p_guess"], 
                                lb=model["p_low"], 
                                ub=model["p_high"])


    # Solve the optimization problem
        # More rigorous optimization settings
        result = solve(
            optprob, 
            BFGS(), 
            maxiters=5000,  # More iterations
            abstol=1e-10,   # Tighter absolute tolerance
            reltol=1e-10,   # Tighter relative tolerance
            f_tol=1e-10,    # Function tolerance
            g_tol=1e-8,     # Gradient tolerance
            allow_f_increases=false  # Stricter line search
        )
    return result.u
end
function get_end_data()
    df=DataFrame(CSV.File("database.csv"))
    reps=Set(["A","B","C"])
    strains = unique(df[:,:strain])
    df2=deepcopy(df)
    for i in size(df,1):-1:1
        if ! (df[i,:replicate] ∈ reps)
            deleteat!(df2, i)
        end
    end
    end_data = Dict{String, Tuple}()
    for strain_name in strains
        # Filter strain-specific data
        strain_data = filter(row -> row.strain == strain_name, df2)
        # Create a plot for this strain
        cur_data = []  # Collect all replicate heights
        cur_time = []  # Collect all replicate times
        cur_slope_data = []
        cur_slope_error_data = []
        for replicate_id in ["A", "B", "C"]
            # Filter replicate-specific data
            replicate_data = filter(row -> row.replicate == replicate_id, strain_data)
            # Collect data for averaging
            cur_h = Array(replicate_data.smooth_height)
            cur_t = Array(replicate_data.time)
            cur_slope = Array(replicate_data.slope)
            cur_slope_error = Array(replicate_data.slope_error)
            end_data[strain_name*"_"*replicate_id] = (cur_t, cur_h, cur_slope, cur_slope_error)
            push!(cur_data, cur_h)
            push!(cur_time, cur_t)
            push!(cur_slope_data, cur_slope)
            push!(cur_slope_error_data, cur_slope_error)
            # Plot replicate's height over time
        end
        # Calculate averages across replicates
        cur_data_matrix = hcat(cur_data...)  # Combine into a matrix
        cur_time_matrix = hcat(cur_time...)  # Combine into a matrix
        cur_slope_matrix = hcat(cur_slope_data...)  # Combine into a matrix
        cur_slope_error_matrix = hcat(cur_slope_error_data...)  # Combine into a matrix
        avg_height = vec(mean(cur_data_matrix, dims=2))  # Average height across replicates
        avg_time = vec(mean(cur_time_matrix, dims=2))  # Average time across replicates
        avg_slope = vec(mean(cur_slope_matrix, dims=2))  # Average slope across replicates
        avg_slope_error = vec(mean(cur_slope_error_matrix, dims=2))  # Average slope across replicates
        std_h = std(cur_data_matrix, dims=2)[:]

        #avg_height = window_smooth(avg_height, length(avg_height)÷10)
        # Plot the average line
        # Check for negative average heights
        if any(avg_height .< 0)
            println("Issue with $strain_name: Negative average height detected.")
        end
        # Store average data for this strain
        first_index = findfirst(avg_height .> 0)
        avg_height = avg_height[first_index:end]
        avg_time = avg_time[first_index:end]
        avg_slope = avg_slope[first_index:end]
        avg_slope_error = avg_slope_error[first_index:end]
        heigt_error = std_h[first_index:end]
        end_data[strain_name*"_Avg"] = (avg_time, avg_height, avg_slope, avg_slope_error, heigt_error)
    end
    return end_data, strains, reps
end
function interface(du, u, p, t)
    h = u[1] 
    α, β, L = p
    du[1] = α*(h < L ? h : L) - β*h 
end
function analytical_approx(du, u, p, t)
    h = u[1] 
    α, β, L = p
    du[1] = α*L*tanh(h/L) - β*h 
end
function get_fit_parms(end_data, strains)
    fit_data = Dict{String,Tuple}()
    for strain in strains
        t_data, h_data, slope, slope_error = end_data[strain*"_Avg"]
        #itp = interpolate((t_data,), h_data, Gridded(Linear()))
        #new_time = collect(range(minimum(t_data),maximum(t_data),length=length(t_data)))
        #new_height = itp.(new_time)
        fitted_interface = fit_to_model(t_data, h_data, interface)
        fitted_analytical = fit_to_model(t_data, h_data, analytical_approx)
        fit_data[strain] = (fitted_analytical, fitted_interface)
    end
    return fit_data
end
function plot_h_v_t_Aeromonas(end_data, strains, fit_data)
    palette = cgrad(:okabe_ito).colors
    #Aeromonas veronii Intro Plot
    strain_name = strains[1]
    # Filter strain-specific data 
    p = plot(title="", xlabel="Time (hours)", ylabel="Height (μm)",top_margin=6mm)
    # Calculate averages and standard deviations across replicates
    cur_t, avg_h, _, _, std_h = end_data["$(strain_name)_Avg"]
    # Plot the average line
    scatter!(p, cur_t, avg_h, label="Data", color="black")
    # Add error bars (standard deviation)
    plot!(p, cur_t, avg_h, ribbon=std_h, color=:gray, label="", lw=0)

    #now we plot Interface and Analytical models on this 
    fitted_analytical, fitted_interface = fit_data[strain_name]

    t_data, h_data, slope, slope_error = end_data[strains[1]*"_Avg"]
    interface_prob = ODEProblem(interface, [h_data[1]], (minimum(t_data), maximum(t_data)), fitted_interface)
    analytical_approx_prob = ODEProblem(analytical_approx, [h_data[1]], (minimum(t_data), maximum(t_data)), fitted_analytical)
    Isol = solve(interface_prob, TRBDF2(), saveat=t_data)
    Asol = solve(analytical_approx_prob, TRBDF2(), saveat=t_data)
    IhFit = Isol[1,:]
    AhFit = Asol[1,:]
    plot!(p, t_data,IhFit,label="Interface Model",linewidth=3, color=palette[1])
    plot!(p, t_data,AhFit,label="Fluid Model",linewidth=3, color=palette[3])
    #Now show dh_dt v h graph on new plot
    dh_dtIFit = diff(IhFit) ./ diff(t_data)
    dh_dtAFit = diff(AhFit) ./ diff(t_data)
    dh_dtDat = diff(h_data) ./ diff(t_data)
    h_mid = (h_data[1:end-1] + h_data[2:end]) ./ 2
    annotate!(p, 0.5, maximum(AhFit)*1.13, text("A.", :black, 20))
    p2=scatter(h_data,slope,xlabel="Height (μm)",ylabel="Δh (μm/hr)",label="Data",linewidth=3, color=:black,
        ylims=(0,12),legend=false)
    plot!(p2, h_data,slope, ribbon=slope_error, color=:gray, label="", lw=0, top_margin=6mm)
    plot!(p2, h_mid,dh_dtIFit,label="Interface Model",linewidth=3, color=palette[1])
    plot!(p2, h_mid,dh_dtAFit,label="Fluid Model",linewidth=3, color=palette[3])
    annotate!(p2, 0.5, maximum(slope)*1.08, text("B.", :black, 20))
    p3 = plot(p,p2, layout=(1,2), size=(900, 400), left_margin=5mm, bottom_margin=5mm)
    return p3 
end
function mega_h_v_t_plot(end_data, strains, fit_data)
    palette = cgrad(:okabe_ito).colors
    fit_h_v_t = []
    for cur_strain in strains
        fitted_analytical, fitted_interface = fit_data[cur_strain]
        t_data, h_data, _, _, std_h = end_data[cur_strain*"_Avg"]
        analytical_approx_prob = ODEProblem(analytical_approx, [h_data[1]], (minimum(t_data), maximum(t_data)), fitted_analytical)
        Asol = solve(analytical_approx_prob, TRBDF2(), saveat=t_data)
        AhFit = Asol[1,:]

        p=scatter(t_data, h_data,label="Data", color=:black, legend=false, grid=false, title="$(translator[cur_strain])")
        plot!(p, t_data, h_data, ribbon=std_h, color=:gray, label="",lw=0)
        plot!(t_data, AhFit, label="Fluid Model",linewidth=3, color=palette[3])
        if translator[cur_strain] == "B. cereus"
            xlabel!(p, "\nTime (hours)", xlabelfontsize=20)
        end
        if translator[cur_strain] == "A. cereviaiae (wt)"
            ylabel!(p, "Height (μm)\n", ylabelfontsize=20, yguidefontrotation=0)
        end
        push!(fit_h_v_t, p)

    end
    fp = plot(fit_h_v_t..., layout=(3, 3), size=(900, 900), left_margin=5mm,
    bottom_margin=4mm);
    #annotate!(fp, 0.5, 0.1, text("Time (hours)", :black, 18))

    return fp
end
function residual_plot(end_data, strains, fit_data)
    palette = cgrad(:okabe_ito).colors
    #Aeromonas veronii Intro Plot
    interface_residuals = Dict{String, Float64}()
    fluid_residuals = Dict{String, Float64}()
    for strain_name in strains
        # Calculate averages and standard deviations across replicates
        t_data, avg_h, _, _, std_h = end_data["$(strain_name)_Avg"]

        #now we plot Interface and Analytical models on this 
        fitted_analytical, fitted_interface = fit_data[strain_name]

        interface_prob = ODEProblem(interface, [avg_h[1]], (minimum(t_data), maximum(t_data)), fitted_interface)
        analytical_approx_prob = ODEProblem(analytical_approx, [avg_h[1]], (minimum(t_data), maximum(t_data)), fitted_analytical)
        Isol = solve(interface_prob, TRBDF2(), saveat=t_data)
        Asol = solve(analytical_approx_prob, TRBDF2(), saveat=t_data)
        interface_height = Isol[1,:]
        fluid_height = Asol[1,:]
        ##
        residuals_interface = sum(abs2,avg_h .- interface_height)
        residuals_fluid = sum(abs2, avg_h .- fluid_height)
        ##
        interface_residuals[strain_name] = residuals_interface
        fluid_residuals[strain_name] = residuals_fluid
    end
    p = plot(collect(keys(interface_residuals)), collect(values(interface_residuals)), label="Interface Model", linewidth=3, color=palette[1])
    plot!(p, collect(keys(fluid_residuals)), collect(values(fluid_residuals)), label="Fluid Model", linewidth=3, color=palette[3])
    return p
end
function maximum_derivative_height_with_error(t::Vector{<:Real}, h::Vector{<:Real}, sigma::Vector{<:Real}; nsim::Integer=1000)
    n = length(t)
    if length(h) != n || length(sigma) != n
        error("t, h, and sigma must have the same length.")
    elseif n < 2
        error("Need at least 2 data points to compute a derivative.")
    end

    # Preallocate an array to store the h location of max derivative for each simulation.
    h_max_values = zeros(nsim)

    # We'll use a simple first order derivative:
    # dh/dt between t[i] and t[i+1] = (h[i+1] - h[i]) / (t[i+1] - t[i])
    for sim in 1:nsim
        # Generate a perturbed h vector from a normal distribution for each measurement.
        h_perturbed = [rand(Normal(h[i], sigma[i])) for i in 1:n]

        # Compute the derivative via forward finite differences.
        dh_dt = similar(h_perturbed, n-1)
        for i in 1:(n-1)
            dt = t[i+1] - t[i]
            if dt == 0
                error("Time difference is zero at index $i. t must be strictly increasing.")
            end
            dh_dt[i] = (h_perturbed[i+1] - h_perturbed[i]) / dt
        end

        # Identify the index where dh/dt is maximal.
        # This derivative is associated with the segment from i to i+1.
        imax = argmax(dh_dt)

        # The corresponding height is taken as the midpoint between h_perturbed[imax] and h_perturbed[imax+1].
        h_max_values[sim] = (h_perturbed[imax] + h_perturbed[imax+1]) / 2
    end

    # Compute statistics from the Monte Carlo simulations
    h_max_mean = mean(h_max_values)
    h_max_std  = std(h_max_values)

    return h_max_mean, h_max_std
end
function maximum_with_error(t::Vector{<:Real}, y::Vector{<:Real}, sigma::Vector{<:Real}; nsim::Integer=1000)
    n = length(t)
    if length(y) != n || length(sigma) != n
        error("t, y, and sigma must have the same length.")
    elseif n < 1
        error("Input arrays must contain at least one element.")
    end

    # Arrays to store maximum value and its t-location for each simulation.
    max_values = zeros(nsim)
    max_locations = zeros(nsim)

    for sim in 1:nsim
        # Perturb the curve using a normal distribution for each point.
        y_perturbed = [rand(Normal(y[i], sigma[i])) for i in 1:n]

        # Find the index where y_perturbed is maximum.
        imax = argmax(y_perturbed)

        # Record the maximum value and corresponding t-location.
        max_values[sim] = y_perturbed[imax]
        max_locations[sim] = t[imax]
    end

    # Calculate mean and standard deviation for the maximum value and its location.
    max_value_mean = mean(max_values)
    max_value_std  = std(max_values)
    max_location_mean = mean(max_locations)
    max_location_std  = std(max_locations)

    return (
        max_value = (max_value_mean, max_value_std),
        max_location = (max_location_mean, max_location_std)
    )
end
function hstar_exp_model(end_data, strains, fit_data; low=0.8,high=2, res_func=abs)
    palette = cgrad(:okabe_ito).colors
    fit_h_v_t = []
    fit_hdot_v_h = []
    exp_hstar = []
    model_hstar = []
    fluid_L =[]
    residual_data_fluid = zeros(length(strains))
    residual_data_inter = zeros(length(strains))
    i=1
    for cur_strain in strains
        fitted_analytical, fitted_interface = fit_data[cur_strain]
        hstar = fitted_analytical[3]*asech(sqrt(fitted_interface[2]/(fitted_interface[1])))
        t_data, h_data, slope, slope_error, std_h = end_data[cur_strain*"_Avg"]
        #slope, slope_error = smooth_data(slope, slope_error; window=8)
        #h_data = window_smooth(h_data, length(h_data)÷12)
        interface_prob = ODEProblem(interface, [h_data[1]], (minimum(t_data), maximum(t_data)), fitted_interface)
        analytical_approx_prob = ODEProblem(analytical_approx, [h_data[1]], (minimum(t_data), maximum(t_data)), fitted_analytical)
        Isol = solve(interface_prob, TRBDF2(), saveat=t_data)
        Asol = solve(analytical_approx_prob, TRBDF2(), saveat=t_data)
        IhFit = Isol[1,:]
        AhFit = Asol[1,:]
        p=plot(t_data,IhFit,label="Best Fit Interface Model\n α=$(round(fitted_interface[1],digits=3)), β=$(round(fitted_interface[2],digits=3)), L=$(round(fitted_interface[3],digits=3))",linewidth=3)
        plot!(t_data,AhFit,label="Best Fit Analytical Model\n α=$(round(fitted_analytical[1],digits=3)), β=$(round(fitted_analytical[2],digits=3)), L=$(round(fitted_analytical[3],digits=3))",linewidth=3)
        plot!(t_data,h_data,label="Data", linewidth=3, linestyle=:dash)


        push!(fit_h_v_t, p)
        # Get error in hstar from data
        #
        dh_dtAFit = diff(AhFit) ./ diff(t_data)
        #plot(t_mid,dh_dtDat,xlabel="Time (hours)",ylabel="dh/dt (μm/hr)",label="Data",linewidth=3)
        #plot!(t_mid,dh_dtIFit,label="Interface Best Fit",linewidth=3)
        #plot!(t_mid,dh_dtAFit,label="Analytical Best Fit", linewidth=3)

        h_midAFit = (AhFit[1:end-1] + AhFit[2:end]) ./ 2
        ALfit = fitted_analytical[3]
        push!(fluid_L,ALfit)

        p2=plot(h_data,slope, ylabel="dh/dt",xlabel="Height",label="Data",linewidth=3)
        plot!(h_data,slope,ribbon=slope_error, color=:gray, label="", lw=0)
        #vline!([ILfit],linewidth=3,linestyle=:dash,label="Interface L=$(round(ILfit,digits=2))")
        #vline!([ALfit],linewidth=3,linestyle=:dash,label="Analytical L=$(round(ALfit,digits=2))")
        #plot!(h_midIFit,dh_dtIFit,label="Interface Best Fit",linewidth=3)
        plot!(h_midAFit,dh_dtAFit,label="Analytical Best Fit", linewidth=3)
        vline!([hstar],label="Predicted Max dh/dt", color=:black, linewidth=3, linestyle=:dash)
        push!(fit_hdot_v_h, p2)
        obj = maximum_with_error(h_data, slope, slope_error)
        push!(exp_hstar,obj.max_location)
        vline!([obj.max_location[1]],label="Experimental Max dh/dt", color=:grey, linewidth=3, linestyle=:dash)
        push!(model_hstar,hstar)

        L = hstar
        #residuals around L 
        istart,iend = (findlast(h_data .< low*L),findlast(h_data .< high*L))
        analtical_new = AhFit[istart:iend]
        interface_new = IhFit[istart:iend]
        #residuals of both now
        residual_data_fluid[i] = sum(res_func,h_data[istart:iend] .- analtical_new)
        residual_data_inter[i] = sum(res_func,h_data[istart:iend] .- interface_new)
        i+=1
    end
    model_hstar = convert(Vector{Float64}, model_hstar)

    p1=plot(fit_h_v_t..., layout=(3, 3), size=(900, 900))

    p2=plot(fit_hdot_v_h..., layout=(3, 3), size=(900, 900))

    # Extract the means and associated errors from exp_hstar.
    x_means = [x[1] for x in exp_hstar]
    x_errs  = [x[2] for x in exp_hstar]
    # Scatter plot with horizontal error bars.
    p3=scatter(x_means, model_hstar,
        xerror = x_errs,
        xlabel = "Experimental Max Δh (μm/hr)",
        ylabel = "Model Max Δh (μm/hr)",
        label = "",
        color = palette[3],
        markersize = 6,
        top_margin = 5mm)
    annotate!(p3, -7, maximum(model_hstar)*1.06, text("A.", :black, 20))

    # For illustrative purposes, plot a dashed line (e.g., for reference y = x).
    # Adjust the range according to your data.
    plot!(1:0.1:120, 1:0.1:120, linestyle = :dash, color=:black, linewidth = 2, label = "")

    # Linear regression for model_hstar
    model1 = lm(@formula(y ~ x), DataFrame(x=x_means, y=model_hstar))
    r2_model = r²(model1)
    a1, b1 = coef(model1)
    x_range = range(0, 120, length=100)
    plot!(x_range, a1 .+ b1 .* x_range, 
        color=palette[3], 
        linewidth=2,
        label="")

    scatter!(x_means, fluid_L,
        xerror = x_errs,
        label = "", markersize = 6, color=palette[1])

    # Linear regression for fluid_L
    fluid_L = convert(Vector{Float64}, fluid_L)
    model2 = lm(@formula(y ~ x), DataFrame(x=x_means, y=fluid_L))
    r2_fluid = r²(model2)
    a2, b2 = coef(model2)
    plot!(x_range, a2 .+ b2 .* x_range, 
        color=palette[1], 
        linewidth=2,
        label="")
    labels = [translator[x] for x in strains]
    p4=bar(labels, [abs.(model_hstar .- x_means), abs.(fluid_L .- x_means)],
        legend = :topleft,
        ylabel = "Residual",
        label = ["Fluid Model" "Interface Model"],
        alpha=0.3,
        color = [palette[3] palette[1]],
        xrotation=45,
        top_margin = 5mm)
    annotate!(p4, 0.0, maximum(abs.(fluid_L .- x_means))*1.04, text("B.", :black, 20))
    p5 = plot(p3,p4,layout=(1,2),size=(900,400),bottom_margin=10mm,left_margin=5mm)
    good_ones = sum(residual_data_fluid .<residual_data_inter)
    p6 = bar(labels, [residual_data_fluid, residual_data_inter],
        legend = :topleft,
        title = "$low h* < h < $high h* using $res_func, $good_ones improved",
        ylabel = "Residual",
        label = ["Fluid Model" "Interface Model"],
        alpha=0.3,
        color = [palette[3] palette[1]],
        xrotation=45,
        top_margin = 5mm)
    return p1,p2,p3,p4,p5,p6,good_ones
end 
function smooth_data(y::Vector{<:Real}, sigma::Vector{<:Real}; window::Integer=5)
    n = length(y)
    if length(sigma) != n
        error("y and sigma must have the same length.")
    end

    # Preallocate output arrays
    y_smoothed = Vector{Float64}(undef, n)
    sigma_smoothed = Vector{Float64}(undef, n)

    # For a moving average, use equal weights.
    half = div(window, 2)  # floor(window/2)

    for i in 1:n
        # Define a symmetric window; reduce window size at boundaries.
        start_index = max(1, i - half)
        end_index   = min(n, i + half)
        eff_window_length = end_index - start_index + 1

        # In a simple moving average, weights are 1/(effective_window_length)
        weights = fill(1.0 / eff_window_length, eff_window_length)

        # Compute the smooth value.
        window_values = y[start_index:end_index]
        weighted_mean = sum(weights .* window_values)
        y_smoothed[i] = weighted_mean

        # Propagate the error:
        # variance = sum(weights.^2 .* (sigma^2)), then take square-root.
        window_sigma = sigma[start_index:end_index]
        weighted_variance = sum((weights.^2) .* (window_sigma .^ 2))
        sigma_smoothed[i] = sqrt(weighted_variance)
    end

    return y_smoothed, sigma_smoothed
end
function h_star_v_parameters(end_data, strains, fit_data)
    palette = cgrad(:okabe_ito).colors
    model_ba = [(strain,(beta/alpha,L)) for (alpha,beta,L,strain) in [(fit_data[strain][1]...,strain) for strain in strains]]
    exp_hstars = [(strain,maximum_with_error(h_data, slope, slope_error).max_location) for (_, h_data, slope, slope_error, _, strain) in [(end_data[strain*"_Avg"]...,strain) for strain in strains]]
    # Step 1: Convert each vector to a dictionary
    model_ba_dict = Dict(model_ba)
    exp_hstars_dict = Dict(exp_hstars)

    # Step 3: Rebuild the vectors in the consistent order
    aligned_model_ba = [model_ba_dict[strain] for strain in strains]
    aligned_exp_hstars = [exp_hstars_dict[strain] for strain in strains]
    xs = 0.000005:0.0001:0.99
    p = plot(xs,asech.(sqrt.(xs)),label="",linestyle=:dash,color=:black,
    xlabel="β/α", ylabel="h*/L",grid=false, ylims=(0,6.5))
    scatter!([x[1] for x in aligned_model_ba], [aligned_exp_hstars[i][1]/aligned_model_ba[i][2] for i in eachindex(aligned_exp_hstars)], yerror=[aligned_exp_hstars[i][2]/aligned_model_ba[i][2] for i in eachindex(aligned_exp_hstars)],
    color=palette[3], label="",markersize=6,top_margin=7mm)
    annotate!(p, -0.05, 6.8, text("A.", :black, 20))
    return p
end 
function improved_res(end_data, strains, fit_data; low=0.8,high=2, res_func=abs)
    residual_data_fluid = zeros(length(strains))
    residual_data_inter = zeros(length(strains))
    i=1
    for cur_strain in strains
        fitted_analytical, fitted_interface = fit_data[cur_strain]
        hstar = fitted_analytical[3]*asech(sqrt(fitted_interface[2]/(fitted_interface[1])))
        t_data, h_data, slope, slope_error, std_h = end_data[cur_strain*"_Avg"]
        #slope, slope_error = smooth_data(slope, slope_error; window=8)
        #h_data = window_smooth(h_data, length(h_data)÷12)
        interface_prob = ODEProblem(interface, [h_data[1]], (minimum(t_data), maximum(t_data)), fitted_interface)
        analytical_approx_prob = ODEProblem(analytical_approx, [h_data[1]], (minimum(t_data), maximum(t_data)), fitted_analytical)
        Isol = solve(interface_prob, TRBDF2(), saveat=t_data)
        Asol = solve(analytical_approx_prob, TRBDF2(), saveat=t_data)
        IhFit = Isol[1,:]
        AhFit = Asol[1,:]
        # Get error in hstar from data
        #
        L = hstar
        #residuals around L 
        istart,iend = (findlast(h_data .< low*L),findlast(h_data .< high*L))
        analtical_new = AhFit[istart:iend]
        interface_new = IhFit[istart:iend]
        #residuals of both now
        residual_data_fluid[i] = sum(res_func,h_data[istart:iend] .- analtical_new)
        residual_data_inter[i] = sum(res_func,h_data[istart:iend] .- interface_new)
        i+=1
    end
    # Extract the means and associated errors from exp_hstar.
    # Scatter plot with horizontal error bars.
    good_ones = sum(residual_data_fluid .<residual_data_inter)
    return good_ones
end 
function R_over_h_hstar_v_parameters(end_data, strains, fit_data)
    palette = cgrad(:okabe_ito).colors
    model_ba = [(strain,(beta/alpha,L)) for (alpha,beta,L,strain) in [(fit_data[strain][1]...,strain) for strain in strains]]
    exp_hstars = [(strain,maximum_with_error(h_data, slope, slope_error).max_location) for (_, h_data, slope, slope_error, _, strain) in [(end_data[strain*"_Avg"]...,strain) for strain in strains]]
    # Step 1: Convert each vector to a dictionary
    model_ba_dict = Dict(model_ba)
    exp_hstars_dict = Dict(exp_hstars)

    # Step 3: Rebuild the vectors in the consistent order
    aligned_model_ba = [model_ba_dict[strain] for strain in strains]
    aligned_exp_hstars = [exp_hstars_dict[strain] for strain in strains]
    xs = 0.000005:0.0001:0.99
    p = plot(xs,asech.(sqrt.(xs)),label="",linestyle=:dash,color=:black,
    xlabel="β/α", ylabel="h*/L",grid=false, ylims=(0,6.5))
    scatter!([x[1] for x in aligned_model_ba], [aligned_exp_hstars[i][1]/aligned_model_ba[i][2] for i in eachindex(aligned_exp_hstars)], yerror=[aligned_exp_hstars[i][2]/aligned_model_ba[i][2] for i in eachindex(aligned_exp_hstars)],
    color=palette[3], label="",markersize=6,top_margin=7mm)
    ##
    R_curve(z,h,L) = exp(z*h/L)/(exp(2*h/L)+1) + exp(-z*h/L)/(exp(-2*h/L)+1)
    L=10
    h=L*5
    h_range = range(L,h,length=200)
    z = range(0,1,length=1000)
    R_at_L = [R_curve.(z,h_range[i],L)[round(Int,(L/h_range[i])*length(z))] for i in eachindex(h_range)]
    h2L_range = range(2*L,h,length=200)
    R_at_2L = [R_curve.(z,h2L_range[i],L)[round(Int,(2*L/h2L_range[i])*length(z))] for i in eachindex(h2L_range)]

    R2_over_h = ones(length(h_range))

    p2 = plot(h_range ./ L, R_at_L ./ exp(-1),
    xlabel = "Height/L",
    color= palette[3],
    ylabel="Normalized Nutrients at h=L",
    linewidth=4,
    label="Reflective Surface",
    grid=false,
    legendfont=10)
    plot!(p2, h_range ./ L, R2_over_h, label="Non-Reflective Surface", linewidth=4, color=palette[1])
    p3 = plot(p2,p, layout=(1,2), size=(900, 400), left_margin=5mm, bottom_margin=5mm)
    return p3
end
function R_over_h_hstar_v_parameters_nutrients_v_height(end_data, strains, fit_data)
    palette = cgrad(:okabe_ito).colors
    model_ba = [(strain,(beta/alpha,L)) for (alpha,beta,L,strain) in [(fit_data[strain][1]...,strain) for strain in strains]]
    exp_hstars = [(strain,maximum_with_error(h_data, slope, slope_error).max_location) for (_, h_data, slope, slope_error, _, strain) in [(end_data[strain*"_Avg"]...,strain) for strain in strains]]
    # Step 1: Convert each vector to a dictionary
    model_ba_dict = Dict(model_ba)
    exp_hstars_dict = Dict(exp_hstars)

    # Step 3: Rebuild the vectors in the consistent order
    aligned_model_ba = [model_ba_dict[strain] for strain in strains]
    aligned_exp_hstars = [exp_hstars_dict[strain] for strain in strains]
    xs = 0.000005:0.0001:0.99
    p = plot(xs,asech.(sqrt.(xs)),label="",linestyle=:dash,color=:black,
    xlabel="β/α", ylabel="h*/L",grid=false, ylims=(0,6.5))
    scatter!([x[1] for x in aligned_model_ba], [aligned_exp_hstars[i][1]/aligned_model_ba[i][2] for i in eachindex(aligned_exp_hstars)], yerror=[aligned_exp_hstars[i][2]/aligned_model_ba[i][2] for i in eachindex(aligned_exp_hstars)],
    color=palette[3], label="",markersize=6,top_margin=7mm)
    annotate!(p, 0.01, 6.8, text("B.", :black, 20))
    ##
    R_curve(z,h,L) = exp(z*h/L)/(exp(2*h/L)+1) + exp(-z*h/L)/(exp(-2*h/L)+1)
    z = range(0,1,length=1000)
    L=10
    h=L*5
    h_range = range(L,h,length=200)
    minz = 1
    R_at_L = [R_curve.(z,h_range[i],L)[round(Int,(L/h_range[i])*length(z))] for i in eachindex(h_range)]

    p2 = plot(h_range ./ L, R_at_L ./ exp(-1),
    xlabel = "Height/L",
    color= palette[3],
    ylabel="Normalized Nutrients Level",
    linewidth=4,
    label="z=L",
    grid=false,
    legendfont=10)
    new_L_factors = [0.75,1.25]
    minz = minimum(new_L_factors)
    color_index = [2,5]
    i=1
    for new_L_factor in new_L_factors
        new_h_range = range(new_L_factor*L,h,length=200)
        R_at_2L = [R_curve.(z,new_h_range[i],L)[round(Int,(new_L_factor*L/new_h_range[i])*length(z))] for i in eachindex(new_h_range)]
        plot!(p2, new_h_range ./ L, R_at_2L ./ exp(-1), label="z=$(new_L_factor)L", linewidth=4, color=palette[color_index[i]])
        i+=1
    end
    dashed_line = ones(length(h_range))
    xs = range(minz*L,h,length=length(dashed_line))
    plot!(p2, xs ./ L, dashed_line, label="", linewidth=4, color=:black, linestyle=:dash)
    annotate!(p2, 0.5, 2.2, text("A.", :black, 20))
    p3 = plot(p2,p, layout=(1,2), size=(900, 400), left_margin=5mm, bottom_margin=5mm)
    return p3
end
function decaying_boundaries()
    palette = cgrad(:okabe_ito).colors
    h_over_L_range = range(0,3,length=2000)
    pos_term = 1 ./ (exp.(2 .* h_over_L_range) .+ 1)
    neg_term = 1 ./ (exp.(-2 .* h_over_L_range) .+ 1)
    p= plot(h_over_L_range, neg_term, color=palette[8], label="C₂", linewidth=4)
    plot!(h_over_L_range, pos_term, color=palette[2], label="C₁", linewidth=4, grid=false,
    xlabel="Height/L", ylabel="Normalized Nutrients Level", legendfont=12)
    R_curve(z,h,L) = exp(z*h/L)/(exp(2*h/L)+1) + exp(-z*h/L)/(exp(-2*h/L)+1)
    L = 1

    return p
end
#end_data, strains, reps = get_end_data()
#JLD2.@save "End_data.jld2" end_data strains reps
JLD2.@load "End_data.jld2" end_data strains reps

#fit_data = get_fit_parms(end_data, strains)
#JLD2.@save "Fit_data.jld2" fit_data
JLD2.@load "Fit_data.jld2" fit_data

p = plot_h_v_t_Aeromonas(end_data, strains, fit_data);
display(p)
savefig(p, "Fig1.pdf")

p = decaying_boundaries();
display(p)
savefig(p, "Fig2B.svg")

p = mega_h_v_t_plot(end_data, strains, fit_data);
display(p)
savefig(p, "Fig3.pdf")

p1,p2,p3,p4,p5,p6,_ = hstar_exp_model(end_data, strains, fit_data; low=0.5,high=2.0)
display(p6)
display(p5)
savefig(p5, "Fig4.pdf")
display(p2)
#

#
p = R_over_h_hstar_v_parameters_nutrients_v_height(end_data, strains, fit_data);
display(p)
savefig(p, "Fig5.pdf")