using CSV
using DataFrames
using RCall


function plot_series(agent_df)
    if !isdir("tmp")
        mkdir("tmp")
    end

    CSV.write("tmp/series.csv", agent_df)
    # TODO call R plotting routine.
    R"""
    source("scripts/plot.R")

    plot_series("tmp/series.csv")
    """
end


function run_series(plot = true; model_params...)
    
    susceptible(x) = isempty(x) ? 0.0 : count(i == Susceptible for i in x) 
    infected(x) = isempty(x) ? 0.0 : count(i == Infected for i in x) 
    recovered(x) = isempty(x) ? 0.0 : count(i == Recovered for i in x) 

    adata = adata = [(:status, f) for f in (susceptible, infected, recovered)]

    m = minmaj_evoid_model(metapop_size = 100, group_zero = Both, 
                           transmissibility = 0.6, recovery_rate_init = 0.3, 
                           mutation_rate = 0.0,  initial_infected_frac = 0.1, 
                           global_birth_rate=0.75, global_death_rate=0.5); 

    agent_df, _ = run!(m, agent_step!, model_step!, stopfn; adata); print(adf)

    rename!(agent_df, :susceptible_status => :susceptible,
                      :infected_status => :infected,
                      :recovered_status => :recovered)

    if plot
        plot_series(agent_df)
    end

    return adf
end
