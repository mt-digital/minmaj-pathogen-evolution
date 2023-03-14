##
# Agent-based SIR model of infectious disease spread where the pathogen's
# virulence (represented here by recovery rate) evolves to an optimal value
# for the given group structure.
#
# Simon Frost's Agents.jl SIR model served as a helpful example which 
# I drew on for the SIR portion of this model 
# (https://github.com/epirecipes/sir-julia/blob/master/script/abm/abm.jl). I
# combined an SIR model with code I wrote for understanding the spread of
# adaptations in metapopulations, currently at 
# https://github.com/eehh-stanford/SustainableCBA/blob/main/src/model.jl. 
# But here we have disease transmission instead of social learning as 
# in SustainbleCBA.
#
# Author: Matthew A. Turner <maturner01@gmail.com>
# Date: March 13, 2023
#
using Agents
using Distributions
using Random



"""
Agents can be in Minority or Majority. `Both` is used for initializing 
both populations with a pathogen.
"""
@enum Group Minority Majority Both


"Classic SIR infectious disease model states"
@enum SIR_Status Susceptible Infected Recovered


mutable struct Person <: AbstractAgent
    
    id::Int

    group::Group
    homophily::Float64

    status::SIR_Status

    # Each agent is infected by a unique pathogen with a certain virulence.
    infected_by::Pathogen
end



"""
Agent steps will use people; pathogens evolve on transmission,
within this agent_step!, for simplicity.
"""
function agent_step!(focal_agent::Person, model::ABM)

    recovery_rate = focal_agent.infected_by.recovery_rate

    if focal_agent.status == Susceptible

        # Select interaction partner, interact, possibly get infected.
        interact!(focal_agent, model)

    # Possibly recover...
    elseif (focal_agent.status == Infected) && (rand() < recovery_rate)

        focal_agent.status = Recovered

    # ...or die if recovery fails.
    elseif (focal_agent.status == Infected) || (rand() < mortality_rate(recovery_rate))

        remove_agent!(focal_agent, model) 

    end
end


function model_step!(model)

    # Possibly birth a new agent in one of the groups, weighted by minority_fraction param.
    if rand() < global_birth_rate
        birth_new_agent!(model)
    end

    # Possibly remove random agent, which group weighted by minority_fraction param.
    if rand() < global_death_rate
        random_agent_dieoff!(model)
    end
end


function birth_new_agent!(model)
    # TODO
end


function random_agent_dieoff!(model)
    # TODO
end


"""
Virulence-transmissibility tradeoff: more virulent 
"""
function mortality_rate(recovery_rate::Float64)
    return 1.0 - recovery_rate
end


"""
Select interaction partner, interact, possibly get infected, and if infection
happens, let the pathogen evolve.
"""
function interact!(focal_agent, model)
    ## First, select group to interact with...
    g_weights = zeros(2)
    in_group_weight = focal_agent.group == Minority ? 
                      model.in_group_freq_min :
                      model.in_group_freq_maj

    # Use index of Group enum, needs shifting by 1 to be in num. array bounds.
    group_idx = Int(focal_agent.group) + 1                                       
    g_weights[group_idx] = in_group_weight
    g_weights[1:end .!= group_idx] = 1 - in_group_weight

    partner_group = sample([Minority, Majority], Weights(g_weights))

    # Select partner at random from partner group.
    partner = sample(
        filter(
            agent -> (agent.group == partner_group) && (agent != focal_agent),
            allagents(model)
        )
    )

    # Possibly get infected.
    if (partner.status == Infected) && (rand() ≤ model.transmissibility)

        focal_agent.status = Infected 

        # Without mutation, the infection has the same recovery rate as partner's.
        recovery_rate = partner.infected_by.recovery_rate

        # The pathogen evolves 
        if rand() < model.mutation_rate
            recovery_rate += rand(model.mutation_dist)

            if recovery_rate < 0.0
                recovery_rate = 0.0
            elseif recovery_rate > 1.0
                recovery_rate = 1.0
            end
        end

        transmitted_pathogen = Pathogen(recovery_rate)
        focal_agent.infected_by = transmitted_pathogen
    end

    # If getting infected, the Pathogen may evolve.
    # ...TODO...
end


# mutable struct Pathogen <: AbstractAgent  <-- not sure we need these to be Agents.
"""
Pathogen agents only hold their recovery_rate; we assume a virulence-transmissibility
tradeoff where increased transmissibility comes from infected agents being 
infected longer (lower recovery rate), at the risk of increased mortality
rate for the pathogen–the mortality rate is calculated using the heuristic that
recovery_rate + mortality_rate = 1, so if the recovery rate is 0.6 the mortality
rate is 0.4, i.e., there is a 60% chance an infected Person recovers on a 
given time step, and if they don't recover, there is a 40% chance the 
infected Person dies. 
"""
struct Pathogen
    recovery_rate::Float64
end


function minmaj_evoid_model(;metapop_size = 100, minority_fraction = 0.5, 
                             homophily_min = 0.0, homophily_maj = 0.0, 
                             group_zero = Majority, recovery_rate_init = 0.95,
                             initial_infected_frac = 0.10, mutation_rate = 0.05,
                             mutation_variance = 0.05, transmissibility = 0.5,
                             global_birth_rate = 0.1, global_death_rate = 0.1)
    
    in_group_freq_min = (1 + homophily_min) / 2.0
    in_group_freq_maj = (1 + homophily_maj) / 2.0

    # Mutations are drawn from normal distros with zero mean and given variance.
    mutation_dist = Normal(0.0, mutation_variance)

    properties = @dict(metapop_size, minority_fraction, homophily_min, homophily_maj,
                       group_zero, recovery_rate_init, initial_infected_frac,
                       in_group_freq_min, in_group_freq_maj, mutation_dist,
                       global_birth_rate, global_death_rate)

    model = ABM(Person; properties)
    initialize_metapopulation!(model)
    
    return model
end


"""
Given a minority fraction; population size; group-level homophily values; which
group starts with the infection, and the ABM where the agents will live, create
and add the appropriate number of agents from each group to the population.
"""
function initialize_metapopulation!(model::ABM)
        
    recovery_rate_init = model.properties[:recovery_rate_init]
    metapop_size = model.properties[:metapop_size]
    minority_fraction = model.properties[:minority_fraction]
    homophily_min  = model.properties[:homophily_min]
    homophily_maj = model.properties[:homophily_maj]
    group_zero = model.properties[:group_zero]
    initial_infected_frac = model.properties[:initial_infected_frac]

    minority_pop_size = Int(ceil(minority_fraction * metapop_size))
    initial_infected_count = ceil(initial_infected_frac * metapop_size)
    if group_zero == Both
        initial_infected_count /= 2.0
    end

    for agent_idx in 1:metapop_size

        # Initialize a minority group agent
        if agent_idx ≤ minority_pop_size

            group = Minority
            homophily = homophily_min

            if (((group_zero == Minority) || (group_zero == Both)) &&
                (agent_idx ≤ ))
                
                status = Infected
                pathogen = Pathogen(recovery_rate_init)
            else
                status = Susceptible
                pathogen = nothing
            end

        else

            group = Majority
            homophily = homophily_maj

            if (((group_zero == Majority) || (group_zero == Both)) 
                && (agent_idx == minority_pop_size + 1)) 
                
                status = Infected
                pathogen = Pathogen(recovery_rate_init)
            else
                status = Susceptible
                pathogen = nothing
            end
        end

        add_agent!(Person(agent_idx, group, homophily, status, pathogen), model)

    end

end


