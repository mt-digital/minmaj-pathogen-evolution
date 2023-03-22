##
# Agent-based SIR model of infectious disease spread where the pathogen's
# virulence (represented here by recovery rate) evolves to an optimal value
# for the given group structure.

# I drew on Simon Frost's Agents.jl SIR model as a helpful example 
# (https://github.com/epirecipes/sir-julia/blob/master/script/abm/abm.jl)
# for the SIR portion of this model. 

# I combined an SIR model with code I wrote for understanding the spread of
# adaptations in metapopulations, currently at 
# https://github.com/eehh-stanford/SustainableCBA/blob/main/src/model.jl. 
# But here we have disease transmission instead of social learning as 
# in SustainbleCBA.

# Author: Matthew A. Turner <maturner01@gmail.com>
# Date: March 21, 2023
#
using Agents
using Distributions
using Random
using StatsBase
# using Statistics


"""
Agents can be in Minority or Majority. `Both` is used for initializing 
both populations with a pathogen.
"""
@enum Group Minority Majority Both


"Classic SIR infectious disease model states"
# @enum SIR_Status Susceptible Infected Recovered Dead
@enum SIR_Status Susceptible Infected Dead

# mutable struct Pathogen <: AbstractAgent  <-- not sure we need these to be Agents.
"""
Pathogen agents only hold their virulence; we assume a virulence-transmissibility
tradeoff where increased transmissibility comes from infected agents being 
infected longer (lower recovery rate), at the risk of increased mortality
rate for the pathogen–the mortality rate is calculated using the heuristic that
virulence + mortality_rate = 1, so if the recovery rate is 0.6 the mortality
rate is 0.4, i.e., there is a 60% chance an infected Person recovers on a 
given time step, and if they don't recover, there is a 40% chance the 
infected Person dies. 
"""
struct Pathogen
    # recovery_rate::Float64
    virulence::Float64
end


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

    virulence = copy(focal_agent.infected_by.virulence)

    # Possibly get infected if not infected...
    if focal_agent.status == Susceptible
        interact!(focal_agent, model)
    # ...or possibly die if infected
    elseif (focal_agent.status == Infected) &&
           (rand() < mortality_rate(virulence))
        
        focal_agent.status = Dead
    end

    # Possibly recover if infected...
    # if (focal_agent.status == Infected) && (rand() < virulence)

    #     if rand() < virulence
    #         focal_agent.status = Recovered
    #     elseif rand() < mortality_rate(virulence)
    #         remove_agent!(focal_agent, model) 
    #     end

    # elseif focal_agent.status == Susceptible

    #     # Select interaction partner, interact, possibly get infected.
    #     interact!(focal_agent, model)

    # end
end


function model_step!(model)

    # Possibly birth a new agent in one of the groups, weighted by minority_fraction param.
    # if rand() < model.global_birth_rate
    #     for _ in 1:10
    #         birth_new_agent!(model)
    #     end
    # end

    # Possibly remove random agent, which group weighted by minority_fraction param.
    if rand() < model.global_death_rate
        for _ in 1:10
            random_agent_dieoff!(model)
        end
    end
end


# function birth_new_agent!(model)

#     if rand() < model.minority_fraction
#         group = Minority
#         homophily = model.homophily_min
#     else
#         group = Majority
#         homophily = model.homophily_maj
#     end
    
#     add_agent!(
#         Person(nextid(model), group, homophily, Susceptible, Pathogen(NaN)), 
#         model
#     )

# end


function random_agent_dieoff!(model)

    if rand() < model.minority_fraction
        group = Minority
    else
        group = Majority
    end

    possibly_remove_agents = filter(agent -> agent.group == group, collect(allagents(model)))

    if isempty(possibly_remove_agents)
        if group == Minority
            group = Majority
        else
            group = Minority
        end
        possibly_remove_agents = filter(agent -> agent.group == group, collect(allagents(model)))
    end

    agent_to_remove = sample(possibly_remove_agents)

    # remove_agent!(agent_to_remove, model) 
    agent_to_remove.status = Dead
end


"""
Virulence tradeoff: more virulent means lower recovery rate, higher death rate.
"""
function mortality_rate(virulence::Float64; c = 0.05)
    # For now use linear virulence-mortality relationship.
    return c * virulence
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
    g_weights[1:end .!= group_idx] .= 1 - in_group_weight

    partner_group = sample([Minority, Majority], Weights(g_weights))

    # Select partner at random from partner group.

    partner_group_agents = filter(
            agent -> (agent.group == partner_group) && (agent != focal_agent),
            collect(allagents(model))
        )

    if isempty(partner_group_agents)
        
        if partner_group == Minority
            partner_group = Majority
        else
            partner_group = Minority
        end

        partner_group_agents = filter(
                agent -> (agent.group == partner_group) && (agent != focal_agent),
                collect(allagents(model))
            )
    end

    partner = sample(partner_group_agents)

    # Possibly get infected.
    if (partner.status == Infected) && 
       (rand() ≤ transmissibility(partner.infected_by.virulence))

        focal_agent.status = Infected 

        # Without mutation, the infection has the same recovery rate as partner's.
        virulence = copy(partner.infected_by.virulence)

        # The pathogen evolves 
        if rand() < model.mutation_rate
            virulence += rand(model.mutation_dist)

            if virulence < 0.0
                virulence = 0.0
            elseif virulence > 1.0
                virulence = 1.0
            end
        end

        transmitted_pathogen = Pathogen(virulence)
        focal_agent.infected_by = transmitted_pathogen

        model.total_infected += 1
    end
end


function minmaj_evoid_model(;metapop_size = 100, minority_fraction = 0.5, 
                             homophily_min = 0.0, homophily_maj = 0.0, 
                             group_zero = Majority, virulence_init = 0.3,
                             initial_infected_frac = 0.10, mutation_rate = 0.05,
                             mutation_variance = 0.05, global_death_rate = 1.0)
    
    in_group_freq_min = (1 + homophily_min) / 2.0
    in_group_freq_maj = (1 + homophily_maj) / 2.0

    # Mutations are drawn from normal distros with zero mean and given variance.
    mutation_dist = Normal(0.0, mutation_variance)

    # Track total number of infections over time.
    total_infected::Int = 0

    properties = @dict(metapop_size, minority_fraction, homophily_min, 
                       homophily_maj, group_zero, 
                       virulence_init, initial_infected_frac,
                       in_group_freq_min, in_group_freq_maj, mutation_rate, 
                       mutation_dist, global_death_rate, total_infected)

    model = UnremovableABM(Person; properties)
    initialize_metapopulation!(model)
    
    return model
end


"""
Given a minority fraction; population size; group-level homophily values; which
group starts with the infection, and the ABM where the agents will live, create
and add the appropriate number of agents from each group to the population.
"""
function initialize_metapopulation!(model::ABM)
        
    virulence_init = model.properties[:virulence_init]
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
                (agent_idx ≤ initial_infected_count))
                
                status = Infected
                pathogen = Pathogen(virulence_init)
            else
                status = Susceptible
                pathogen = Pathogen(NaN)
            end

        else

            group = Majority
            homophily = homophily_maj

            if (((group_zero == Majority) || (group_zero == Both)) 
                && (agent_idx ≤ minority_pop_size + initial_infected_count)) 
                
                status = Infected
                pathogen = Pathogen(virulence_init)
            else
                status = Susceptible
                pathogen = Pathogen(NaN)
            end
        end

        add_agent!(
            Person(agent_idx, group, homophily, status, pathogen), model
        )
    end
end


function transmissibility(virulence; a = 1.0, b = 10.0)
    return (a * virulence) / (b + virulence)
end
