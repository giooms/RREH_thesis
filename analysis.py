import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import pandas as pd
import os
import argparse
import numpy as np
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze GBOML model results with specific criteria.")
    # parser.add_argument('-s', '--scenarios', nargs='+', help='Scenarios to analyze (e.g., hydro wind_onshore)', default=None)
    parser.add_argument('-t', '--timehorizon', type=int, help='Time horizon to filter by', default=17544)
    parser.add_argument('-r', '--report', action='store_true', help='Save as PDF without title for reports')
    return parser.parse_args()

def load_results(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return MakeMeReadable(data)

def plot_balance(ax, data, title):
    ylim = (-0.1, 0.1)

    ax.plot(data)
    ax.set_ylim(ylim)
    ax.set_title(title)

def extract_base_scenario(scenario):
    if '_stdwacc' in scenario:
        return scenario.split('_stdwacc')[0]
    elif '_varwacc' in scenario:
        return scenario.split('_varwacc')[0]
    return scenario

def compute_and_plot_balances(scenario, d, results_path, timehorizon):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 15), constrained_layout=True)
    axs = axs.flatten()  # Ensuring axs is a 1D array for easy indexing
    
    # Generalized approach to handle variations in power_balance_gr
    power_balances = {
        'wind_offshore': lambda d: np.array(d.solution.elements.OFF_WIND_PLANTS_GR.variables.electricity.values),
        'wind_onshore': lambda d: np.array(d.solution.elements.ON_WIND_PLANTS_GR.variables.electricity.values),
        'hydro': lambda d: np.array(d.solution.elements.HYDRO_PLANT_03h_GR.variables.electricity.values) +
                           np.array(d.solution.elements.HYDRO_PLANT_03j_GR.variables.electricity.values) +
                           np.array(d.solution.elements.HYDRO_PLANT_05h_GR.variables.electricity.values),
        'wave': lambda d: np.array(d.solution.elements.WAVE_PLANT_GR.variables.electricity.values),
        'combined': lambda d: np.array(d.solution.elements.OFF_WIND_PLANTS_GR.variables.electricity.values) +
                              np.array(d.solution.elements.ON_WIND_PLANTS_GR.variables.electricity.values) +
                              np.array(d.solution.elements.HYDRO_PLANT_03h_GR.variables.electricity.values) +
                              np.array(d.solution.elements.HYDRO_PLANT_03j_GR.variables.electricity.values) +
                              np.array(d.solution.elements.HYDRO_PLANT_05h_GR.variables.electricity.values) + 
                              np.array(d.solution.elements.WAVE_PLANT_GR.variables.electricity.values)
    }

    battery_storage_gr_out = np.array(d.solution.elements.BATTERY_STORAGE_GR.variables.electricity_out.values)
    battery_storage_gr_in = np.array(d.solution.elements.BATTERY_STORAGE_GR.variables.electricity_in.values)
    hvdc_gr_in = np.array(d.solution.elements.HVDC_GR.variables.electricity_in.values)

    base_scenario = extract_base_scenario(scenario)

    # Compute and plot power balance for the scenario
    if base_scenario in power_balances:
        power_balance_gr = power_balances[base_scenario](d) + battery_storage_gr_out - battery_storage_gr_in - hvdc_gr_in
        plot_balance(axs[0], power_balance_gr, f'{scenario} GR_Inland Power Balance')

    # GR_CO2 Balance
    co2_storage_gr_in = np.array(d.solution.elements.CARBON_DIOXIDE_STORAGE_GR.variables.carbon_dioxide_in.values)
    co2_storage_gr_out = np.array(d.solution.elements.CARBON_DIOXIDE_STORAGE_GR.variables.carbon_dioxide_out.values)
    dac_gr = np.array(d.solution.elements.DIRECT_AIR_CAPTURE_PLANTS_GR.variables.carbon_dioxide.values)
    methanation_gr = np.array(d.solution.elements.METHANATION_PLANTS_GR.variables.carbon_dioxide.values)
    co2_balance_gr = dac_gr + co2_storage_gr_out - co2_storage_gr_in - methanation_gr

    plot_balance(axs[1], co2_balance_gr, 'GR_CO2 Balance')

    # GR_LCH4 Balance
    lch4_carriers_gr = np.array(d.solution.elements.LIQUEFIED_METHANE_CARRIERS_GR.variables.liquefied_methane_in.values)
    lch4_storage_gr_in = np.array(d.solution.elements.LIQUEFIED_METHANE_STORAGE_HUB_GR.variables.liquefied_methane_in.values)
    lch4_storage_gr_out = np.array(d.solution.elements.LIQUEFIED_METHANE_STORAGE_HUB_GR.variables.liquefied_methane_out.values)
    ch4_liquefaction_gr = np.array(d.solution.elements.METHANE_LIQUEFACTION_PLANTS_GR.variables.liquefied_methane.values)
    lch4_balance_gr = ch4_liquefaction_gr + lch4_storage_gr_out - lch4_storage_gr_in - lch4_carriers_gr

    plot_balance(axs[2], lch4_balance_gr, 'GR_LCH4 Balance')

    # Check BE_CH4 Balance
    conversion_factor = d.model.hyperedges.DESTINATION_METHANE_BALANCE.parameters.conversion_factor
    demand_CH4 = np.array(d.model.hyperedges.DESTINATION_METHANE_BALANCE.parameters.demand)
    lch4_regasification_be = np.array(d.solution.elements.LIQUEFIED_METHANE_REGASIFICATION.variables.methane.values)
    ch4_balance_be = conversion_factor * lch4_regasification_be - 10 * demand_CH4[:timehorizon]

    plot_balance(axs[3], ch4_balance_be, 'BE_CH4 Balance')

    # Check BE_LCH4 Balance
    lch4_carriers_gr = np.array(d.solution.elements.LIQUEFIED_METHANE_CARRIERS_GR.variables.liquefied_methane_out.values)
    lch4_storage_dest_in = np.array(d.solution.elements.LIQUEFIED_METHANE_STORAGE_DESTINATION.variables.liquefied_methane_in.values)
    lch4_storage_dest_out = np.array(d.solution.elements.LIQUEFIED_METHANE_STORAGE_DESTINATION.variables.liquefied_methane_out.values)
    lch4_regasification_be = np.array(d.solution.elements.LIQUEFIED_METHANE_REGASIFICATION.variables.liquefied_methane.values)
    lch4_balance_be = lch4_carriers_gr + lch4_storage_dest_out - lch4_storage_dest_in - lch4_regasification_be

    plot_balance(axs[4], lch4_balance_be, 'BE_LCH4 Balance')

    # Check GR_Coastal Power Balance
    hvdc_gr_out = np.array(d.solution.elements.HVDC_GR.variables.electricity_out.values)
    electrolysis_gr = np.array(d.solution.elements.ELECTROLYSIS_PLANTS_GR.variables.electricity.values)
    h2_storage_gr = np.array(d.solution.elements.HYDROGEN_STORAGE_GR.variables.electricity.values)
    desalination_gr = np.array(d.solution.elements.DESALINATION_PLANTS_GR.variables.electricity.values)
    h2o_storage_gr = np.array(d.solution.elements.WATER_STORAGE_GR.variables.electricity.values)
    co2_storage_gr = np.array(d.solution.elements.CARBON_DIOXIDE_STORAGE_GR.variables.electricity.values)
    ch4_liquefaction_gr = np.array(d.solution.elements.METHANE_LIQUEFACTION_PLANTS_GR.variables.electricity.values)
    dac_gr = np.array(d.solution.elements.DIRECT_AIR_CAPTURE_PLANTS_GR.variables.electricity.values)
    power_balance_coast_gr = hvdc_gr_out - electrolysis_gr - h2_storage_gr - desalination_gr - h2o_storage_gr - co2_storage_gr - ch4_liquefaction_gr - dac_gr 

    plot_balance(axs[5], power_balance_coast_gr, 'GR_Coastal Power Balance')

    # Check GR_Coastal H2 Balance
    electrolysis_gr = np.array(d.solution.elements.ELECTROLYSIS_PLANTS_GR.variables.hydrogen.values)
    h2_storage_gr_out = np.array(d.solution.elements.HYDROGEN_STORAGE_GR.variables.hydrogen_out.values)
    h2_storage_gr_in = np.array(d.solution.elements.HYDROGEN_STORAGE_GR.variables.hydrogen_in.values)
    methanation_gr = np.array(d.solution.elements.METHANATION_PLANTS_GR.variables.hydrogen.values)
    dac_gr = np.array(d.solution.elements.DIRECT_AIR_CAPTURE_PLANTS_GR.variables.hydrogen.values)
    h2_balance_gr = electrolysis_gr + h2_storage_gr_out - h2_storage_gr_in - methanation_gr - dac_gr

    plot_balance(axs[6], h2_balance_gr, 'GR_Coastal H2 Balance')

    # Check GR_Coastal H2O Balance
    desalination_gr = np.array(d.solution.elements.DESALINATION_PLANTS_GR.variables.water.values)
    methanation_gr = np.array(d.solution.elements.METHANATION_PLANTS_GR.variables.water.values)
    h2o_storage_gr_out = np.array(d.solution.elements.WATER_STORAGE_GR.variables.water_out.values)
    h2o_storage_gr_in = np.array(d.solution.elements.WATER_STORAGE_GR.variables.water_in.values)
    electrolysis_gr = np.array(d.solution.elements.ELECTROLYSIS_PLANTS_GR.variables.water.values)
    dac_gr = np.array(d.solution.elements.DIRECT_AIR_CAPTURE_PLANTS_GR.variables.water.values)
    h20_balance_gr = desalination_gr + methanation_gr + h2o_storage_gr_out - h2o_storage_gr_in - electrolysis_gr - dac_gr

    plot_balance(axs[7], h20_balance_gr, 'GR_Coastal H2O Balance')

    # Check GR_Coastal CH4 Balance
    methanation_gr = np.array(d.solution.elements.METHANATION_PLANTS_GR.variables.methane.values)
    ch4_liquefaction_gr = np.array(d.solution.elements.METHANE_LIQUEFACTION_PLANTS_GR.variables.liquefied_methane.values)
    ch4_balance_gr = methanation_gr - ch4_liquefaction_gr

    plot_balance(axs[8], ch4_balance_gr, 'GR_Coastal CH4 Balance')
    
    img_folder_name = f"img_{timehorizon}/{scenario}" if timehorizon else "img_all"
    
    # Create the img_{timehorizon} folder inside the results_path if it doesn't exist
    img_folder_path = os.path.join(results_path, img_folder_name)
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    
    # Define the save path for the plot within the newly created folder
    save_filename = f"{scenario}_{timehorizon}_balance_plots.png"  # Or include more details in the name as needed
    fig_save_path = os.path.join(img_folder_path, save_filename)
    
    # Save the figure
    fig.savefig(fig_save_path)
    print(f"Saved balance plots to {fig_save_path}")
    
    plt.close(fig)

def cost_rreh(d, ls_nodes):
    if not ls_nodes:
        return 0.0

    cost = 0
    for ele in ls_nodes:
        element = getattr(d.solution.elements, ele, None)
        if element is not None:
            try:
                # Adjusted to use attribute access instead of dictionary access
                cost += np.sum(element.objectives.unnamed)
            except AttributeError:
                try:
                    # If 'unnamed' objectives do not exist, sum up 'named' objectives
                    for e in element.objectives.named.values():
                        cost += e
                except AttributeError:
                    # If both attempts fail, it means the expected structure isn't present
                    print(f"Warning: Objectives not found for {ele}")
        else:
            print(f"Warning: Element {ele} not found in solution")
    
    return cost

def filter_nodes(ls_nodes, suffix, exclude_patterns=None):
    filtered_nodes = [node for node in ls_nodes if node.endswith(suffix)]
    if exclude_patterns:
        filtered_nodes = [node for node in filtered_nodes if not any(re.match(pattern, node) for pattern in exclude_patterns)]
    return filtered_nodes

def analyze_system_costs(d):
    ls_nodes = list(d.model.nodes.keys())

    excluded_hydro_basins = [node for node in ls_nodes if re.match(r'HYDRO_BASIN_[\w]+_GR$', node)]
    GR_nodes = filter_nodes(ls_nodes, "_GR", exclude_patterns=excluded_hydro_basins + ["PCCC", "PROD_CO2", "CO2_EXPORT"])
    BE_nodes = filter_nodes(ls_nodes, "_BE", exclude_patterns=["PROD_CO2", "ENERGY_DEMAND_BE"]) + ["LIQUEFIED_METHANE_STORAGE_DESTINATION", "LIQUEFIED_METHANE_REGASIFICATION"]

    cost_GR = cost_rreh(d, GR_nodes)
    cost_BE = cost_rreh(d, BE_nodes)

    obj_cost = d.solution.objective
    tot_cost = cost_GR + cost_BE
    abs_diff = np.abs(obj_cost - tot_cost)

    print(f"GR cost: {cost_GR}")
    print(f"BE cost: {cost_BE}")
    print(f"Objective cost: {obj_cost}")
    print(f"Total computed cost: {tot_cost}")
    print(f"Cost difference is within tolerance: {abs_diff < 0.1}")

    return GR_nodes, BE_nodes, cost_GR, cost_BE, tot_cost

def analyze_and_plot_capacities(scenario, d, results_path, timehorizon, report=False):
    plant_capacities = calculate_plant_capacities(d)
    plot_and_save_capacities(plant_capacities, scenario, results_path, timehorizon, report)
    
    return plant_capacities

def calculate_plant_capacities(d):
    plant_capacities = {}

    # Dynamically calculate capacities based on available data in 'd'
    if hasattr(d.solution.elements, "ON_WIND_PLANTS_GR"):
        plant_capacities['wind_onshore_gr'] = np.sum(d.solution.elements.ON_WIND_PLANTS_GR.variables.capacity.values)
    if hasattr(d.solution.elements, "OFF_WIND_PLANTS_GR"):
        plant_capacities['wind_offshore_gr'] = np.sum(d.solution.elements.OFF_WIND_PLANTS_GR.variables.capacity.values)
    if hasattr(d.solution.elements, "WAVE_PLANT_GR"):
        wave_units = d.solution.elements.WAVE_PLANT_GR.variables.num_units.values[0]
        wave_rp = d.model.nodes.WAVE_PLANT_GR.parameters.unit_rated_power[0]
        plant_capacities['wave_gr'] = wave_units * wave_rp

    # Adding hydro plant capacities
    ls_nodes = list(d.model.nodes.keys())
    for element_name in ls_nodes:
        if re.match(r'HYDRO_PLANT_\w+_GR$', element_name):
            element = getattr(d.solution.elements, element_name, None)
            if element is not None and hasattr(element, 'variables'):
                capacity_values = element.variables.capacity.values
                if capacity_values:
                    plant_capacities[element_name] = np.sum(capacity_values)

    plant_capacities['battery_flow'] = np.sum(d.solution.elements.BATTERY_STORAGE_GR.variables.capacity_flow.values)
    plant_capacities['battery_stock'] = np.sum(d.solution.elements.BATTERY_STORAGE_GR.variables.capacity_stock.values)

    return plant_capacities

def plot_and_save_capacities(capacities, scenario, results_path, timehorizon, report):
    # Calculate total hydro capacity and prepare other categories and their capacities
    hydro_keys = [k for k in capacities if "HYDRO_PLANT" in k]
    total_hydro_capacity = sum(capacities[k] for k in hydro_keys)
    
    categories = ['Onshore Wind', 'Offshore Wind', 'Wave', 'Battery Flow', 'Battery Stock']
    capacity_values = [
        capacities.get('wind_onshore_gr', 0),
        capacities.get('wind_offshore_gr', 0),
        capacities.get('wave_gr', 0),
        capacities.get('battery_flow', 0),
        capacities.get('battery_stock', 0),
    ]
    
    # Only add the total hydro capacity if there are hydro plants
    if hydro_keys:
        categories.append('Total Hydro')
        capacity_values.append(total_hydro_capacity)
    
    color = [plt.cm.viridis(0.75)]

    # Setup the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(categories, capacity_values, color=color)
    
    ax.set_ylabel('Installed Capacity (GW)')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Adjust y-axis to fit the labels
    ax.set_ylim(0, max(capacity_values) * 1.2)
    
    # Add labels above the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f} GW',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords='offset points',
                    ha='center', va='bottom')

    # Save the general capacities plot
    img_folder_name = f"img_{timehorizon}/{scenario}" if timehorizon else "img_all"
    img_folder_path = os.path.join(results_path, img_folder_name)
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    
    if report:
        ax.set_title('')
        general_cap_fig_path = os.path.join(img_folder_path, f"{scenario}_general_capacities.pdf")
    else: 
        ax.set_title(f'Installed Renewable Capacities for {scenario.capitalize()} Scenario')
        general_cap_fig_path = os.path.join(img_folder_path, f"{scenario}_general_capacities.png")

    fig.savefig(general_cap_fig_path)
    plt.close(fig)
    
    # For scenarios with hydro, create a separate plot for hydro plant capacities
    if hydro_keys:
        plot_individual_hydro_capacities(hydro_keys, capacities, scenario, img_folder_path, report)

def plot_individual_hydro_capacities(hydro_keys, capacities, scenario, img_folder_path, report):
    
    color = [plt.cm.viridis(0.25)]
    
    # Plot individual hydro capacities
    fig, ax = plt.subplots(figsize=(10, 6))
    hydro_values = [capacities[k] for k in hydro_keys]
    bars = ax.bar(hydro_keys, hydro_values, color=color)
    ax.set_ylabel('Installed Capacity (GW)')
    ax.set_ylim(0, max(hydro_values) * 1.2)  # Ensure there is space for labels
    
    # Add labels above the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f} GW',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords='offset points',
                    ha='center', va='bottom')
    
    plt.tight_layout()

    if report: 
        ax.set_title('')
        hydro_cap_fig_path = os.path.join(img_folder_path, f"{scenario}_hydro_capacities.pdf")
    else: 
        ax.set_title(f'Hydro Plant Capacities for {scenario.capitalize()} Scenario')
        hydro_cap_fig_path = os.path.join(img_folder_path, f"{scenario}_hydro_capacities.png")
    
    fig.savefig(hydro_cap_fig_path)
    plt.close(fig)

def calculate_price_per_mwh(d, tot_cost, scenario, timehorizon):
    gas = np.array(d.solution.elements.LIQUEFIED_METHANE_REGASIFICATION.variables.methane.values) * 15.31  # kt/h * GWh/kt
    gas_prod = np.sum(gas)  # GWh
    demand_in_mwh = gas_prod * 1e3  # Convert GWh to MWh
    total_cost_in_euros = tot_cost * 1e6  # Convert million euros to euros

    # Compute price per MWh in euros
    price_per_mwh = total_cost_in_euros / demand_in_mwh
    print(f"The price in the scenario {scenario} with the time horizon {timehorizon} is {price_per_mwh:.3f} €/MWh")

    return price_per_mwh, demand_in_mwh

def cost_rreh_detailed(d, ls):
    detailed_cost = {}
    if not ls:
        return detailed_cost

    for ele in ls:
        node_cost = 0
        element = getattr(d.solution.elements, ele, None)
        if element is not None:
            try:
                node_cost += np.sum(element.objectives.unnamed)
            except AttributeError:
                node_cost += sum(element.objectives.named.values())
        detailed_cost[ele] = node_cost
    
    return detailed_cost

def plot_cost_breakdown(detailed_costs, demand_in_mwh, title, results_path, scenario, timehorizon, source='GR', report=False):
    # Convert costs to €/MWh
    adjusted_costs = {node: cost * 1e6 / demand_in_mwh for node, cost in detailed_costs.items()}

    # Sort nodes and costs
    sorted_nodes_and_costs = sorted(adjusted_costs.items(), key=lambda item: item[1], reverse=True)
    sorted_nodes = [item[0] for item in sorted_nodes_and_costs]
    sorted_costs = [item[1] for item in sorted_nodes_and_costs]

    # Generate colors - ensuring consistent colors across plots
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_nodes)))[::-1]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(sorted_nodes, sorted_costs, color=colors)

    ax.set_xlabel('Cost (€/MWh)')
    ax.invert_yaxis()  # To match the provided image's layout

    # Adding the cost next to each bar
    for bar, cost in zip(bars, sorted_costs):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"{cost:.2f}",
                va='center', ha='left')

    plt.tight_layout()  # Adjust layout

    # Save the cost breakdown plot
    img_folder_name = f"img_{timehorizon}/{scenario}" if timehorizon else "img_all"
    img_folder_path = os.path.join(results_path, img_folder_name)
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    
    if report:
        ax.set_title('')
        if source == 'BE':
            cost_breakdown_fig_path = os.path.join(img_folder_path, f"{scenario}_cost_breakdown_BE.pdf")
        else:  # Default to 'GR' if not specified or for any other value
            cost_breakdown_fig_path = os.path.join(img_folder_path, f"{scenario}_cost_breakdown_GR.pdf")
    else:
        ax.set_title(title)
        if source == 'BE':
            cost_breakdown_fig_path = os.path.join(img_folder_path, f"{scenario}_cost_breakdown_BE.png")
        else:  # Default to 'GR' if not specified or for any other value
            cost_breakdown_fig_path = os.path.join(img_folder_path, f"{scenario}_cost_breakdown_GR.png")        
    fig.savefig(cost_breakdown_fig_path)
    plt.close(fig)  # Close the plot to release memory

def plot_production_dynamics(d, results_path, scenario, timehorizon, start_date='2015-01-01'):
    production_data = {}

    # Determine which production types to include based on the scenario
    production_types_mapping = {
        'combined': ['OFF_WIND_PLANTS_GR', 'ON_WIND_PLANTS_GR', 'HYDRO_PLANT_03h_GR', 'HYDRO_PLANT_03j_GR', 'HYDRO_PLANT_05h_GR', 'WAVE_PLANT_GR'],
        'wind_onshore': ['ON_WIND_PLANTS_GR'],
        'wind_offshore': ['OFF_WIND_PLANTS_GR'],
        'hydro': ['HYDRO_PLANT_03h_GR', 'HYDRO_PLANT_03j_GR', 'HYDRO_PLANT_05h_GR'],
        'wave': ['WAVE_PLANT_GR']
    }
    production_types = production_types_mapping.get(scenario, [])

    for production_type in production_types:
        data = np.array(getattr(d.solution.elements, production_type).variables.electricity.values)
        production_data[production_type] = data if np.max(data) >= 0.001 else np.zeros_like(data)

    if not production_data:
        return

    date_range = pd.date_range(start=start_date, periods=len(next(iter(production_data.values()))), freq='H')
    num_plots = len(production_data)
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, num_plots * 2), sharex=True)
    if num_plots == 1:
        axs = [axs]
        colors = [plt.cm.viridis(0.5)]
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, num_plots))

    for ax, ((production_type, production_data), color) in zip(axs, zip(production_data.items(), colors)):
        data_series = pd.Series(production_data).rolling(window=24).mean()
        if data_series.max() < 0.001:
            ax.axhline(y=0, color='k', linestyle='--')
            ax.set_ylim(-0.1, 0.1)
        else:
            ax.fill_between(date_range, data_series, color=color)
        ax.set_title(f'{production_type.replace("_", " ").title()} Production (Greenland)', fontsize=10, loc='left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.set_ylabel('GWh')

    img_folder_name = f"img_{timehorizon}/{scenario}" if timehorizon else "img_all"
    img_folder_path = os.path.join(results_path, img_folder_name)
    os.makedirs(img_folder_path, exist_ok=True)
    production_fig_path = os.path.join(img_folder_path, f"{scenario}_production_dynamics.png")
    fig.savefig(production_fig_path)
    plt.close(fig)

def plot_basin_dynamics(d, results_path, scenario, timehorizon, start_date='2015-01-01'):
    if scenario not in ['hydro', 'combined']:
        return  # Only proceed if the correct scenario

    ls_nodes = list(d.model.nodes.keys())
    hydro_basin_nodes = [node for node in ls_nodes if 'HYDRO_BASIN' in node]

    date_range = pd.date_range(start=start_date, periods=17544, freq='H')
    colors = plt.cm.viridis(np.linspace(0, 1, 4))

    for basin_name in hydro_basin_nodes:
        basin_data = {
            'inflow': getattr(d.model.nodes, basin_name).parameters.inflow_series[:17544],
            'storage': getattr(d.solution.elements, basin_name).variables.storage.values,
            'release': getattr(d.solution.elements, basin_name).variables.release.values,
            'spill': getattr(d.solution.elements, basin_name).variables.spill.values
        }

        fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        for idx, (param, ax) in enumerate(zip(['inflow', 'storage', 'release', 'spill'], axs)):
            data_series = pd.Series(basin_data[param]).rolling(window=24).mean()
            if data_series.max() < 0.001:
                ax.axhline(y=0, color='k', linestyle='--')
                ax.set_ylim(-0.1, 0.1)
            else:
                ax.plot(date_range, data_series, color=colors[idx])
            ax.set_title(f"{basin_name} {param} dynamics", fontsize=10, loc='left')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.set_ylabel('Volume (TCM/h)')
            ax.set_ylim(bottom=0)

        plt.xticks(rotation=45)
        plt.xlabel('Time')
        fig.tight_layout()

        img_folder_name = f"img_{timehorizon}/{scenario}" if timehorizon else "img_all"
        img_folder_path = os.path.join(results_path, img_folder_name)
        os.makedirs(img_folder_path, exist_ok=True)
        basin_fig_path = os.path.join(img_folder_path, f"{scenario}_{basin_name}_dynamics.png")
        fig.savefig(basin_fig_path)
        plt.close(fig)

def plot_storage_dynamics(d, results_path, scenario, timehorizon, start_date='2015-01-01'):
    date_range = pd.date_range(start=start_date, periods=17544, freq='H')

    battery_data = np.array(d.solution.elements.BATTERY_STORAGE_GR.variables.electricity_stored.values)
    hydrogen_data = np.array(d.solution.elements.HYDROGEN_STORAGE_GR.variables.hydrogen_stored.values)
    methane_data_src = np.array(d.solution.elements.LIQUEFIED_METHANE_STORAGE_HUB_GR.variables.liquefied_methane_stored.values)
    methane_data_dest = np.array(d.solution.elements.LIQUEFIED_METHANE_STORAGE_DESTINATION.variables.liquefied_methane_stored.values)

    fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    storage_labels = ['Battery Storage Dynamics', 'Hydrogen Storage Dynamics', 'Liquid Methane Storage Dynamics (Greenland)', 'Liquid Methane Storage Dynamics (Belgium)']

    data_frames = [battery_data, hydrogen_data, methane_data_src, methane_data_dest]

    for ax, data, color, label in zip(axs, data_frames, colors, storage_labels):
        data_series = pd.Series(data).rolling(window=24).mean()
        if data_series.max() < 0.001:
            ax.axhline(y=0, color='k', linestyle='--')
            ax.set_ylim(-0.1, 0.1)
        else:
            ax.plot(date_range, data_series, label=label, color=color)
        ax.set_title(label, fontsize=10, loc='left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.set_ylabel('GWh')
        ax.legend(loc='upper right')

    fig.tight_layout()

    img_folder_name = f"img_{timehorizon}/{scenario}" if timehorizon else "img_all"
    img_folder_path = os.path.join(results_path, img_folder_name)
    os.makedirs(img_folder_path, exist_ok=True)
    
    storage_dynamics_fig_path = os.path.join(img_folder_path, f"{scenario}_storage_dynamics.png")
    fig.savefig(storage_dynamics_fig_path)
    plt.close(fig)

def analyze_and_plot_tech_capacities(scenario, d, results_path, timehorizon, report=False):
    plant_capacities = calculate_plant_tech_capacities(d)
    plot_and_save_tech_capacities(plant_capacities, scenario, results_path, timehorizon, report)
    
    return plant_capacities

def calculate_plant_tech_capacities(d):
    plant_capacities = {}
    
    plant_capacities['electrolysis'] = np.sum(d.solution.elements.ELECTROLYSIS_PLANTS_GR.variables.capacity.values)
    plant_capacities['dac'] = np.sum(d.solution.elements.DIRECT_AIR_CAPTURE_PLANTS_GR.variables.capacity.values)
    plant_capacities['methanation'] = np.sum(d.solution.elements.METHANATION_PLANTS_GR.variables.capacity.values)

    return plant_capacities

def plot_and_save_tech_capacities(capacities, scenario, results_path, timehorizon, report):
    
    categories = ['Electrolysis ', 'DAC', 'Methanation']
    capacity_values = [
        capacities.get('electrolysis', 0),
        capacities.get('dac', 0),
        capacities.get('methanation', 0)
    ]
    
    color = [plt.cm.viridis(0)]

    # Setup the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(categories, capacity_values, color=color)
    
    ax.set_ylabel('Installed Capacity (kt/h)')
    ax.set_title(f'Installed Tech Capacities for {scenario.capitalize()} Scenario')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Adjust y-axis to fit the labels
    ax.set_ylim(0, max(capacity_values) * 1.2)
    
    # Add labels above the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f} kt/h',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords='offset points',
                    ha='center', va='bottom')

    # Save the general capacities plot
    img_folder_name = f"img_{timehorizon}/{scenario}" if timehorizon else "img_all"
    img_folder_path = os.path.join(results_path, img_folder_name)
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    
    if report:
        ax.set_title('')
        general_cap_fig_path = os.path.join(img_folder_path, f"{scenario}_tech_capacities.png")
    else: 
        ax.set_title(f'Installed Tech Capacities for {scenario.capitalize()} Scenario')
        general_cap_fig_path = os.path.join(img_folder_path, f"{scenario}_tech_capacities.png")

    fig.savefig(general_cap_fig_path)
    plt.close(fig)


class MakeMeReadable:
    def __init__(self, d):
        self.d = d
    
    def __dir__(self):
        return self.d.keys()
    
    def __getattr__(self, v):
        try:
            out = self.d[v]
            if isinstance(out, dict):
                return MakeMeReadable(out)
            return out
        except:
            return getattr(self.d, v)
        
    def __str__(self):
        return str(self.d)
    
    def __repr__(self):
        return repr(self.d)

def main():
    args = parse_args()
    
    scenarios = ['wind_onshore', 'wind_offshore', 'wave', 'hydro', 'hydro_wind', 'combined']

    base_path = 'models'  # Adjust this path as necessary

    csv_path = f'scripts/results/scenario_analysis_results_{args.timehorizon}.csv'
    csv_exists = os.path.isfile(csv_path)
    if csv_exists:
        os.remove(csv_path)

    all_data = []

    for scenario in scenarios:
        aggregated_data = {
            "scenario": [],
            "wind_onshore_gr": [],
            "wind_offshore_gr": [],
            "wave_gr": [],
            "hydro_3h_gr": [],
            "hydro_3j_gr": [],
            "hydro_5h_gr": [],
            "battery_flow": [],
            "battery_stock": [],
            "Electrolysis": [],
            "DAC": [],
            "Methanation": [],
            "price per mwh": [],
            "total cost": []
        }

        results_path = os.path.join(base_path, scenario, 'results')

        for file_name in os.listdir(results_path):
            if file_name.endswith('.json'):
                # Split the filename to extract parts
                parts = file_name.split('_')
                # Ensure the filename structure matches expected convention
                if len(parts) >= 3 and parts[-1] == "results.json":
                    try:
                        # Attempt to extract the time horizon part based on the new naming convention
                        file_timehorizon = int(parts[-2])
                    except ValueError:
                        # If the time horizon part cannot be converted to an integer, skip the file
                        continue

                    # Check if the extracted time horizon matches the specified one (if any)
                    if args.timehorizon == file_timehorizon:
                        # Only proceed with loading and analyzing the data if time horizons match
                        file_path = os.path.join(results_path, file_name)
                        print(f"File path: {file_path}")
                        data = load_results(file_path)
                
                        # Split the filename on the underscores
                        parts = file_name.split('_')
                        # Remove the part that is just digits and the 'results.json' part
                        scenario_parts = [part for part in parts if not part.isdigit() and not part.endswith('.json')]
                        # Join the remaining parts back together to get the scenario name
                        scenario_name = '_'.join(scenario_parts)

                        aggregated_data['scenario'].append(scenario_name)

                        # Initial check
                        print(f"Plot of balances for {scenario_name} with time horizon {args.timehorizon}")
                        compute_and_plot_balances(scenario_name, data, results_path, args.timehorizon)

                        # Compute total costs
                        print(f"Computing total costs for {scenario_name} with time horizon {args.timehorizon}")
                        GR_nodes, BE_nodes, cost_GR, cost_BE, tot_cost = analyze_system_costs(data)
                        
                        aggregated_data['total cost'].append(tot_cost)

                        # Retrieve and plot the installed renewable capacities
                        print(f"Retrieving and plotting installed capacities for {scenario_name} with time horizon {args.timehorizon}")
                        plant_capacities = analyze_and_plot_capacities(scenario_name, data, results_path, args.timehorizon, args.report)

                        # Aggregate plant capacities
                        # Direct mapping for non-hydro capacities
                        for capacity_type in ["wind_onshore_gr", "wind_offshore_gr", "wave_gr", "battery_flow", "battery_stock"]:
                            aggregated_data[capacity_type].append(plant_capacities.get(capacity_type, "NA"))

                        # Handle hydro capacities
                        # Adapt these keys based on how they're actually named in your returned 'plant_capacities'
                        hydro_mappings = {
                            "HYDRO_PLANT_03h_GR": "hydro_3h_gr",
                            "HYDRO_PLANT_03j_GR": "hydro_3j_gr",
                            "HYDRO_PLANT_05h_GR": "hydro_5h_gr",
                        }
                        for original_key, new_key in hydro_mappings.items():
                            # Use 'NA' if not found
                            capacity = plant_capacities.get(original_key, "NA")
                            aggregated_data[new_key].append(capacity)

                        # Retrieve and plot the installed capacities
                        print(f"Computing the price of {scenario_name} with time horizon {args.timehorizon}")
                        price_per_mwh, demand_in_mwh = calculate_price_per_mwh(data, tot_cost, scenario_name, args.timehorizon)

                        aggregated_data['price per mwh'].append(price_per_mwh)
                        
                        # Plot cost breakdown
                        print(f"Plotting the cost breakdown of {scenario_name} with time horizon {args.timehorizon}")                
                        cost_details_GR = cost_rreh_detailed(data, GR_nodes)
                        plot_cost_breakdown(cost_details_GR, demand_in_mwh, 'Synthetic Methane (Greenland) Cost Breakdown (€/MWh)', results_path, scenario_name, args.timehorizon, 'GR' ,args.report)
                        cost_details_BE = cost_rreh_detailed(data, BE_nodes)
                        plot_cost_breakdown(cost_details_BE, demand_in_mwh, 'Synthetic Methane (Belgium) Cost Breakdown (€/MWh)', results_path, scenario_name, args.timehorizon, 'BE', args.report)

                        # Plot energy production dynamic
                        plot_production_dynamics(data, results_path, scenario_name, args.timehorizon)
                        
                        # Plot basins dynamic
                        if scenario in ['hydro', 'combined']:
                            plot_basin_dynamics(data, results_path, scenario_name, args.timehorizon)

                        # Plot storage dynamic
                        plot_storage_dynamics(data, results_path, scenario_name, args.timehorizon)

                        # Retrieve and plot the installed side techs capacities
                        print(f"Retrieving and plotting installed side tech capacities for {scenario_name} with time horizon {args.timehorizon}")
                        tech_capacities = analyze_and_plot_tech_capacities(scenario_name, data, results_path, args.timehorizon, args.report)

                        for tech_type in ["Electrolysis", "DAC", "Methanation"]:
                            aggregated_data[tech_type].append(tech_capacities.get(tech_type.lower().replace(" ", "_"), "NA"))  # Assuming the keys in tech_capacities are lowercase with underscores

        all_data.append(aggregated_data)

    df = pd.DataFrame(all_data)  # Ensuring data is in a list to form a single row
    df.to_csv(csv_path, index=False)

    print(f"Data for {scenario} appended to {csv_path}")

if __name__ == "__main__":
    main()
