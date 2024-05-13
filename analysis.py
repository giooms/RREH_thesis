import json
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import BrokenBarHCollection
import matplotlib.dates as mdates
import matplotlib.patches as patches
import pandas as pd
import os
import argparse
from matplotlib.lines import Line2D
import numpy as np
import re
import ast

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze GBOML model results with specific criteria.")
    parser.add_argument('-m', '--mode', help="Mode of operation: analyze_json or analyze_csv", default='analyze_json', choices=['analyze_json', 'analyze_csv'])
    # parser.add_argument('-s', '--scenarios', nargs='+', help='Scenarios to analyze (e.g., hydro wind_onshore)', default=None)
    parser.add_argument('-t', '--timehorizon', type=int, help='Time horizon to filter by', default=17544)
    parser.add_argument('-r', '--report', action='store_true', help='Save as PDF without title for reports')
    return parser.parse_args()

# =============== ============== ===============
# ============= JSON FILES ANALYSIS =============
# =============== ============== ===============

def load_results(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return MakeMeReadable(data)

def plot_balance(ax, data, title):
    ylim = (-0.1, 0.1)

    ax.plot(data)
    ax.set_ylim(ylim)
    ax.set_title(title)

def compute_and_plot_balances(scenario, d, results_path, timehorizon, wacc_label):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 15), constrained_layout=True)
    axs = axs.flatten()  # Ensuring axs is a 1D array for easy indexing
    
    # Generalized approach to handle variations in power_balance_rreh
    power_balances = {
        'wind_offshore': lambda d: np.array(d.solution.elements.OFF_WIND_PLANTS_RREH.variables.electricity.values),
        'wind_onshore': lambda d: np.array(d.solution.elements.ON_WIND_PLANTS_RREH.variables.electricity.values),
        'hydro': lambda d: np.array(d.solution.elements.HYDRO_PLANT_03h_RREH.variables.electricity.values) +
                           np.array(d.solution.elements.HYDRO_PLANT_03j_RREH.variables.electricity.values) +
                           np.array(d.solution.elements.HYDRO_PLANT_05h_RREH.variables.electricity.values),
        'wave': lambda d: np.array(d.solution.elements.WAVE_PLANT_RREH.variables.electricity.values),
        'hydro_wind': lambda d: np.array(d.solution.elements.OFF_WIND_PLANTS_RREH.variables.electricity.values) +
                        np.array(d.solution.elements.ON_WIND_PLANTS_RREH.variables.electricity.values) +
                        np.array(d.solution.elements.HYDRO_PLANT_03h_RREH.variables.electricity.values) +
                        np.array(d.solution.elements.HYDRO_PLANT_03j_RREH.variables.electricity.values) +
                        np.array(d.solution.elements.HYDRO_PLANT_05h_RREH.variables.electricity.values), 
        'combined': lambda d: np.array(d.solution.elements.OFF_WIND_PLANTS_RREH.variables.electricity.values) +
                              np.array(d.solution.elements.ON_WIND_PLANTS_RREH.variables.electricity.values) +
                              np.array(d.solution.elements.HYDRO_PLANT_03h_RREH.variables.electricity.values) +
                              np.array(d.solution.elements.HYDRO_PLANT_03j_RREH.variables.electricity.values) +
                              np.array(d.solution.elements.HYDRO_PLANT_05h_RREH.variables.electricity.values) + 
                              np.array(d.solution.elements.WAVE_PLANT_RREH.variables.electricity.values),
        'spain': lambda d: np.array(d.solution.elements.ON_WIND_PLANTS_RREH.variables.electricity.values) +
                    np.array(d.solution.elements.SOLAR_PV_PLANTS_RREH.variables.electricity.values),
        'algeria': lambda d: np.array(d.solution.elements.ON_WIND_PLANTS_RREH.variables.electricity.values) +
                    np.array(d.solution.elements.SOLAR_PV_PLANTS_RREH.variables.electricity.values),
        'germany': lambda d: np.array(d.solution.elements.ON_WIND_PLANTS_RREH.variables.electricity.values) +
                    np.array(d.solution.elements.SOLAR_PV_PLANTS_RREH.variables.electricity.values)
    }

    battery_storage_rreh_out = np.array(d.solution.elements.BATTERY_STORAGE_RREH.variables.electricity_out.values)
    battery_storage_rreh_in = np.array(d.solution.elements.BATTERY_STORAGE_RREH.variables.electricity_in.values)
    hvdc_rreh_in = np.array(d.solution.elements.HVDC_RREH.variables.electricity_in.values)

    parts = scenario.split('_')
    scenario = '_'.join(parts[:-1])

    # Compute and plot power balance for the scenario
    if scenario in power_balances:
        power_balance_rreh = power_balances[scenario](d) + battery_storage_rreh_out - battery_storage_rreh_in - hvdc_rreh_in
        plot_balance(axs[0], power_balance_rreh, f'{scenario} RREH_Inland Power Balance')

    # RREH_CO2 Balance
    co2_storage_rreh_in = np.array(d.solution.elements.CARBON_DIOXIDE_STORAGE_RREH.variables.carbon_dioxide_in.values)
    co2_storage_rreh_out = np.array(d.solution.elements.CARBON_DIOXIDE_STORAGE_RREH.variables.carbon_dioxide_out.values)
    dac_rreh = np.array(d.solution.elements.DIRECT_AIR_CAPTURE_PLANTS_RREH.variables.carbon_dioxide.values)
    methanation_rreh = np.array(d.solution.elements.METHANATION_PLANTS_RREH.variables.carbon_dioxide.values)
    co2_balance_rreh = dac_rreh + co2_storage_rreh_out - co2_storage_rreh_in - methanation_rreh

    plot_balance(axs[1], co2_balance_rreh, 'RREH_CO2 Balance')

    # RREH_LCH4 Balance
    lch4_carriers_rreh = np.array(d.solution.elements.LIQUEFIED_METHANE_CARRIERS_RREH.variables.liquefied_methane_in.values)
    lch4_storage_rreh_in = np.array(d.solution.elements.LIQUEFIED_METHANE_STORAGE_HUB_RREH.variables.liquefied_methane_in.values)
    lch4_storage_rreh_out = np.array(d.solution.elements.LIQUEFIED_METHANE_STORAGE_HUB_RREH.variables.liquefied_methane_out.values)
    ch4_liquefaction_rreh = np.array(d.solution.elements.METHANE_LIQUEFACTION_PLANTS_RREH.variables.liquefied_methane.values)
    lch4_balance_rreh = ch4_liquefaction_rreh + lch4_storage_rreh_out - lch4_storage_rreh_in - lch4_carriers_rreh

    plot_balance(axs[2], lch4_balance_rreh, 'RREH_LCH4 Balance')

    # Check BE_CH4 Balance
    conversion_factor = d.model.hyperedges.DESTINATION_METHANE_BALANCE.parameters.conversion_factor
    demand_CH4 = np.array(d.model.hyperedges.DESTINATION_METHANE_BALANCE.parameters.demand)
    lch4_regasification_be = np.array(d.solution.elements.LIQUEFIED_METHANE_REGASIFICATION.variables.methane.values)
    ch4_balance_be =  lch4_regasification_be - demand_CH4[:timehorizon]

    plot_balance(axs[3], ch4_balance_be, 'BE_CH4 Balance')

    # Check BE_LCH4 Balance
    lch4_carriers_rreh = np.array(d.solution.elements.LIQUEFIED_METHANE_CARRIERS_RREH.variables.liquefied_methane_out.values)
    lch4_storage_dest_in = np.array(d.solution.elements.LIQUEFIED_METHANE_STORAGE_DESTINATION.variables.liquefied_methane_in.values)
    lch4_storage_dest_out = np.array(d.solution.elements.LIQUEFIED_METHANE_STORAGE_DESTINATION.variables.liquefied_methane_out.values)
    lch4_regasification_be = np.array(d.solution.elements.LIQUEFIED_METHANE_REGASIFICATION.variables.liquefied_methane.values)
    lch4_balance_be = lch4_carriers_rreh + lch4_storage_dest_out - lch4_storage_dest_in - lch4_regasification_be

    plot_balance(axs[4], lch4_balance_be, 'BE_LCH4 Balance')

    # Check RREH_Coastal Power Balance
    hvdc_rreh_out = np.array(d.solution.elements.HVDC_RREH.variables.electricity_out.values)
    electrolysis_rreh = np.array(d.solution.elements.ELECTROLYSIS_PLANTS_RREH.variables.electricity.values)
    h2_storage_rreh = np.array(d.solution.elements.HYDROGEN_STORAGE_RREH.variables.electricity.values)
    desalination_rreh = np.array(d.solution.elements.DESALINATION_PLANTS_RREH.variables.electricity.values)
    h2o_storage_rreh = np.array(d.solution.elements.WATER_STORAGE_RREH.variables.electricity.values)
    co2_storage_rreh = np.array(d.solution.elements.CARBON_DIOXIDE_STORAGE_RREH.variables.electricity.values)
    ch4_liquefaction_rreh = np.array(d.solution.elements.METHANE_LIQUEFACTION_PLANTS_RREH.variables.electricity.values)
    dac_rreh = np.array(d.solution.elements.DIRECT_AIR_CAPTURE_PLANTS_RREH.variables.electricity.values)
    power_balance_coast_rreh = hvdc_rreh_out - electrolysis_rreh - h2_storage_rreh - desalination_rreh - h2o_storage_rreh - co2_storage_rreh - ch4_liquefaction_rreh - dac_rreh 

    plot_balance(axs[5], power_balance_coast_rreh, 'RREH_Coastal Power Balance')

    # Check RREH_Coastal H2 Balance
    electrolysis_rreh = np.array(d.solution.elements.ELECTROLYSIS_PLANTS_RREH.variables.hydrogen.values)
    h2_storage_rreh_out = np.array(d.solution.elements.HYDROGEN_STORAGE_RREH.variables.hydrogen_out.values)
    h2_storage_rreh_in = np.array(d.solution.elements.HYDROGEN_STORAGE_RREH.variables.hydrogen_in.values)
    methanation_rreh = np.array(d.solution.elements.METHANATION_PLANTS_RREH.variables.hydrogen.values)
    dac_rreh = np.array(d.solution.elements.DIRECT_AIR_CAPTURE_PLANTS_RREH.variables.hydrogen.values)
    h2_balance_rreh = electrolysis_rreh + h2_storage_rreh_out - h2_storage_rreh_in - methanation_rreh - dac_rreh

    plot_balance(axs[6], h2_balance_rreh, 'RREH_Coastal H2 Balance')

    # Check RREH_Coastal H2O Balance
    desalination_rreh = np.array(d.solution.elements.DESALINATION_PLANTS_RREH.variables.water.values)
    methanation_rreh = np.array(d.solution.elements.METHANATION_PLANTS_RREH.variables.water.values)
    h2o_storage_rreh_out = np.array(d.solution.elements.WATER_STORAGE_RREH.variables.water_out.values)
    h2o_storage_rreh_in = np.array(d.solution.elements.WATER_STORAGE_RREH.variables.water_in.values)
    electrolysis_rreh = np.array(d.solution.elements.ELECTROLYSIS_PLANTS_RREH.variables.water.values)
    dac_rreh = np.array(d.solution.elements.DIRECT_AIR_CAPTURE_PLANTS_RREH.variables.water.values)
    h20_balance_rreh = desalination_rreh + methanation_rreh + h2o_storage_rreh_out - h2o_storage_rreh_in - electrolysis_rreh - dac_rreh

    plot_balance(axs[7], h20_balance_rreh, 'RREH_Coastal H2O Balance')

    # Check RREH_Coastal CH4 Balance
    methanation_rreh = np.array(d.solution.elements.METHANATION_PLANTS_RREH.variables.methane.values)
    ch4_liquefaction_rreh = np.array(d.solution.elements.METHANE_LIQUEFACTION_PLANTS_RREH.variables.liquefied_methane.values)
    ch4_balance_rreh = methanation_rreh - ch4_liquefaction_rreh

    plot_balance(axs[8], ch4_balance_rreh, 'RREH_Coastal CH4 Balance')
    
    img_folder_name = f"img_{timehorizon}/{wacc_label}" if timehorizon else "img_all"
    
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

    excluded_hydro_basins = [node for node in ls_nodes if re.match(r'HYDRO_BASIN_[\w]+_RREH$', node)]
    RREH_nodes = filter_nodes(ls_nodes, "_RREH", exclude_patterns=excluded_hydro_basins + ["PCCC", "PROD_CO2", "CO2_EXPORT"])
    BE_nodes = filter_nodes(ls_nodes, "_BE", exclude_patterns=["PROD_CO2", "ENERGY_DEMAND_BE"]) + ["LIQUEFIED_METHANE_STORAGE_DESTINATION", "LIQUEFIED_METHANE_REGASIFICATION"]

    cost_RREH = cost_rreh(d, RREH_nodes)
    cost_BE = cost_rreh(d, BE_nodes)

    obj_cost = d.solution.objective
    tot_cost = cost_RREH + cost_BE
    abs_diff = np.abs(obj_cost - tot_cost)

    print(f"RREH cost: {cost_RREH}")
    print(f"BE cost: {cost_BE}")
    print(f"Objective cost: {obj_cost}")
    print(f"Total computed cost: {tot_cost}")
    print(f"Cost difference is within tolerance: {abs_diff < 0.1}")

    return RREH_nodes, BE_nodes, cost_RREH, cost_BE, tot_cost

def analyze_and_plot_capacities(scenario, d, results_path, timehorizon, wacc_label, report=False):
    plant_capacities = calculate_plant_capacities(d)
    plot_and_save_capacities(plant_capacities, scenario, results_path, timehorizon, report, wacc_label)
    
    return plant_capacities

def calculate_plant_capacities(d):
    plant_capacities = {}

    # Dynamically calculate capacities based on available data in 'd'
    if hasattr(d.solution.elements, "ON_WIND_PLANTS_RREH"):
        plant_capacities['wind_onshore_rreh'] = np.sum(d.solution.elements.ON_WIND_PLANTS_RREH.variables.capacity.values)
    if hasattr(d.solution.elements, "OFF_WIND_PLANTS_RREH"):
        plant_capacities['wind_offshore_rreh'] = np.sum(d.solution.elements.OFF_WIND_PLANTS_RREH.variables.capacity.values)
    if hasattr(d.solution.elements, "SOLAR_PV_PLANTS_RREH"):
        plant_capacities['solar_rreh'] = np.sum(d.solution.elements.SOLAR_PV_PLANTS_RREH.variables.capacity.values)
    if hasattr(d.solution.elements, "WAVE_PLANT_RREH"):
        wave_units = d.solution.elements.WAVE_PLANT_RREH.variables.num_units.values[0]
        wave_rp = d.model.nodes.WAVE_PLANT_RREH.parameters.unit_rated_power[0]
        plant_capacities['wave_rreh'] = wave_units * wave_rp

    # Adding hydro plant capacities
    ls_nodes = list(d.model.nodes.keys())
    for element_name in ls_nodes:
        if re.match(r'HYDRO_PLANT_\w+_RREH$', element_name):
            element = getattr(d.solution.elements, element_name, None)
            if element is not None and hasattr(element, 'variables'):
                capacity_values = element.variables.capacity.values
                if capacity_values:
                    plant_capacities[element_name] = np.sum(capacity_values)

    plant_capacities['battery_flow'] = np.sum(d.solution.elements.BATTERY_STORAGE_RREH.variables.capacity_flow.values)
    plant_capacities['battery_stock'] = np.sum(d.solution.elements.BATTERY_STORAGE_RREH.variables.capacity_stock.values)
    plant_capacities['electrolysis'] = np.sum(d.solution.elements.ELECTROLYSIS_PLANTS_RREH.variables.capacity.values)

    return plant_capacities

def plot_and_save_capacities(capacities, scenario, results_path, timehorizon, report, wacc_label):
    # Calculate total hydro capacity and prepare other categories and their capacities
    hydro_keys = [k for k in capacities if "HYDRO_PLANT" in k]
    total_hydro_capacity = sum(capacities[k] for k in hydro_keys)
    
    categories = ['Onshore Wind', 'Offshore Wind', 'Solar', 'Wave', 'Battery Flow', 'Battery Stock', 'Electrolysis']
    capacity_values = [
        capacities.get('wind_onshore_rreh', 0),
        capacities.get('wind_offshore_rreh', 0),
        capacities.get('solar_rreh',0),
        capacities.get('wave_rreh', 0),
        capacities.get('battery_flow', 0),
        capacities.get('battery_stock', 0),
        capacities.get('electrolysis', 0),        
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
    img_folder_name = f"img_{timehorizon}/{wacc_label}" if timehorizon else "img_all"
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
    demand_in_twh = sum(d.solution.elements.LIQUEFIED_METHANE_REGASIFICATION.variables.methane.values) * 0.015441  # kt/h * MWh/kg

    # Compute price per MWh in euros
    price_per_mwh = tot_cost / demand_in_twh # M€/GWh <=> €/MWh
    print(f"The price in the scenario {scenario} with the time horizon {timehorizon} is {price_per_mwh:.3f} €/MWh")

    return price_per_mwh, demand_in_twh

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

def plot_cost_breakdown(detailed_costs, demand_in_twh, title, results_path, scenario, timehorizon, wacc_label, source='RREH', report=False):
    # Convert costs to €/MWh
    adjusted_costs = {node: cost / demand_in_twh for node, cost in detailed_costs.items()}

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
    img_folder_name = f"img_{timehorizon}/{wacc_label}" if timehorizon else "img_all"
    img_folder_path = os.path.join(results_path, img_folder_name)
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    
    if report:
        ax.set_title('')
        if source == 'BE':
            cost_breakdown_fig_path = os.path.join(img_folder_path, f"{scenario}_cost_breakdown_BE.pdf")
        else:  # Default to 'RREH' if not specified or for any other value
            cost_breakdown_fig_path = os.path.join(img_folder_path, f"{scenario}_cost_breakdown_RREH.pdf")
    else:
        ax.set_title(title)
        if source == 'BE':
            cost_breakdown_fig_path = os.path.join(img_folder_path, f"{scenario}_cost_breakdown_BE.png")
        else:  # Default to 'RREH' if not specified or for any other value
            cost_breakdown_fig_path = os.path.join(img_folder_path, f"{scenario}_cost_breakdown_RREH.png")        
    fig.savefig(cost_breakdown_fig_path)
    plt.close(fig)  # Close the plot to release memory

def plot_production_dynamics(d, results_path, scenario, timehorizon, wacc_label, start_date='2015-01-01'):
    production_data = {}

    # Determine which production types to include based on the scenario
    production_types_mapping = {
        'combined': ['OFF_WIND_PLANTS_RREH', 'ON_WIND_PLANTS_RREH', 'HYDRO_PLANT_03h_RREH', 'HYDRO_PLANT_03j_RREH', 'HYDRO_PLANT_05h_RREH', 'WAVE_PLANT_RREH'],
        'wind_onshore': ['ON_WIND_PLANTS_RREH'],
        'wind_offshore': ['OFF_WIND_PLANTS_RREH'],
        'hydro': ['HYDRO_PLANT_03h_RREH', 'HYDRO_PLANT_03j_RREH', 'HYDRO_PLANT_05h_RREH'],
        'wave': ['WAVE_PLANT_RREH'],
        'spain': ['ON_WIND_PLANTS_RREH', 'SOLAR_PV_PLANTS_RREH'],
        'algeria': ['ON_WIND_PLANTS_RREH', 'SOLAR_PV_PLANTS_RREH'],
        'germany': ['ON_WIND_PLANTS_RREH', 'SOLAR_PV_PLANTS_RREH']
    }

    parts = scenario.split('_')
    scenario = '_'.join(parts[:-1])

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
        ax.set_title(f'{production_type.replace("_", " ").title()} Production (RREH)', fontsize=10, loc='left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.set_ylabel('GWh')

    img_folder_name = f"img_{timehorizon}/{wacc_label}" if timehorizon else "img_all"
    img_folder_path = os.path.join(results_path, img_folder_name)
    os.makedirs(img_folder_path, exist_ok=True)
    production_fig_path = os.path.join(img_folder_path, f"{scenario}_production_dynamics.png")
    fig.savefig(production_fig_path)
    plt.close(fig)

def plot_basin_dynamics(d, results_path, scenario, timehorizon, wacc_label, start_date='2015-01-01'):
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

        img_folder_name = f"img_{timehorizon}/{wacc_label}" if timehorizon else "img_all"
        img_folder_path = os.path.join(results_path, img_folder_name)
        os.makedirs(img_folder_path, exist_ok=True)
        basin_fig_path = os.path.join(img_folder_path, f"{scenario}_{basin_name}_dynamics.png")
        fig.savefig(basin_fig_path)
        plt.close(fig)

def plot_storage_dynamics(d, results_path, scenario, timehorizon, wacc_label, start_date='2015-01-01'):
    date_range = pd.date_range(start=start_date, periods=17544, freq='H')

    battery_data = np.array(d.solution.elements.BATTERY_STORAGE_RREH.variables.electricity_stored.values)
    hydrogen_data = np.array(d.solution.elements.HYDROGEN_STORAGE_RREH.variables.hydrogen_stored.values)
    methane_data_src = np.array(d.solution.elements.LIQUEFIED_METHANE_STORAGE_HUB_RREH.variables.liquefied_methane_stored.values)
    methane_data_dest = np.array(d.solution.elements.LIQUEFIED_METHANE_STORAGE_DESTINATION.variables.liquefied_methane_stored.values)

    fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    storage_labels = ['Battery Storage Dynamics', 'Hydrogen Storage Dynamics', 'Liquid Methane Storage Dynamics (RREH)', 'Liquid Methane Storage Dynamics (Belgium)']

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

    img_folder_name = f"img_{timehorizon}/{wacc_label}" if timehorizon else "img_all"
    img_folder_path = os.path.join(results_path, img_folder_name)
    os.makedirs(img_folder_path, exist_ok=True)
    
    storage_dynamics_fig_path = os.path.join(img_folder_path, f"{scenario}_storage_dynamics.png")
    fig.savefig(storage_dynamics_fig_path)
    plt.close(fig)

def analyze_and_plot_tech_capacities(scenario, d, results_path, timehorizon, wacc_label, report=False):
    plant_capacities = calculate_plant_tech_capacities(d)
    plot_and_save_tech_capacities(plant_capacities, scenario, results_path, timehorizon, wacc_label, report)
    
    return plant_capacities

def calculate_plant_tech_capacities(d):
    plant_capacities = {}
    
    plant_capacities['dac'] = np.sum(d.solution.elements.DIRECT_AIR_CAPTURE_PLANTS_RREH.variables.capacity.values)
    plant_capacities['methanation'] = np.sum(d.solution.elements.METHANATION_PLANTS_RREH.variables.capacity.values)

    return plant_capacities

def plot_and_save_tech_capacities(capacities, scenario, results_path, timehorizon, wacc_label, report):
    
    categories = ['DAC', 'Methanation']
    capacity_values = [
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
    img_folder_name = f"img_{timehorizon}/{wacc_label}" if timehorizon else "img_all"
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

def analyze_json_files(args):
    
    # scenarios = ['wind_onshore', 'wind_offshore', 'wave', 'hydro', 'hydro_wind', 'combined', 'spain', 'algeria', 'germany']
    scenarios = ['wind_onshore', 'wind_offshore', 'wave', 'hydro_wind', 'combined', 'spain', 'algeria', 'germany']

    base_path = 'models'  # Adjust this path as necessary

    csv_path = f'scripts/results/scenario_analysis_results_{args.timehorizon}.csv'
    csv_exists = os.path.isfile(csv_path)
    if csv_exists:
        os.remove(csv_path)

    all_data = []

    for scenario in scenarios:
        aggregated_data = {
            "scenario": [],
            "wind_onshore_rreh": [],
            "wind_offshore_rreh": [],
            "solar_rreh": [],
            "wave_rreh": [],
            "hydro_3h_rreh": [],
            "hydro_3j_rreh": [],
            "hydro_5h_rreh": [],
            "hydro_rreh": [],
            "battery_flow": [],
            "battery_stock": [],
            "Electrolysis": [],
            "DAC": [],
            "Methanation": [],
            "price per mwh": [],
            "total cost": [],
            "total cost_rreh": [],
            "total cost_be": [],
            "demand in twh": []
        }

        results_path = os.path.join(base_path, scenario, 'results')

        for file_name in os.listdir(results_path):
                if file_name.endswith('.json'):
                    parts = file_name.split('_')
                    if parts[-1] == "results.json":
                        wacc_label = parts[-3]
                        try:
                            file_timehorizon = int(parts[-2])
                        except ValueError:
                            continue  # Skip if the time horizon cannot be parsed as integer

                        if args.timehorizon == file_timehorizon:
                            # Reconstruct the scenario name from the parts excluding the last three elements
                            scenario_name = '_'.join(parts[:-3])
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
                        compute_and_plot_balances(scenario_name, data, results_path, args.timehorizon, wacc_label)

                        # Compute total costs
                        print(f"Computing total costs for {scenario_name} with time horizon {args.timehorizon}")
                        RREH_nodes, BE_nodes, cost_RREH, cost_BE, tot_cost = analyze_system_costs(data)

                        aggregated_data['total cost'].append(tot_cost)
                        aggregated_data['total cost_rreh'].append(cost_RREH)
                        aggregated_data['total cost_be'].append(cost_BE)

                        # Retrieve and plot the installed renewable capacities
                        print(f"Retrieving and plotting installed capacities for {scenario_name} with time horizon {args.timehorizon}")
                        plant_capacities = analyze_and_plot_capacities(scenario_name, data, results_path, args.timehorizon, wacc_label, args.report)

                        # Aggregate plant capacities
                        # Direct mapping for non-hydro capacities
                        for capacity_type in ["wind_onshore_rreh", "wind_offshore_rreh", "wave_rreh", "solar_rreh", "battery_flow", "battery_stock", "Electrolysis"]:
                            aggregated_data[capacity_type].append(plant_capacities.get(capacity_type, "NA"))

                        # Handle hydro capacities
                        # Adapt these keys based on how they're actually named in your returned 'plant_capacities'
                        hydro_mappings = {
                            "HYDRO_PLANT_03h_RREH": "hydro_3h_rreh",
                            "HYDRO_PLANT_03j_RREH": "hydro_3j_rreh",
                            "HYDRO_PLANT_05h_RREH": "hydro_5h_rreh",
                        }
                        hydro_rreh_sum = 0
                        for original_key, new_key in hydro_mappings.items():
                            # Use 'NA' if not found
                            capacity = plant_capacities.get(original_key, "NA")
                            aggregated_data[new_key].append(capacity)
                            if isinstance(capacity, (int, float)):
                                hydro_rreh_sum += capacity
                        aggregated_data['hydro_rreh'].append(hydro_rreh_sum)

                        # Retrieve and plot the installed capacities
                        print(f"Computing the price of {scenario_name} with time horizon {args.timehorizon}")
                        price_per_mwh, demand_in_twh = calculate_price_per_mwh(data, tot_cost, scenario_name, args.timehorizon)

                        # # Convert demand_in_twh to demand_in_twh
                        # demand_in_twh = demand_in_twh / 1e6

                        aggregated_data['price per mwh'].append(price_per_mwh)
                        aggregated_data['demand in twh'].append(demand_in_twh)
                        
                        # Plot cost breakdown
                        print(f"Plotting the cost breakdown of {scenario_name} with time horizon {args.timehorizon}")                
                        cost_details_RREH = cost_rreh_detailed(data, RREH_nodes)
                        plot_cost_breakdown(cost_details_RREH, demand_in_twh, 'Synthetic Methane (RREH) Cost Breakdown (€/MWh)', results_path, scenario_name, args.timehorizon, wacc_label, 'RREH' ,args.report)
                        cost_details_BE = cost_rreh_detailed(data, BE_nodes)
                        plot_cost_breakdown(cost_details_BE, demand_in_twh, 'Synthetic Methane (Belgium) Cost Breakdown (€/MWh)', results_path, scenario_name, args.timehorizon, wacc_label, 'BE', args.report)

                        # Plot energy production dynamic
                        plot_production_dynamics(data, results_path, scenario_name, args.timehorizon, wacc_label)
                        
                        # Plot basins dynamic
                        if scenario in ['hydro', 'combined']:
                            plot_basin_dynamics(data, results_path, scenario_name, args.timehorizon, wacc_label)

                        # Plot storage dynamic
                        plot_storage_dynamics(data, results_path, scenario_name, args.timehorizon, wacc_label)

                        # Retrieve and plot the installed side techs capacities
                        print(f"Retrieving and plotting installed side tech capacities for {scenario_name} with time horizon {args.timehorizon}")
                        tech_capacities = analyze_and_plot_tech_capacities(scenario_name, data, results_path, args.timehorizon, wacc_label, args.report)

                        for tech_type in ["DAC", "Methanation"]:
                            aggregated_data[tech_type].append(tech_capacities.get(tech_type.lower().replace(" ", "_"), "NA"))  # Assuming the keys in tech_capacities are lowercase with underscores

        all_data.append(aggregated_data)

        df = pd.DataFrame(all_data)  # Ensuring data is in a list to form a single row
        df.to_csv(csv_path, index=False)

        print(f"Data for {scenario} appended to {csv_path}")

# =============== ============== ===============
# ============= CSV FILES ANALYSIS =============
# =============== ============== ===============

def prepare_plot_data(df, column_name):
    scenarios_expanded = []
    constant_values = []
    diff_values = []

    for index, row in df.iterrows():
        scenarios = ast.literal_eval(row['scenario'])
        values = ast.literal_eval(row[column_name])

        for scenario, value in zip(scenarios, values):
            scenario_parts = scenario.split('_')
            scenario_name = '_'.join(scenario_parts[:-1])
            scenario_type = scenario_parts[-1]

            if scenario_type == 'constant':
                constant_values.append(value)
            elif scenario_type == 'diff':
                diff_values.append(value)

            if scenario_type == 'diff':
                scenarios_expanded.append(scenario_name)

    return pd.DataFrame({
        'Scenario': scenarios_expanded,
        f'{column_name} constant': constant_values,
        f'{column_name} diff': diff_values
    })

def plot_price_intervals(df, time_horizon, report=False):
    # Prepare the data
    plot_data = prepare_plot_data(df, 'price per mwh')

    # Filter to include only the specified scenarios
    valid_scenarios = ['hydro_wind', 'germany', 'algeria', 'spain']
    plot_data = plot_data[plot_data['Scenario'].isin(valid_scenarios)]

    # Rename 'hydro_wind' scenario to 'greenland'
    plot_data.loc[plot_data['Scenario'] == 'hydro_wind', 'Scenario'] = 'greenland'

    scenarios = plot_data['Scenario'].tolist()
    constant_values = plot_data['price per mwh constant'].tolist()
    diff_values = plot_data['price per mwh diff'].tolist()

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(scenarios))
    bar_width = 0.35
    opacity = 0.7
    bars1 = ax.bar(x - bar_width/2, constant_values, bar_width, alpha=opacity, color='tab:blue')
    colors = ['tab:green' if constant - diff > 0 else 'tab:red' for constant, diff in zip(constant_values, diff_values)]
    bars2 = ax.bar(x + bar_width/2, diff_values, bar_width, alpha=opacity, color=colors)

    ax.set_xlabel('Country')
    ax.set_ylabel('Price per MWh (€)')
    if not report:
        ax.set_title('Price per MWh Intervals per Country')
    ax.set_xticks(x)
    ax.set_xticklabels([scenario.capitalize() for scenario in scenarios])
    red_patch = mpatches.Patch(color='tab:red', label='Differentiated (Worse)', alpha=0.7)
    blue_patch = mpatches.Patch(color='tab:blue', label='Constant (7%)', alpha=0.7)
    green_patch = mpatches.Patch(color='tab:green', label='Differentiated (Better)', alpha=0.7)
    ax.legend(title='WACC is', handles=[blue_patch, green_patch, red_patch], loc='upper left')

    for i, (constant_value, diff_value, scenario) in enumerate(zip(constant_values, diff_values, scenarios)):
        percent_change = ((diff_value - constant_value) / constant_value) * 100
        sign = "+" if percent_change > 0 else ""
        annotation_text = f"{sign}{percent_change:.0f}%"
        max_val = max(constant_value, diff_value)
        annotation_y = max_val + 5
        color = 'black'

        ax.text(i, annotation_y, annotation_text, ha='center', bbox=dict(facecolor='white', alpha=0.8))
        ax.text(i - bar_width/2, constant_value / 2, f"{round(constant_value, 2)}", ha='center', va='center', color='white')
        ax.text(i + bar_width/2, diff_value / 2, f"{round(diff_value, 2)}", ha='center', va='center', color='white')

    img_folder_path = f'scripts/results/img/{time_horizon}'
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    if report:
        plt.savefig(os.path.join(img_folder_path, 'price_intervals.pdf'), bbox_inches='tight', format='pdf')
    else:
        plt.tight_layout()
        plt.savefig(os.path.join(img_folder_path, 'price_intervals.png'), bbox_inches='tight', format='png')

    # Write data to LaTeX table in a .txt file
    table_data = plot_data.rename(columns={'Scenario': 'Country', 'price per mwh constant': 'Price at WACC constant', 'price per mwh diff': 'Price at WACC differentiated'})[['Country', 'Price at WACC constant', 'Price at WACC differentiated']].reset_index(drop=True)

    styled = (table_data.style
          .format({"Price at WACC constant": "{:.2f}", "Price at WACC differentiated": "{:.2f}"})
          .format_index(escape="latex", axis=0)
          .hide(axis=0))

# Export the styled DataFrame to LaTeX
    styled.to_latex(
        os.path.join(img_folder_path, 'latex.txt'),
        position_float='centering',
        hrules=True
    )

    plt.close(fig)

def plot_technology_capacities_and_prices(df, time_horizon, report=False):
    valid_scenarios = ['wind_onshore', 'wind_offshore', 'wave', 'hydro']

    # Prepare the price data
    price_data = prepare_plot_data(df, 'price per mwh')
    price_data = price_data[price_data['Scenario'].isin(valid_scenarios)]
    price_data = price_data[['Scenario', 'price per mwh constant']]

    # Prepare the capacity data
    capacity_data = pd.DataFrame()

    for valid_scenario in valid_scenarios:
        capacity_col = f'{valid_scenario}_rreh'

        temp_data = prepare_plot_data(df, capacity_col)
        temp_data = temp_data[temp_data['Scenario'] == valid_scenario]  # Select rows with matching scenario
        temp_data = temp_data[['Scenario', f'{capacity_col} constant']].rename(
            columns={f'{capacity_col} constant': 'installed capacity'}
        )
        capacity_data = pd.concat([capacity_data, temp_data], ignore_index=True)

    # Combine the two dataframes on 'Scenario'
    plot_data = pd.merge(price_data, capacity_data, on='Scenario')

    # Ensure the values are numeric
    plot_data['installed capacity'] = pd.to_numeric(plot_data['installed capacity'], errors='coerce')
    plot_data['price per mwh constant'] = pd.to_numeric(plot_data['price per mwh constant'], errors='coerce')

    print(plot_data)

    # Settings for the dual bars
    n = len(plot_data)
    index = np.arange(n)
    bar_width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Colors and aesthetics
    color_capacity = 'tab:blue'
    color_price = 'tab:red'

    # Plot installed capacity
    capacity_bars = ax1.bar(index - bar_width / 2, plot_data['installed capacity'].round(3), bar_width,
                            color=color_capacity, label='Installed Capacity')

    # Create a twin Axes sharing the x-axis for the price per MWh
    ax2 = ax1.twinx()
    price_bars = ax2.bar(index + bar_width / 2, plot_data['price per mwh constant'].round(3), bar_width,
                         color=color_price, label='Price per MWh', alpha=0.7)

    # Labeling and aesthetics
    ax1.set_xlabel('Technology')
    ax1.set_ylabel('Installed Capacity (GW)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax2.set_ylabel('Price per MWh (€)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks(index)
    ax1.set_xticklabels(plot_data['Scenario'])

    # Annotate the bars with their values
    for bar in capacity_bars:
        height = bar.get_height()
        ax1.annotate(f'{height}',
                     xy=(bar.get_x() + bar_width / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    for bar in price_bars:
        height = bar.get_height()
        ax2.annotate(f'{height}',
                     xy=(bar.get_x() + bar_width / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    if not report:
        plt.title('Installed Capacity and Price per MWh by Isolated Technology')
    fig.tight_layout()

    img_folder_path = f'scripts/results/img/{time_horizon}'
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    if report:
        plt.savefig(os.path.join(img_folder_path, 'solo_tech_capa.pdf'), bbox_inches='tight', format='pdf')
    else:
        plt.tight_layout()
        plt.savefig(os.path.join(img_folder_path, 'solo_tech_capa.png'), bbox_inches='tight', format='png')

    # Write data to LaTeX table in a .txt file
    table_data = plot_data.rename(columns={'Scenario': 'Technology', 'price per mwh constant': 'Price (€/MWh)', 'installed capacity': 'Installed Capacity (GW)'})[['Technology', 'Price (€/MWh)', 'Installed Capacity (GW)']].reset_index(drop=True)

    styled = (table_data.style
          .format({"Price (€/MWh)": "{:.2f}", "Installed Capacity (GW)": "{:.2f}"})
          .format_index(escape="latex", axis=0)
          .hide(axis=0))

    # Export the styled DataFrame to LaTeX
    with open(os.path.join(img_folder_path, 'latex.txt'), 'a') as file:
            file.write('\n')  # Add a space before
            styled.to_latex(file, 
                            position_float='centering',
                            hrules=True
            )
            file.write('\n')  # Add a space after

    plt.close(fig)

def plot_stacked_bar_costs(df, time_horizon, report=False):
    exclude_scenarios = ['spain', 'germany', 'algeria']

    # Prepare the total cost data
    demand_in_twh = 12.970272171668919

    # Retrieve cost_rreh and cost_be with prepare_plot_data
    cost_rreh = prepare_plot_data(df, 'total cost_rreh')
    cost_be = prepare_plot_data(df, 'total cost_be')
    total_cost_mwh = prepare_plot_data(df, 'price per mwh')

    # Convert the column with 'constant' in it to floats
    cost_rreh['price_gr'] = cost_rreh['total cost_rreh constant'].astype(float)
    cost_be['price_be'] = cost_be['total cost_be constant'].astype(float)
    total_cost_mwh['price_mwh'] = total_cost_mwh['price per mwh constant'].astype(float)


    # Convert to €/MWh
    cost_rreh['price_gr'] = cost_rreh['price_gr'] / demand_in_twh
    cost_be['price_be'] = cost_be['price_be'] / demand_in_twh

    # Remove the column with 'diff'
    cost_rreh.drop('total cost_rreh diff', axis=1, inplace=True)
    cost_be.drop('total cost_be diff', axis=1, inplace=True)
    total_cost_mwh.drop('price per mwh diff', axis=1, inplace=True)

    # Merge dataframes
    cost_data = pd.merge(cost_rreh, cost_be, on='Scenario')
    cost_data = pd.merge(cost_data, total_cost_mwh, on='Scenario')
    cost_data = cost_data[~cost_data['Scenario'].isin(exclude_scenarios)]

    # Settings for the stacked bars
    n = len(cost_data)
    index = np.arange(n)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the stacked bars with the specified colors
    rreh_bars = ax.bar(index, cost_data['price_gr'], bar_width, label='Cost GR (€MWh)', color='tab:blue', alpha=0.7)
    be_bars = ax.bar(index, cost_data['price_be'], bar_width,
                     bottom=cost_data['price_gr'], label='Cost BE (€MWh)', color='tab:red', alpha=0.7)

    # Add horizontal line
    ax.axhline(y=156.14, color='darkred', linestyle='--', label='Reference Case')

    # Labeling and aesthetics
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Cost (€MWh)', color='black')
    ax.tick_params(axis='y', labelcolor='black')
    ax.set_xticks(index)
    ax.set_xticklabels(cost_data['Scenario'])

    # Annotate the bars with their values
    for bar, cost in zip(rreh_bars, cost_data['price_mwh']):
        height = bar.get_height()
        ax.annotate(f'{cost:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    if not report:
        plt.title('Total Costs per MWh by Scenario')
    fig.tight_layout()
    img_folder_path = f'scripts/results/img/{time_horizon}'
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    if report:
        plt.savefig(os.path.join(img_folder_path, 'costs_scenarios_constant.pdf'), bbox_inches='tight', format='pdf')
    else:
        plt.tight_layout()
        plt.savefig(os.path.join(img_folder_path, 'costs_scenarios_constant.png'), bbox_inches='tight', format='png')

    # Write data to LaTeX table in a .txt file
    table_data = cost_data.rename(columns={'Scenario': 'Scenario', 'price_gr': 'Price GR (€/MWh)', 'price_be': 'Price BE (€/MWh)', 'price_mwh': 'Total Price (€/MWh)'})[['Scenario', 'Price GR (€/MWh)', 'Price BE (€/MWh)', 'Total Price (€/MWh)']].reset_index(drop=True)

    styled = (table_data.style
          .format({"Price GR (€/MWh)": "{:.2f}", "Price BE (€/MWh)": "{:.2f}", "Total Price (€/MWh)": "{:.2f}"})
          .format_index(escape="latex", axis=0)
          .hide(axis=0))

    # Export the styled DataFrame to LaTeX
    with open(os.path.join(img_folder_path, 'latex.txt'), 'a') as file:
            file.write('\n')  # Add a space before
            styled.to_latex(file, 
                            position_float='centering',
                            hrules=True
            )
            file.write('\n')  # Add a space after

    plt.close(fig)

def analyze_csv_files(args):
    results_dir = 'scripts/results/'
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        # Extract the time horizon from the file name using regex
        match = re.search(r'scenario_analysis_results_(\d+).csv', csv_file)
        if match:
            time_horizon = match.group(1)
            print(f"Analyzing CSV file: {csv_file} with time horizon: {time_horizon}")
            
            # Check if 'latex.txt' exists in the folder
            latex_file_path = os.path.join(results_dir, 'img', time_horizon, 'latex.txt')
            if os.path.exists(latex_file_path):
                # If it exists, erase its content
                with open(latex_file_path, 'w') as latex_file:
                    latex_file.write('')
            else:
                # If it doesn't exist, create a new file
                os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
                with open(latex_file_path, 'w') as latex_file:
                    pass

            # Load the CSV file for analysis
            df = pd.read_csv(os.path.join(results_dir, csv_file))

            plot_price_intervals(df, time_horizon, args.report)

            plot_technology_capacities_and_prices(df, time_horizon, args.report)

            plot_stacked_bar_costs(df, time_horizon, args.report)
    

# =============== ============== ===============
# ==================== MAIN ====================
# =============== ============== ===============

def main():
    args = parse_args()

    if args.mode == 'analyze_json':
        analyze_json_files(args)
    elif args.mode == 'analyze_csv':
        analyze_csv_files(args)

if __name__ == "__main__":
    main()

