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
    plant_capacities = calculate_plant_capacities(scenario, d, results_path, timehorizon, wacc_label)
    plot_and_save_capacities(plant_capacities, scenario, results_path, timehorizon, report, wacc_label)
    
    return plant_capacities

def calculate_plant_capacities(scenario, d, results_path, timehorizon, wacc_label):
    plant_capacities = {}

    if hasattr(d.solution.elements, "WAVE_PLANT_RREH"):
        wave_units = d.solution.elements.WAVE_PLANT_RREH.variables.num_units.values[0]
        wave_rp = d.model.nodes.WAVE_PLANT_RREH.parameters.unit_rated_power[0]
        plant_capacities['WAVE_PLANT_RREH'] = wave_units * wave_rp

    # Dynamically calculate capacities based on available data in 'd'
    ls_nodes = list(d.model.nodes.keys())
    excluded_elements = ['WAVE_PLANT_RREH']
    
    for element_name in ls_nodes:
        if element_name not in excluded_elements:
            element = getattr(d.solution.elements, element_name, None)
            if element is not None and hasattr(element, 'variables'):
                capacity_values = None
                capacity_stock_values = None
                capacity_flow_values = None

                try:
                    capacity_values = element.variables.capacity.values
                except AttributeError:
                    pass

                if capacity_values is None:
                    try:
                        capacity_stock_values = element.variables.capacity_stock.values
                        capacity_flow_values = element.variables.capacity_flow.values
                    except AttributeError:
                        pass
                
                if capacity_values:
                    plant_capacities[element_name] = np.sum(capacity_values)
                elif capacity_stock_values and capacity_flow_values:
                    plant_capacities[element_name+'_stock'] = np.sum(capacity_stock_values)
                    plant_capacities[element_name+'_flow'] = np.sum(capacity_flow_values)


    # Filter plant capacities to include only those with _flow and _stock
    regular_capacities = {k: v for k, v in plant_capacities.items() if '_flow' not in k and '_stock' not in k}
    flow_stock_capacities = {k: v for k, v in plant_capacities.items() if '_flow' in k or '_stock' in k}

    # Define the directory for saving images and LaTeX files
    img_folder_name = f"img_{timehorizon}/{wacc_label}" if timehorizon else "img_all"
    img_folder_path = os.path.join(results_path, img_folder_name)
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    
    tex_file_path = os.path.join(img_folder_path, f"{scenario}__capacities.tex")
    with open(tex_file_path, 'w') as tex_file:
        tex_file.write("\\begin{table}[h]\n")
        tex_file.write("    \\centering\n")
        tex_file.write("    \\begin{minipage}{0.45\\textwidth}\n")
        tex_file.write("    \\centering\n")
        tex_file.write("    \\begin{tabular}{lc}\n")
        tex_file.write("        \\toprule\n")
        tex_file.write("        Category & Capacity (GW) \\\\\n")
        tex_file.write("        \\midrule\n")
        for category, value in flow_stock_capacities.items():
            tex_file.write(f"        {category.replace('_', ' ').title()} & {value:.3f} \\\\\n")
        tex_file.write("        \\bottomrule\n")
        tex_file.write("    \\end{tabular}\n")
        tex_file.write(f"    \\caption{{Flow and Stock Capacities for {scenario} scenario}}\n")
        tex_file.write(f"    \\label{{tab:capacities_flow_stock_{scenario}}}\n")
        tex_file.write("    \\end{minipage}\n")
        tex_file.write("    \\hspace{0.05\\textwidth}\n")
        tex_file.write("    \\begin{minipage}{0.45\\textwidth}\n")
        tex_file.write("    \\centering\n")
        tex_file.write("    \\begin{tabular}{lc}\n")
        tex_file.write("        \\toprule\n")
        tex_file.write("        Category & Capacity (GW) \\\\\n")
        tex_file.write("        \\midrule\n")
        for category, value in regular_capacities.items():
            tex_file.write(f"        {category.replace('_', ' ').title()} & {value:.3f} \\\\\n")
        tex_file.write("        \\bottomrule\n")
        tex_file.write("    \\end{tabular}\n")
        tex_file.write(f"    \\caption{{Regular Capacities for {scenario} scenario}}\n")
        tex_file.write(f"    \\label{{tab:capacities_regular_{scenario}}}\n")
        tex_file.write("    \\end{minipage}\n")
        tex_file.write("\\end{table}\n")

    return plant_capacities

def plot_and_save_capacities(capacities, scenario, results_path, timehorizon, report, wacc_label):
    # Calculate total hydro capacity and prepare other categories and their capacities
    hydro_keys = [k for k in capacities if "HYDRO_PLANT" in k]
    total_hydro_capacity = sum(capacities[k] for k in hydro_keys)
    
    categories = ['Onshore Wind', 'Offshore Wind', 'Solar', 'Wave', 'Battery Flow', 'Battery Stock', 'Electrolysis']
    capacity_values = [
        capacities.get('ON_WIND_PLANTS_RREH', 0),
        capacities.get('OFF_WIND_PLANTS_RREH', 0),
        capacities.get('SOLAR_PV_PLANTS_RREH',0),
        capacities.get('WAVE_PLANT_RREH', 0),
        capacities.get('BATTERY_STORAGE_RREH_flow', 0),
        capacities.get('BATTERY_STORAGE_RREH_stock', 0),
        capacities.get('ELECTROLYSIS_PLANTS_RREH', 0),        
    ]
    
    # Only add the total hydro capacity if there are hydro plants
    if hydro_keys:
        categories.append('Total Hydro')
        capacity_values.append(total_hydro_capacity)
    
    # Filter out categories with zero capacity except for Battery Flow, Battery Stock, and Electrolysis
    filtered_categories = []
    filtered_capacity_values = []
    for category, value in zip(categories, capacity_values):
        if value > 0 or category in ['Battery Flow', 'Battery Stock', 'Electrolysis']:
            filtered_categories.append(category)
            filtered_capacity_values.append(value)

    # Define the directory for saving images
    img_folder_name = f"img_{timehorizon}/{wacc_label}" if timehorizon else "img_all"
    img_folder_path = os.path.join(results_path, img_folder_name)
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)

    # Setup the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(filtered_categories, filtered_capacity_values, color=[plt.cm.viridis(0.75) for _ in filtered_categories])
    
    ax.set_ylabel('Installed Capacity (GW)')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Adjust y-axis to fit the labels
    ax.set_ylim(0, max(filtered_capacity_values) * 1.2)
    
    # Add labels above the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f} GW',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords='offset points',
                    ha='center', va='bottom')
    
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
    if scenario == 'hydrogen' or scenario == 'hydrogen_algeria':
        demand_in_twh = sum(d.solution.elements.LIQUEFIED_HYDROGEN_REGASIFICATION_BE.variables.hydrogen.values) * 0.0394
    elif scenario == 'ammonia' or scenario == 'ammonia_algeria':
        demand_in_twh = sum(d.solution.elements.LIQUEFIED_NH3_REGASIFICATION_BE.variables.ammonia.values) * 0.00625
    elif scenario == 'methanol' or scenario == 'methanol_algeria':
        demand_in_twh = sum(d.solution.elements.LIQUEFIED_METHANOL_CARRIERS_RREH.variables.liquefied_methanol_out.values) * 0.00639
    elif scenario == "germany_pipe" or scenario == "spain_pipe":
        demand_in_twh = sum(d.solution.elements.METHANATION_PLANTS_RREH.variables.methane.values) * 0.015441  # kt/h * MWh/kg
    else:
        demand_in_twh = sum(d.solution.elements.LIQUEFIED_METHANE_REGASIFICATION.variables.methane.values) * 0.015441  # kt/h * MWh/kg

    # Compute price per MWh in euros
    price_per_mwh = tot_cost / demand_in_twh # M€/GW /TWH <=> €/MWh
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

def plot_production_dynamics(d, plant_capacities, results_path, scenario, timehorizon, wacc_label, start_date='2015-01-01', end_date='2020-01-01'):
    production_data = {}

    # Determine which production types to include based on the scenario
    production_types_mapping = {
        'combined': ['OFF_WIND_PLANTS_RREH', 'ON_WIND_PLANTS_RREH', 'HYDRO_PLANT_03h_RREH', 'HYDRO_PLANT_03j_RREH', 'HYDRO_PLANT_05h_RREH', 'WAVE_PLANT_RREH'],
        'hydro_wind': ['OFF_WIND_PLANTS_RREH', 'ON_WIND_PLANTS_RREH', 'HYDRO_PLANT_03h_RREH', 'HYDRO_PLANT_03j_RREH', 'HYDRO_PLANT_05h_RREH'],
        'wind_onshore': ['ON_WIND_PLANTS_RREH'],
        'wind_offshore': ['OFF_WIND_PLANTS_RREH'],
        'hydro': ['HYDRO_PLANT_03h_RREH', 'HYDRO_PLANT_03j_RREH', 'HYDRO_PLANT_05h_RREH'],
        'wave': ['WAVE_PLANT_RREH'],
        'spain': ['ON_WIND_PLANTS_RREH', 'SOLAR_PV_PLANTS_RREH'],
        'spain_pipe': ['ON_WIND_PLANTS_RREH', 'SOLAR_PV_PLANTS_RREH'],
        'algeria': ['ON_WIND_PLANTS_RREH', 'SOLAR_PV_PLANTS_RREH'],
        'germany': ['ON_WIND_PLANTS_RREH', 'SOLAR_PV_PLANTS_RREH'],
        'germany_pipe': ['ON_WIND_PLANTS_RREH', 'SOLAR_PV_PLANTS_RREH'],
        'ammonia': ['OFF_WIND_PLANTS_RREH', 'ON_WIND_PLANTS_RREH', 'HYDRO_PLANT_03h_RREH', 'HYDRO_PLANT_03j_RREH', 'HYDRO_PLANT_05h_RREH'],
        'hydrogen': ['OFF_WIND_PLANTS_RREH', 'ON_WIND_PLANTS_RREH', 'HYDRO_PLANT_03h_RREH', 'HYDRO_PLANT_03j_RREH', 'HYDRO_PLANT_05h_RREH'],
        'methanol': ['OFF_WIND_PLANTS_RREH', 'ON_WIND_PLANTS_RREH', 'HYDRO_PLANT_03h_RREH', 'HYDRO_PLANT_03j_RREH', 'HYDRO_PLANT_05h_RREH'],
        'ammonia_algeria': ['ON_WIND_PLANTS_RREH', 'SOLAR_PV_PLANTS_RREH'],
        'hydrogen_algeria': ['ON_WIND_PLANTS_RREH', 'SOLAR_PV_PLANTS_RREH'],
        'methanol_algeria': ['ON_WIND_PLANTS_RREH', 'SOLAR_PV_PLANTS_RREH']
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
    fig, axs = plt.subplots(num_plots, 1, figsize=(15, num_plots * 3), sharex=True)
    if num_plots == 1:
        axs = [axs]
        colors = [plt.cm.viridis(0.5)]
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, num_plots))

    for ax, ((production_type, production_array), color) in zip(axs, zip(production_data.items(), colors)):
        df = pd.DataFrame({'date': date_range, 'production': production_array})
        df.set_index('date', inplace=True)
        data_series = df.resample('2W').mean()

        if data_series['production'].max() < 0.001:
            ax.axhline(y=0, color='k', linestyle='--')
            ax.set_ylim(-0.1, 0.1)
        else:
            ax.fill_between(data_series.index, data_series['production'], color=color, label='Electricity Production')
        
        # Calculate capacity factor
        capacity = plant_capacities.get(production_type, 0)
        total_production = sum(production_array)
        normalized_production = total_production / (capacity * timehorizon)
        capacity_factor_line = capacity * normalized_production

        ax.axhline(y=capacity_factor_line, color='red', linestyle='--', label='Average Production')
        
        ax.set_ylabel('GWh', fontsize=14)  # Increase font size of y-axis label
        ax.tick_params(axis='both', which='major', labelsize=12)  # Increase font size of tick labels
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        ax.tick_params(axis='x', which='major', length=10)
        ax.tick_params(axis='x', which='minor', length=5)
        plt.setp(ax.get_xticklabels(minor=True), rotation=0, ha='center')
        ax.legend(loc='lower right', fontsize=12)  # Increase font size of legend

    # Set the x-axis limits to the start and end dates
    plt.xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
    
    fig.tight_layout()
    fig.autofmt_xdate()
    
    img_folder_name = f"img_{timehorizon}/{wacc_label}" if timehorizon else "img_all"
    img_folder_path = os.path.join(results_path, img_folder_name)
    os.makedirs(img_folder_path, exist_ok=True)
    production_fig_path = os.path.join(img_folder_path, f"{scenario}_production_dynamics.pdf")
    fig.savefig(production_fig_path)
    plt.close(fig)

    production_data_normalized = {}

    for production_type, data in production_data.items():
        capacity = plant_capacities.get(production_type, 0)
        total_production = sum(data)
        normalized_production = total_production / (capacity * timehorizon)
        production_data_normalized[production_type] = normalized_production

    # Calculate onshore and offshore data for load factors
    if scenario in ['hydro_wind', 'methanol', 'ammonia', 'hydrogen']:
        onshore_data = np.array(d.solution.elements.ON_WIND_PLANTS_RREH.variables.electricity.values) / \
                       np.array(d.solution.elements.ON_WIND_PLANTS_RREH.variables.capacity.values)

        offshore_data = np.array(d.solution.elements.OFF_WIND_PLANTS_RREH.variables.electricity.values) / \
                        np.array(d.solution.elements.OFF_WIND_PLANTS_RREH.variables.capacity.values)

        total_hours = len(onshore_data)
        dates = pd.date_range(start='2015-01-01', periods=total_hours, freq='H')
        data = pd.DataFrame({
            'onshore': onshore_data,
            'offshore': offshore_data,
            'datetime': dates
        })

        # Calculate the combined capacity factor
        onshore_capacity = plant_capacities.get('ON_WIND_PLANTS_RREH', 0)
        offshore_capacity = plant_capacities.get('OFF_WIND_PLANTS_RREH', 0)
        data['combined_capacity_factor'] = (data['onshore'] * onshore_capacity + data['offshore'] * offshore_capacity)

        # Resample data to weekly averages
        weekly_data = data.resample('2W', on='datetime').mean()

        fig, ax1 = plt.subplots(figsize=(15, 3))

        # Plotting the weekly averaged data
        ax1.plot(weekly_data.index, weekly_data['onshore'], label='Onshore', linewidth=2)
        ax1.plot(weekly_data.index, weekly_data['offshore'], label='Offshore', linewidth=2)
        ax1.set_ylim(0, 1)  # Set the y-axis limits from 0 to 1
        ax1.set_xlim(pd.Timestamp('2015-01-01'), pd.Timestamp('2020-01-01'))  # Set the x-axis limits from 2015-01-01 to 2020-01-01
        ax1.set_ylabel('Load Factor', fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
        ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        ax1.tick_params(axis='x', which='major', length=10)
        ax1.tick_params(axis='x', which='minor', length=5)
        plt.setp(ax1.get_xticklabels(minor=True), rotation=0, ha='center')
        handles1, labels1 = ax1.get_legend_handles_labels()

        ax2 = ax1.twinx()
        ax2.fill_between(weekly_data.index, 0, weekly_data['onshore'] * onshore_capacity, color='lightgrey', alpha=0.3, label='Onshore Production')
        ax2.fill_between(weekly_data.index, weekly_data['onshore'] * onshore_capacity, weekly_data['combined_capacity_factor'], color='darkgrey', alpha=0.3, label='Offshore Production')
        ax2.set_ylim(0, max(weekly_data['combined_capacity_factor'])*1.1)
        ax2.set_ylabel('Combined Production (GWh)', fontsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        handles2, labels2 = ax2.get_legend_handles_labels()

        ax1.legend(handles=handles1, labels=labels1, loc='upper left', fontsize=12)
        ax2.legend(handles=handles2, labels=labels2, loc='upper right', fontsize=12)

        fig.tight_layout()
        img_folder_name = f"img_{timehorizon}/{wacc_label}" if timehorizon else "img_all"
        img_folder_path = os.path.join(results_path, img_folder_name)
        if not os.path.exists(img_folder_path):
            os.makedirs(img_folder_path)
        plt.savefig(os.path.join(img_folder_path, f'diurnal_seasonal_wind_{scenario}_{wacc_label}.pdf'), bbox_inches='tight')
        plt.close()

    # Calculate onshore and offshore data for load factors
    if scenario in ['algeria', 'methanol_algeria', 'ammonia_algeria', 'hydrogen_algeria']:
        onshore_data = np.array(d.solution.elements.ON_WIND_PLANTS_RREH.variables.electricity.values) / \
                       np.array(d.solution.elements.ON_WIND_PLANTS_RREH.variables.capacity.values)

        pv_data = np.array(d.solution.elements.SOLAR_PV_PLANTS_RREH.variables.electricity.values) / \
                        np.array(d.solution.elements.SOLAR_PV_PLANTS_RREH.variables.capacity.values)

        total_hours = len(onshore_data)
        dates = pd.date_range(start='2015-01-01', periods=total_hours, freq='H')
        onshore_data = onshore_data[:len(dates)]
        pv_data = pv_data[:len(dates)]
        data = pd.DataFrame({
            'onshore': onshore_data,
            'solar pv': pv_data,
            'datetime': dates
        })

        # Calculate the combined capacity factor
        onshore_capacity = plant_capacities.get('ON_WIND_PLANTS_RREH', 0)
        pv_capacity = plant_capacities.get('SOLAR_PV_PLANTS_RREH', 0)
        data['combined_capacity_factor'] = (data['onshore'] * onshore_capacity + data['solar pv'] * pv_capacity)

        # Resample data to weekly averages
        weekly_data = data.resample('2W', on='datetime').mean()

        fig, ax1 = plt.subplots(figsize=(15, 3))

        # Plotting the weekly averaged data
        ax1.plot(weekly_data.index, weekly_data['onshore'], label='Onshore', linewidth=2)
        ax1.plot(weekly_data.index, weekly_data['solar pv'], label='Solar PV', linewidth=2)
        ax1.set_ylim(0, 1)  # Set the y-axis limits from 0 to 1
        ax1.set_xlim(pd.Timestamp('2015-01-01'), pd.Timestamp('2020-01-01'))  # Set the x-axis limits from 2015-01-01 to 2020-01-01
        ax1.set_ylabel('Load Factor', fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
        ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        ax1.tick_params(axis='x', which='major', length=10)
        ax1.tick_params(axis='x', which='minor', length=5)
        plt.setp(ax1.get_xticklabels(minor=True), rotation=0, ha='center')
        handles1, labels1 = ax1.get_legend_handles_labels()

        ax2 = ax1.twinx()
        ax2.fill_between(weekly_data.index, 0, weekly_data['onshore'] * onshore_capacity, color='lightgrey', alpha=0.5, label='Onshore Production')
        ax2.fill_between(weekly_data.index, weekly_data['onshore'] * onshore_capacity, weekly_data['combined_capacity_factor'], color='darkgrey', alpha=0.5, label='Solar PV Production')
        ax2.set_ylabel('Combined Production (GWh)', fontsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.set_ylim(0, max(weekly_data['combined_capacity_factor'])*1.1)
        handles2, labels2 = ax2.get_legend_handles_labels()

        ax1.legend(handles=handles1, labels=labels1, loc='upper left', fontsize=12)
        ax2.legend(handles=handles2, labels=labels2, loc='upper right', fontsize=12)

        fig.tight_layout()
        img_folder_name = f"img_{timehorizon}/{wacc_label}" if timehorizon else "img_all"
        img_folder_path = os.path.join(results_path, img_folder_name)
        if not os.path.exists(img_folder_path):
            os.makedirs(img_folder_path)
        plt.savefig(os.path.join(img_folder_path, f'diurnal_seasonal_wind_{scenario}_{wacc_label}.pdf'), bbox_inches='tight')
        plt.close()

    return production_data_normalized

def plot_basin_dynamics(d, results_path, scenario, timehorizon, wacc_label, start_date='2015-01-01'):
    if scenario not in ['hydro', 'hydro_wind', 'combined']:
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

        plt.xticks(rotation=0)
        plt.xlabel('Time')
        fig.tight_layout()

        img_folder_name = f"img_{timehorizon}/{wacc_label}" if timehorizon else "img_all"
        img_folder_path = os.path.join(results_path, img_folder_name)
        os.makedirs(img_folder_path, exist_ok=True)
        basin_fig_path = os.path.join(img_folder_path, f"{scenario}_{basin_name}_dynamics.png")
        fig.savefig(basin_fig_path)
        plt.close(fig)

def plot_storage_dynamics(d, results_path, scenario, timehorizon, wacc_label, start_date='2015-01-01'):
    date_range = pd.date_range(start=start_date, periods=timehorizon, freq='H')

    battery_data = np.array(d.solution.elements.BATTERY_STORAGE_RREH.variables.electricity_stored.values)
    hydrogen_data = np.array(d.solution.elements.HYDROGEN_STORAGE_RREH.variables.hydrogen_stored.values)

    # Check if battery data is all zeros
    if np.all(battery_data <= 0.1):
        data_frames = [hydrogen_data]
        colors = ['royalblue']
        storage_labels = ['Hydrogen Storage Dynamics (kt)']
        n_plots = 1
    else:
        data_frames = [battery_data, hydrogen_data]
        colors = ['orange', 'royalblue']
        storage_labels = ['Battery Storage Dynamics (GWh)', 'Hydrogen Storage Dynamics (kt)']
        n_plots = 2

    fig, axs = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)

    if n_plots == 1:
        axs = [axs]

    for ax, data, color, label in zip(axs, data_frames, colors, storage_labels):
        data_series = pd.Series(data).rolling(window=24).mean()
        if data_series.max() < 0.001:
            ax.axhline(y=0, color='k', linestyle='--')
            ax.set_ylim(-0.1, 0.1)
        else:
            ax.plot(date_range, data_series, label=label, color=color)
        
        ax.set_ylabel('GWh' if 'Battery' in label else 'kt', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(loc='upper right', fontsize=12)

    axs[-1].xaxis.set_major_locator(mdates.YearLocator())
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axs[-1].xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    axs[-1].xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    axs[-1].tick_params(axis='x', which='major', length=10)
    axs[-1].tick_params(axis='x', which='minor', length=5)
    plt.setp(axs[-1].get_xticklabels(minor=True), rotation=0, ha='center')

    fig.tight_layout()

    img_folder_name = f"img_{timehorizon}/{wacc_label}" if timehorizon else "img_all"
    img_folder_path = os.path.join(results_path, img_folder_name)
    os.makedirs(img_folder_path, exist_ok=True)
    
    storage_dynamics_fig_path = os.path.join(img_folder_path, f"{scenario}_storage_dynamics.png")
    storage_dynamics_pdf_path = os.path.join(img_folder_path, f"{scenario}_storage_dynamics.pdf")
    
    fig.savefig(storage_dynamics_fig_path, bbox_inches='tight')
    fig.savefig(storage_dynamics_pdf_path, bbox_inches='tight')
    plt.close(fig)

def plot_hydrogen_dynamics(d, results_path, scenario, timehorizon, wacc_label):
    def debug_print(name, array):
        if isinstance(array, np.ndarray):
            print(name, array.shape)
            print(array[:5])

    colors_prod = ['steelblue', 'lightblue']

    electrolyzer_H2 = np.array(d.solution.elements.ELECTROLYSIS_PLANTS_RREH.variables.hydrogen.values)
    h2_storage_out = np.array(d.solution.elements.HYDROGEN_STORAGE_RREH.variables.hydrogen_out.values)
    production_data = pd.DataFrame({
        'Electrolyzer output': electrolyzer_H2,
        '$H_{2}$ Storage output': h2_storage_out,
    })

    hydrogen_prod = np.sum(electrolyzer_H2) #in kt

    if scenario in ['ammonia', 'ammonia_algeria']:
        nh3_prod = np.array(d.solution.elements.NH3_PROD_RREH.variables.hydrogen.values)
        total_consumption = nh3_prod
        labels = ['Electrolyzer output', '$H_{2}$ Storage output', 'NH3 Production']

        consumption_data = pd.DataFrame({
            'Total Consumption': total_consumption,
        })
    elif scenario in ['hydrogen', 'hydrogen_algeria']:
        liquefaction_h2 = np.array(d.solution.elements.HYDROGEN_LIQUEFACTION_PLANTS_RREH.variables.hydrogen.values)
        total_consumption = liquefaction_h2
        labels = ['Electrolyzer H2', '$H_{2}$ Storage output', 'Liquefaction H2']

        consumption_data = pd.DataFrame({
            'Total Consumption': total_consumption,
        })
    elif scenario in ['methanol', 'methanol_algeria']:
        methanol_H2 = np.array(d.solution.elements.METHANOL_PLANTS_RREH.variables.hydrogen.values)
        dac_H2 = np.array(d.solution.elements.DIRECT_AIR_CAPTURE_PLANTS_RREH.variables.hydrogen.values)
        total_consumption = methanol_H2 + dac_H2
        labels = ['Electrolyzer H2', '$H_{2}$ Storage output', 'Methanol H2', 'DAC H2']

        consumption_data = pd.DataFrame({
            'Total Consumption': total_consumption,
        })
    elif scenario in ['hydro_wind', 'algeria']:
        methanation_H2 = np.array(d.solution.elements.METHANATION_PLANTS_RREH.variables.hydrogen.values)
        dac_H2 = np.array(d.solution.elements.DIRECT_AIR_CAPTURE_PLANTS_RREH.variables.hydrogen.values)
        total_consumption = methanation_H2 + dac_H2
        labels = ['Electrolyzer H2', '$H_{2}$ Storage output', 'Methanation H2', 'DAC H2']

        consumption_data = pd.DataFrame({
            'Total Consumption': total_consumption,
        })
    else:
        return

    # Create a time index with datetime for resampling
    time_index = pd.date_range(start='2015-01-01', periods=len(production_data), freq='H')
    production_data['Time'] = time_index
    consumption_data['Time'] = time_index
    production_data.set_index('Time', inplace=True)
    consumption_data.set_index('Time', inplace=True)

    # Resample the data to 2-week averages
    production_data = production_data.resample('2W').mean()
    consumption_data = consumption_data.resample('2W').mean()

    # Plot the data
    fig, ax1 = plt.subplots(figsize=(14, 3))

    # Plot production data
    cumulative_sum = np.zeros_like(production_data.iloc[:, 0])
    for column, color in zip(production_data.columns, colors_prod):
        ax1.fill_between(production_data.index, cumulative_sum, cumulative_sum + production_data[column], label=column, color=color, alpha=0.8)
        cumulative_sum += production_data[column]
    ax1.set_ylabel('$H_{2}$ Flows (kt/h)', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # Plot total consumption line
    ax1.plot(consumption_data.index, consumption_data['Total Consumption'], label='Total $H_{2}$ Consumption', color='red', linewidth=2)
    
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    ax1.tick_params(axis='x', which='major', length=10)
    ax1.tick_params(axis='x', which='minor', length=5)
    plt.setp(ax1.get_xticklabels(minor=True), rotation=0, ha='center')
    plt.xlim(pd.Timestamp('2015-01-01'), pd.Timestamp('2020-01-01'))

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles1, labels1 = handles1[::-1], labels1[::-1]
    ax1.legend(handles1, labels1, loc='lower right', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    img_folder_name = f"img_{timehorizon}/{wacc_label}" if timehorizon else "img_all"
    img_folder_path = os.path.join(results_path, img_folder_name)
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    plt.savefig(os.path.join(img_folder_path, f'hydrogen_dynamics_{scenario}.pdf'), bbox_inches='tight')
    plt.close()

    return hydrogen_prod

def plot_typical_weeks_prod(d, results_path, scenario, timehorizon, report=False):
    hours_per_week = 24 * 7

    # Number of complete weeks in the dataset
    num_weeks = timehorizon // hours_per_week

    # Helper function to calculate a typical week
    def typical_week(data):
        # Reshape the data to separate weeks
        reshaped_data = data[:num_weeks * hours_per_week].reshape(num_weeks, hours_per_week)
        # Calculate the mean across all weeks
        return np.mean(reshaped_data, axis=0)

    # Electricity Production (GW)
    onshore = np.array(d.solution.elements.ON_WIND_PLANTS_RREH.variables.electricity.values)
    tot_elec = np.sum(onshore)
    onshore = typical_week(onshore)

    if scenario in ['hydro_wind', 'ammonia', 'hydrogen', 'methanol']:
        offshore = np.array(d.solution.elements.OFF_WIND_PLANTS_RREH.variables.electricity.values)
        hydro = np.sum([np.array(d.solution.elements.HYDRO_PLANT_03h_RREH.variables.electricity.values),
                        np.array(d.solution.elements.HYDRO_PLANT_03j_RREH.variables.electricity.values),
                        np.array(d.solution.elements.HYDRO_PLANT_05h_RREH.variables.electricity.values)], axis=0)
        
        tot_elec += np.sum(offshore)
        tot_elec += np.sum(hydro)

        offshore = typical_week(offshore)
        hydro = typical_week(hydro)
    else:
        pv = np.array(d.solution.elements.SOLAR_PV_PLANTS_RREH.variables.electricity.values)

        tot_elec += np.sum(pv)
        
        pv = typical_week(pv)

    battery_out = typical_week(np.array(d.solution.elements.BATTERY_STORAGE_RREH.variables.electricity_out.values))

    # Electricity Consumption & PtG Production (GW)
    electrolysis = np.array(d.solution.elements.ELECTROLYSIS_PLANTS_RREH.variables.electricity.values)
    electrolyzer_elec = np.sum(electrolysis)
    electrolysis = typical_week(electrolysis)

    desalination = typical_week(np.array(d.solution.elements.DESALINATION_PLANTS_RREH.variables.electricity.values))
    battery_in = typical_week(np.array(d.solution.elements.BATTERY_STORAGE_RREH.variables.electricity_in.values))
    h2o_storage = typical_week(np.array(d.solution.elements.WATER_STORAGE_RREH.variables.electricity.values))
    h2_storage = typical_week(np.array(d.solution.elements.HYDROGEN_STORAGE_RREH.variables.electricity.values))

    h2_liquefaction_elec = 0

    if scenario in ['ammonia', 'ammonia_algeria']:
        conv = typical_week(np.array(d.solution.elements.ASU_RREH.variables.electricity.values))
        n2_storage = typical_week(np.array(d.solution.elements.NITROGEN_STORAGE_RREH.variables.electricity.values))
        prod = typical_week(np.array(d.solution.elements.NH3_PROD_RREH.variables.ammonia.values) * 6.25)
        ammonia = typical_week(np.array(d.solution.elements.NH3_PROD_RREH.variables.electricity.values))
    elif scenario in ['hydrogen', 'hydrogen_algeria']:
        prod = typical_week(np.array(d.solution.elements.ELECTROLYSIS_PLANTS_RREH.variables.hydrogen.values) * 39.4)
        h2_liquefaction = np.array(d.solution.elements.HYDROGEN_LIQUEFACTION_PLANTS_RREH.variables.electricity.values)
        h2_liquefaction_elec = np.sum(h2_liquefaction)
        h2_liquefaction = typical_week(h2_liquefaction)

        conv = 0
    else:
        conv = typical_week(np.array(d.solution.elements.DIRECT_AIR_CAPTURE_PLANTS_RREH.variables.electricity.values))
        co2_storage = typical_week(np.array(d.solution.elements.CARBON_DIOXIDE_STORAGE_RREH.variables.electricity.values))
        if scenario in ['methanol', 'methanol_algeria']:
            prod = typical_week(np.array(d.solution.elements.METHANOL_PLANTS_RREH.variables.methanol.values) * 6.39)
            methanol = typical_week(np.array(d.solution.elements.METHANOL_PLANTS_RREH.variables.elec.values))
        else:
            prod = typical_week(np.array(d.solution.elements.METHANATION_PLANTS_RREH.variables.methane.values) * 15.441)
            ch4_liquefaction = typical_week(np.array(d.solution.elements.METHANE_LIQUEFACTION_PLANTS_RREH.variables.electricity.values))

    # Set up the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    time = np.arange(hours_per_week) / 24

    # Plotting production
    ax1.fill_between(time, 0, onshore, color='#00BA9B', label='Onshore Wind', alpha=0.8)
    if scenario in ['hydro_wind', 'ammonia', 'hydrogen', 'methanol']:
        ax1.fill_between(time, onshore, onshore + offshore, color='#B4E35D', label='Offshore Wind', alpha=0.8)
        ax1.fill_between(time, onshore + offshore, onshore + offshore + hydro, color='#61DEE3', label='Hydro', alpha=0.8)
        ax1.fill_between(time, onshore + offshore + hydro, onshore + offshore + hydro + battery_out, color='#FEDC00', label='Battery Discharge', alpha=0.8)
    else:
        ax1.fill_between(time, onshore, onshore + pv, color='#B4E35D', label='PV', alpha=0.8)
        ax1.fill_between(time, onshore + pv, onshore + pv + battery_out, color='#FEDC00', label='Battery Discharge', alpha=0.8)

    ax3 = ax1.twinx()
    ax3.plot(time, prod, 'k-', label='Production (PtG)', linewidth=2)
    ax3.set_ylim(bottom=0)

    # Plotting consumption
    cumulative_sum = np.zeros_like(electrolysis)
    if np.any(electrolysis != 0):
        ax2.fill_between(time, cumulative_sum, cumulative_sum + electrolysis, color='#00ABD3', label='Electrolysis', alpha=0.8)
        cumulative_sum += electrolysis
    if np.any(desalination != 0):
        ax2.fill_between(time, cumulative_sum, cumulative_sum + desalination, color='#61DEDF', label='Desalination', alpha=0.8)
        cumulative_sum += desalination
    if np.any(conv != 0):
        if scenario in ['ammonia', 'ammonia_algeria']:
            label = 'ASU'
        elif scenario in ['methanol', 'methanol_algeria']:
            label = 'e-Methanol'
        else:
            label = 'Methanation'
        ax2.fill_between(time, cumulative_sum, cumulative_sum + conv, color='#FF8002', label=label, alpha=0.8)
        cumulative_sum += conv
    if np.any(battery_in != 0):
        ax2.fill_between(time, cumulative_sum, cumulative_sum + battery_in, color='#FEDC00', label='Battery Charge', alpha=0.8)
        cumulative_sum += battery_in
    if np.any(h2o_storage != 0):
        ax2.fill_between(time, cumulative_sum, cumulative_sum + h2o_storage, color='#00BA9B', label='H2O Storage', alpha=0.8)
        cumulative_sum += h2o_storage
    if np.any(h2_storage != 0):
        ax2.fill_between(time, cumulative_sum, cumulative_sum + h2_storage, color='#B4E35D', label='H2 Storage', alpha=0.8)
        cumulative_sum += h2_storage

    # Scenario-specific additions
    if scenario in ['ammonia', 'ammonia_algeria'] and np.any(n2_storage != 0):
        ax2.fill_between(time, cumulative_sum, cumulative_sum + n2_storage, color='#9260D0', label='N2 Storage', alpha=0.8)
        cumulative_sum += n2_storage
        if np.any(ammonia != 0):
            ax2.fill_between(time, cumulative_sum, cumulative_sum + ammonia, color='violet', label='Ammonia Production', alpha=0.8)
            cumulative_sum += ammonia
    elif scenario in ['hydrogen', 'hydrogen_algeria'] and np.any(h2_liquefaction != 0):
        ax2.fill_between(time, cumulative_sum, cumulative_sum + h2_liquefaction, color='#00394E', label='H2 Liquefaction', alpha=0.8)
        cumulative_sum += h2_liquefaction
    else:
        if np.any(co2_storage != 0):
            ax2.fill_between(time, cumulative_sum, cumulative_sum + co2_storage, color='#FF4B00', label='CO2 Storage', alpha=0.8)
            cumulative_sum += co2_storage
        if scenario in ['methanol', 'methanol_algeria'] and np.any(methanol != 0):
            ax2.fill_between(time, cumulative_sum, cumulative_sum + methanol, color='violet', label='Methanol Production', alpha=0.8)
            cumulative_sum += methanol
        elif scenario != ['methanol', 'methanol_algeria'] and np.any(ch4_liquefaction != 0):
            ax2.fill_between(time, cumulative_sum, cumulative_sum + ch4_liquefaction, color='#D1D7CE', label='CH4 Liquefaction', alpha=0.8)
            cumulative_sum += ch4_liquefaction

    ax4 = ax2.twinx()
    ax4.plot(time, prod, 'k-', label='Production (PtG)', linewidth=2)
    ax4.set_ylim(bottom=0)

    # Set labels, titles, and grid
    ax1.set_xlabel('Time (days)', fontsize=14)
    ax1.set_ylabel('Power in GWh', fontsize=14)
    ax2.set_xlabel('Time (days)', fontsize=14)
    ax2.set_ylabel('Power in GWh', fontsize=14)
    ax3.set_ylabel('Production Output in GWh (HHV)', fontsize=14)
    ax4.set_ylabel('Production Output in GWh (HHV)', fontsize=14)

    # Adjust tick label sizes
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    ax4.tick_params(axis='both', which='major', labelsize=12)
    ax2.set_ylim(ax1.get_ylim())

    # Ensure that the right y-axes match the left y-axes and start at 0
    ax1_ylim = ax1.get_ylim()
    ax2_ylim = ax2.get_ylim()
    ax3.set_ylim(0, ax1_ylim[1])
    ax4.set_ylim(0, ax2_ylim[1])
    ax2.set_ylim(0, ax1_ylim[1])

    # Set grid for better visibility
    ax1.grid(True)
    ax2.grid(True)

    # Handling legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()
    handles4, labels4 = ax4.get_legend_handles_labels()

    # Reverse the order of the legend handles and labels
    handles1, labels1 = handles1[::-1], labels1[::-1]
    handles2, labels2 = handles2[::-1], labels2[::-1]

    fig.subplots_adjust(wspace=0.3)  # Increase the spacing between subplots

    ax1.legend(handles1 + handles3, labels1 + labels3, loc='lower right', fontsize=12)
    ax2.legend(handles2 + handles4, labels2 + labels4, loc='lower right', fontsize=12)

    # Save the plot with naming convention
    img_folder_name = f"img_{timehorizon}/constant" if timehorizon else "img_all"
    img_folder_path = os.path.join(results_path, img_folder_name)
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    
    file_format = 'pdf' if report else 'png'
    filename = f"{scenario}_elec_prod_cons.{file_format}"  # Define a meaningful filename
    plot_path = os.path.join(img_folder_path, filename)
    
    if report:
        ax1.set_title('')
        ax2.set_title('')
    else:
        ax1.set_title(f'Electricity Production for {scenario} Scenario')
        ax2.set_title(f'Electricity Consumption for {scenario} Scenario')
    
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    return tot_elec, electrolyzer_elec, h2_liquefaction_elec

def conversion_factor(d, scenario):
    production_types_mapping = {
        'combined': ['OFF_WIND_PLANTS_RREH', 'ON_WIND_PLANTS_RREH', 'HYDRO_PLANT_03h_RREH', 'HYDRO_PLANT_03j_RREH', 'HYDRO_PLANT_05h_RREH', 'WAVE_PLANT_RREH'],
        'hydro_wind': ['OFF_WIND_PLANTS_RREH', 'ON_WIND_PLANTS_RREH', 'HYDRO_PLANT_03h_RREH', 'HYDRO_PLANT_03j_RREH', 'HYDRO_PLANT_05h_RREH'],
        'wind_onshore': ['ON_WIND_PLANTS_RREH'],
        'wind_offshore': ['OFF_WIND_PLANTS_RREH'],
        'hydro': ['HYDRO_PLANT_03h_RREH', 'HYDRO_PLANT_03j_RREH', 'HYDRO_PLANT_05h_RREH'],
        'wave': ['WAVE_PLANT_RREH'],
        'spain': ['ON_WIND_PLANTS_RREH', 'SOLAR_PV_PLANTS_RREH'],
        'spain_pipe': ['ON_WIND_PLANTS_RREH', 'SOLAR_PV_PLANTS_RREH'],
        'algeria': ['ON_WIND_PLANTS_RREH', 'SOLAR_PV_PLANTS_RREH'],
        'germany': ['ON_WIND_PLANTS_RREH', 'SOLAR_PV_PLANTS_RREH'],
        'germany_pipe': ['ON_WIND_PLANTS_RREH', 'SOLAR_PV_PLANTS_RREH'],
        'ammonia': ['OFF_WIND_PLANTS_RREH', 'ON_WIND_PLANTS_RREH', 'HYDRO_PLANT_03h_RREH', 'HYDRO_PLANT_03j_RREH', 'HYDRO_PLANT_05h_RREH'],
        'hydrogen': ['OFF_WIND_PLANTS_RREH', 'ON_WIND_PLANTS_RREH', 'HYDRO_PLANT_03h_RREH', 'HYDRO_PLANT_03j_RREH', 'HYDRO_PLANT_05h_RREH'],
        'methanol': ['OFF_WIND_PLANTS_RREH', 'ON_WIND_PLANTS_RREH', 'HYDRO_PLANT_03h_RREH', 'HYDRO_PLANT_03j_RREH', 'HYDRO_PLANT_05h_RREH'],
        'ammonia_algeria': ['ON_WIND_PLANTS_RREH', 'SOLAR_PV_PLANTS_RREH'],
        'hydrogen_algeria': ['ON_WIND_PLANTS_RREH', 'SOLAR_PV_PLANTS_RREH'],
        'methanol_algeria': ['ON_WIND_PLANTS_RREH', 'SOLAR_PV_PLANTS_RREH']
    }
    
    # Get the relevant plant types for the given scenario
    plant_types = production_types_mapping.get(scenario, [])
    # Calculate total electricity produced by relevant plants
    electricity = 0
    for plant_type in plant_types:
        electricity += np.sum(getattr(d.solution.elements, plant_type).variables.electricity.values)

    if scenario in ['wind_onshore', 'wind_offshore', 'hydro', 'wave', 'spain', 'germany', 'algeria','combined','hydro_wind']:
        prod_ch4 = sum(d.solution.elements.LIQUEFIED_METHANE_REGASIFICATION.variables.methane.values) * 15.441
        conversion_factor = prod_ch4 / electricity
    elif scenario == 'ammonia' or scenario == 'ammonia_algeria':
        prod_nh3 = sum(d.solution.elements.LIQUEFIED_NH3_REGASIFICATION_BE.variables.ammonia.values) * 6.25
        conversion_factor = prod_nh3 / electricity
    elif scenario == 'methanol' or scenario == 'methanol_algeria':
        prod_ch3oh = sum(d.solution.elements.LIQUEFIED_METHANOL_CARRIERS_RREH.variables.liquefied_methanol_out.values) * 6.39
        conversion_factor = prod_ch3oh / electricity
    elif scenario == 'hydrogen' or scenario == 'hydrogen_algeria':
        prod_h2 = sum(d.solution.elements.LIQUEFIED_HYDROGEN_REGASIFICATION_BE.variables.hydrogen.values) * 39.4
        conversion_factor = prod_h2 / electricity
    else:
        conversion_factor = None

    return round(conversion_factor*100,2)

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
    
    scenarios = ['wind_onshore', 'wind_offshore', 'wave', 'hydro', 'hydro_wind', 'combined', 'spain', 'algeria', 'germany', 
                 'germany_pipe', 'spain_pipe', 'ammonia', 'methanol', 'hydrogen', 'ammonia_algeria', 'hydrogen_algeria', 'methanol_algeria']

    base_path = 'models'  # Adjust this path as necessary

    csv_path = f'scripts/results/scenario_analysis_results_{args.timehorizon}.csv'
    csv_exists = os.path.isfile(csv_path)
    if csv_exists:
        os.remove(csv_path)

    all_data = []
    
    capacity_factors_df = pd.DataFrame(columns=['Country', 'Onshore Wind Turbines', 'Offshore Wind Turbines', 'Wave Energy Converters', 'Solar PV', 'Hydropower'])
    conversion_factors_df = pd.DataFrame(columns=['Energy Carrier', 'Conversion Factor'])
    
    for scenario in scenarios:
        aggregated_data = {
            "scenario": [],
            "ON_WIND_PLANTS_RREH": [],
            "OFF_WIND_PLANTS_RREH": [],
            "SOLAR_PV_PLANTS_RREH": [],
            "WAVE_PLANT_RREH": [],
            "HYDRO_PLANT_03h_RREH": [],
            "HYDRO_PLANT_03j_RREH": [],
            "HYDRO_PLANT_05h_RREH": [],
            "hydro_rreh": [],
            "BATTERY_STORAGE_RREH_flow": [],
            "BATTERY_STORAGE_RREH_stock": [],
            "ELECTROLYSIS_PLANTS_RREH": [],
            "hydrogen production (kt)": [],
            "electricity production (GWh)": [],
            "electrolyzer consumption (GWh)": [],
            "H2 liquefaction consumption (GWh)": [],
            "price per mwh": [],
            "total cost": [],
            "total cost_rreh": [],
            "total cost_be": [],
            "demand in twh": [],
            "other capacities": []
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
                
                            # Remove the part that is just digits and the 'results.json' part
                            scenario_parts = [part for part in parts if not part.isdigit() and not part.endswith('.json')]
                            # Join the remaining parts back together to get the scenario name
                            scenario_name = '_'.join(scenario_parts)
                        
                            aggregated_data['scenario'].append(scenario_name)

                            # Initial check
                            print(f"Plot of balances for {scenario_name} with time horizon {args.timehorizon}")
                            if scenario not in ['ammonia', 'methanol', 'hydrogen', 'germany_pipe', 'spain_pipe', 'ammonia_algeria', 'methanol_algeria', 'hydrogen_algeria']:
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
                            for capacity_type in ["ON_WIND_PLANTS_RREH", "OFF_WIND_PLANTS_RREH", "WAVE_PLANT_RREH", "SOLAR_PV_PLANTS_RREH", "BATTERY_STORAGE_RREH_flow", "BATTERY_STORAGE_RREH_stock", "ELECTROLYSIS_PLANTS_RREH"]:
                                aggregated_data[capacity_type].append(plant_capacities.get(capacity_type, "NA"))

                            # Handle hydro capacities
                            hydro_rreh_sum = 0
                            for capacity_type in ["HYDRO_PLANT_03h_RREH", "HYDRO_PLANT_03j_RREH", "HYDRO_PLANT_05h_RREH"]:
                                capacity = plant_capacities.get(capacity_type, "NA")
                                aggregated_data[capacity_type].append(capacity)
                                if isinstance(capacity, (int, float)):
                                    hydro_rreh_sum += capacity
                            aggregated_data['hydro_rreh'].append(hydro_rreh_sum)

                            # Append other capacities
                            other_capacities = []
                            for capacity, value in plant_capacities.items():
                                if capacity not in aggregated_data.keys():
                                    other_capacities.append((capacity, value))
                            aggregated_data['other capacities'].append(other_capacities)

                            # Retrieve and plot the installed capacities
                            print(f"Computing the price of {scenario_name} with time horizon {args.timehorizon}")
                            price_per_mwh, demand_in_twh = calculate_price_per_mwh(data, tot_cost, scenario, args.timehorizon)

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
                            
                            # Plot basins dynamic
                            if scenario in ['hydro', 'hydro_wind', 'combined']:
                                plot_basin_dynamics(data, results_path, scenario_name, args.timehorizon, wacc_label)

                            # Plot storage dynamic
                            plot_storage_dynamics(data, results_path, scenario_name, args.timehorizon, wacc_label)

                            # Plot energy production dynamic
                            capacity_factors = plot_production_dynamics(data, plant_capacities, results_path, scenario_name, args.timehorizon, wacc_label)

                            hydropower_values = [capacity_factors.get('HYDRO_PLANT_03h_RREH', 'NA'), capacity_factors.get('HYDRO_PLANT_03j_RREH', 'NA'), capacity_factors.get('HYDRO_PLANT_05h_RREH', 'NA')]
                            if any(value != 'NA' for value in hydropower_values):
                                hydropower_sum = sum(float(value) if value != 'NA' else 0 for value in hydropower_values)
                                hydropower_avg = hydropower_sum / 3
                            else:
                                hydropower_avg = 'NA'

                            hydrogen_prod = plot_hydrogen_dynamics(data, results_path, scenario, args.timehorizon, wacc_label)

                            aggregated_data['hydrogen production (kt)'].append(hydrogen_prod)

                            if scenario not in ['germany_pipe', 'spain_pipe'] and wacc_label == 'constant':
                                new_row = pd.DataFrame({
                                    'Country': ['greenland' if scenario == 'hydro_wind' else scenario],
                                    'Onshore Wind Turbines': [capacity_factors.get('ON_WIND_PLANTS_RREH', 'NA')],
                                    'Offshore Wind Turbines': [capacity_factors.get('OFF_WIND_PLANTS_RREH', 'NA')],
                                    'Wave Energy Converters': [capacity_factors.get('WAVE_PLANT_RREH', 'NA')],
                                    'Solar PV': [capacity_factors.get('SOLAR_PV_PLANTS_RREH', 'NA')],
                                    'Hydropower': [hydropower_avg]
                                })
                                capacity_factors_df = pd.concat([capacity_factors_df, new_row], ignore_index=True)

                                new_row = pd.DataFrame({
                                    'Energy Carrier': [scenario],
                                    'Conversion Factor': [conversion_factor(data, scenario)]
                                })
                                conversion_factors_df = pd.concat([conversion_factors_df, new_row], ignore_index=True)
                            
                            if scenario in ['hydro_wind', 'ammonia', 'methanol', 'hydrogen', 'algeria', 'ammonia_algeria', 'methanol_algeria', 'hydrogen_algeria']:
                                tot_elec, electrolyzer_elec, h2_liquefaction_elec = plot_typical_weeks_prod(data, results_path, scenario, args.timehorizon, args.report)
                                aggregated_data['electricity production (GWh)'].append(tot_elec)
                                aggregated_data['electrolyzer consumption (GWh)'].append(electrolyzer_elec)
                                aggregated_data['H2 liquefaction consumption (GWh)'].append(h2_liquefaction_elec)
                            
        all_data.append(aggregated_data)

        df = pd.DataFrame(all_data)  # Ensuring data is in a list to form a single row
        df.to_csv(csv_path, index=False)

        print(f"Data for {scenario} appended to {csv_path}")

    # Check if 'cf_latex.txt' exists in the folder
    latex_file_path = f'scripts/results/img/{args.timehorizon}/factors_latex.txt'
    if os.path.exists(latex_file_path):
        # If it exists, erase its content
        with open(latex_file_path, 'w') as latex_file:
            latex_file.write('')
    else:
        # If it doesn't exist, create a new file
        os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
        with open(latex_file_path, 'w') as latex_file:
            pass

    styled = (capacity_factors_df.style
                .format_index(escape="latex", axis=0)
                .hide(axis=0))

    # Export the styled DataFrame to LaTeX
    with open(latex_file_path, 'a') as file:
        file.write('\n')  # Add a space before
        styled.to_latex(file, 
                        position_float='centering',
                        hrules=True
        )
        file.write('\n')  # Add a space after

    with open(latex_file_path, 'a') as file:
        file.write("\\begin{table}[h]\n")
        file.write("\\centering\n")
        file.write("\\begin{tabular}{cc}\n")
        file.write("\\toprule\n")
        file.write("Chemical & Conversion Factor (\\%) \\\\ \n")
        file.write("\\midrule\n")
        for index, row in conversion_factors_df.iterrows():
            file.write(f"{row['Energy Carrier']} & {row['Conversion Factor']}\\% \\\\ \n")
        file.write("\\bottomrule\n")
        file.write("\\end{tabular}\n")
        file.write("\\caption{System Conversion Factors}\n")
        file.write("\\label{tab:system_conversion_factors}\n")
        file.write("\\end{table}\n")
        file.write('\n')

# =============== ============== ===============
# ============= CSV FILES ANALYSIS =============
# =============== ============== ===============

def prepare_plot_data(df, column_name, add_constant_suffix=False):
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
                if add_constant_suffix:
                    scenarios_expanded.append(scenario_name + '_constant')
                else:
                    scenarios_expanded.append(scenario_name)
            elif scenario_type == 'diff':
                diff_values.append(value)

    # Adjust lengths if necessary to avoid errors
    min_length = min(len(scenarios_expanded), len(constant_values))
    scenarios_expanded = scenarios_expanded[:min_length]
    constant_values = constant_values[:min_length]

    # Handle missing diff_values by filling with NaNs
    if len(diff_values) < min_length:
        diff_values.extend([float('nan')] * (min_length - len(diff_values)))
    else:
        diff_values = diff_values[:min_length]

    # Print the prepared data for debugging
    prepared_data = pd.DataFrame({
        'Scenario': scenarios_expanded,
        f'{column_name} constant': constant_values,
        f'{column_name} diff': diff_values
    })

    return prepared_data

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
    bars1 = ax.bar(x - bar_width / 2, constant_values, bar_width, alpha=opacity, color='tab:blue')
    colors = ['tab:green' if constant - diff > 0 else 'tab:red' for constant, diff in zip(constant_values, diff_values)]
    bars2 = ax.bar(x + bar_width / 2, diff_values, bar_width, alpha=opacity, color=colors)

    ax.set_ylabel('Price per MWh (€)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    if not report:
        ax.set_title('Price per MWh Intervals per Country', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([scenario.capitalize() for scenario in scenarios], fontsize=12)
    red_patch = mpatches.Patch(color='tab:red', label='Differentiated (> 7%)', alpha=0.7)
    blue_patch = mpatches.Patch(color='tab:blue', label='Constant (7%)', alpha=0.7)
    green_patch = mpatches.Patch(color='tab:green', label='Differentiated (< 7%)', alpha=0.7)
    ax.legend(title='WACC is', handles=[blue_patch, green_patch, red_patch], loc='upper left', fontsize=12)

    for i, (constant_value, diff_value, scenario) in enumerate(zip(constant_values, diff_values, scenarios)):
        percent_change = ((diff_value - constant_value) / constant_value) * 100
        sign = "+" if percent_change > 0 else ""
        annotation_text = f"{sign}{percent_change:.0f}%"
        max_val = max(constant_value, diff_value)
        annotation_y = max_val + 5
        color = 'black'

        ax.text(i, annotation_y, annotation_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        ax.text(i - bar_width / 2, constant_value / 2, f"{round(constant_value, 2)}", ha='center', va='center', color='white', fontsize=12)
        ax.text(i + bar_width / 2, diff_value / 2, f"{round(diff_value, 2)}", ha='center', va='center', color='white', fontsize=12)

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
    valid_columns = {
        'wind_onshore': 'ON_WIND_PLANTS_RREH',
        'wind_offshore': 'OFF_WIND_PLANTS_RREH',
        'wave': 'WAVE_PLANT_RREH',
        'hydro': 'hydro_rreh'
    }

    # Prepare the price data
    price_data = prepare_plot_data(df, 'price per mwh')
    price_data = price_data[price_data['Scenario'].isin(valid_scenarios)]
    price_data = price_data[['Scenario', 'price per mwh constant']]

    # Prepare the capacity data
    capacity_data = pd.DataFrame()

    for scenario in valid_scenarios:
        capacity_col = valid_columns[scenario]

        temp_data = prepare_plot_data(df, capacity_col)
        temp_data = temp_data[temp_data['Scenario'] == scenario]  # Select rows with matching scenario
        temp_data = temp_data[['Scenario', f'{capacity_col} constant']].rename(
            columns={f'{capacity_col} constant': 'installed capacity'}
        )
        capacity_data = pd.concat([capacity_data, temp_data], ignore_index=True)

    # Combine the two dataframes on 'Scenario'
    plot_data = pd.merge(price_data, capacity_data, on='Scenario')

    # Ensure the values are numeric
    plot_data['installed capacity'] = pd.to_numeric(plot_data['installed capacity'], errors='coerce')
    plot_data['price per mwh constant'] = pd.to_numeric(plot_data['price per mwh constant'], errors='coerce')

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
                            color=color_capacity, label='Installed Capacity', alpha=0.7)

    # Create a twin Axes sharing the x-axis for the price per MWh
    ax2 = ax1.twinx()
    price_bars = ax2.bar(index + bar_width / 2, plot_data['price per mwh constant'].round(3), bar_width,
                         color=color_price, label='Price per MWh', alpha=0.7)

    # Labeling and aesthetics
    ax1.set_ylabel('Installed Capacity (GW)', color='black', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
    ax2.set_ylabel('Price per MWh (€)', color='black', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='black', labelsize=12)
    ax1.set_xticks(index)
    ax1.set_xticklabels([scenario.capitalize() for scenario in plot_data['Scenario']], fontsize=12)

    # Annotate the bars with their values
    for bar in capacity_bars:
        height = bar.get_height()
        ax1.annotate(f'{height}',
                     xy=(bar.get_x() + bar_width / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=12)

    for bar in price_bars:
        height = bar.get_height()
        ax2.annotate(f'{height}',
                     xy=(bar.get_x() + bar_width / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=12)

    # Combine the legends for ax1 and ax2
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles + handles2, labels + labels2, loc="upper right", fontsize=12)

    if not report:
        plt.title('Installed Capacity and Price per MWh by Isolated Technology', fontsize=16)
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

def plot_installed_capacity_specific_technologies(df, time_horizon, report=False):
    scenarios = ['hydro_wind_constant', 'methanol_constant', 'ammonia_constant', 'hydrogen_constant']
    scenario_names = {
        'hydro_wind_constant': 'Methane',
        'methanol_constant': 'Methanol',
        'ammonia_constant': 'Ammonia',
        'hydrogen_constant': 'Hydrogen'
    }
    technologies = ['ON_WIND_PLANTS_RREH', 'OFF_WIND_PLANTS_RREH', 'hydro_rreh', 'ELECTROLYSIS_PLANTS_RREH']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']  # Colors for each scenario

    # Step 1: Determine the maximum installed capacity value
    max_capacity = 0
    for tech in technologies:
        if tech not in df.columns:
            continue

        plot_data = prepare_plot_data(df, tech, add_constant_suffix=True)
        plot_data = plot_data[plot_data['Scenario'].isin(scenarios)]
        plot_data['Scenario'] = plot_data['Scenario'].map(scenario_names)

        if not plot_data.empty:
            max_capacity = max(max_capacity, plot_data[f'{tech} constant'].max())

    # Set the y-axis limit to 1.2 times the maximum value
    y_max = 1.2 * max_capacity

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))

    for ax, tech in zip(axs.flat, technologies):
        if tech not in df.columns:
            continue

        plot_data = prepare_plot_data(df, tech, add_constant_suffix=True)

        plot_data = plot_data[plot_data['Scenario'].isin(scenarios)]

        plot_data['Scenario'] = plot_data['Scenario'].map(scenario_names)

        if plot_data.empty:
            continue

        scenarios_list = plot_data['Scenario'].tolist()
        capacities = plot_data[f'{tech} constant'].tolist()

        # Assign colors to each scenario by mapping back to the original identifiers
        scenario_colors = [colors[scenarios.index(key)] for key in scenario_names.keys() if scenario_names[key] in scenarios_list]

        ax.bar(scenarios_list, capacities, color=scenario_colors, alpha=0.7)
        ax.set_title(tech)
        ax.set_ylabel('Installed Capacity')
        ax.set_xticklabels(scenarios_list, rotation=0)
        ax.set_ylim(0, y_max)  # Set the same y-axis limit for all subplots

        for i, capacity in enumerate(capacities):
            ax.text(i, capacity + 0.03, f"{round(capacity, 3)}", ha='center')

    plt.tight_layout()
    img_folder_path = f'scripts/results/img/{time_horizon}'
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    if report:
        plt.savefig(os.path.join(img_folder_path, 'installed_capacity_specific_technologies.pdf'), bbox_inches='tight', format='pdf')
    else:
        plt.savefig(os.path.join(img_folder_path, 'installed_capacity_specific_technologies.png'), bbox_inches='tight', format='png')

    plt.close(fig)

def plot_cost_comparison_specific_scenarios(df, time_horizon, report=False):
    # Scenarios of interest with updated names
    specific_scenarios = ['hydro_wind', 'methanol', 'hydrogen', 'ammonia']
    scenario_names = {
        'hydro_wind': 'Methane',
        'methanol': 'Methanol',
        'hydrogen': 'Hydrogen',
        'ammonia': 'Ammonia'
    }

    # Prepare the total cost data
    demand_in_twh = round(int(time_horizon) / 8760 * 10)

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
    print("Data before filtering:")
    print(cost_data.head())

    # Debugging step: Print scenarios present in the merged cost_data
    print("Scenarios in merged cost_data:")
    print(cost_data['Scenario'].unique())

    # Filter the data to include only specific scenarios
    cost_data = cost_data[cost_data['Scenario'].isin(specific_scenarios)]
    print("Data after filtering:")
    print(cost_data.head())

    # Map scenario names
    cost_data['Scenario'] = cost_data['Scenario'].map(scenario_names)

    # Debugging step: Print the scenarios after mapping
    print("Scenarios after mapping:")
    print(cost_data['Scenario'].unique())

    if cost_data.empty:
        print("No data available for the specified scenarios. Exiting.")
        return

    # Sort by total cost (sum of price_gr and price_be) in descending order
    cost_data['total_cost'] = cost_data['price_gr'] + cost_data['price_be']
    cost_data = cost_data.sort_values(by='total_cost', ascending=False).drop(columns=['total_cost'])

    print("Plotting cost comparison data:")
    print(cost_data)

    # Regenerate the index variable after sorting
    n = len(cost_data)
    index = np.arange(n)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the stacked bars with the specified colors
    rreh_bars = ax.bar(index, cost_data['price_gr'], bar_width, label='Cost GR (€/MWh)', color='tab:blue', alpha=0.7)
    be_bars = ax.bar(index, cost_data['price_be'], bar_width, bottom=cost_data['price_gr'], label='Cost BE (€/MWh)', color='tab:red', alpha=0.7)

    # Labeling and aesthetics
    ax.set_ylabel('Cost (€MWh)', color='black', fontsize=14)
    ax.tick_params(axis='y', labelcolor='black', labelsize=12)
    ax.set_xticks(index)
    ax.set_xticklabels(cost_data['Scenario'], fontsize=12)

    # Annotate the bars with their values
    for rreh_bar, be_bar, cost in zip(rreh_bars, be_bars, cost_data['price_mwh']):
        rreh_height = rreh_bar.get_height()
        be_height = be_bar.get_height()
        height = rreh_height + be_height
        ax.annotate(f'{cost:.2f}', xy=(rreh_bar.get_x() + rreh_bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=12)

    # Create a single legend combining all the labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", fontsize=12)

    # Adjust the layout to include space for the legend
    fig.tight_layout()

    if not report:
        plt.title('Total Costs per MWh for Specific Scenarios', fontsize=16)
    img_folder_path = f'scripts/results/img/{time_horizon}'
    
    # Ensure the directory exists
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
        print(f"Created directory: {img_folder_path}")

    # Determine file name and path
    file_name = 'costs_specific_scenarios_constant.pdf' if report else 'costs_specific_scenarios_constant.png'
    file_path = os.path.join(img_folder_path, file_name)
    
    # Save the plot
    plt.savefig(file_path, bbox_inches='tight', format='pdf' if report else 'png')
    print(f"Plot saved to: {file_path}")

    plt.close(fig)

def plot_stacked_bar_costs(df, time_horizon, report=False):
    exclude_scenarios = ['spain', 'germany', 'algeria', 'ammonia', 'methanol', 'hydrogen', 'germany_pipe', 'spain_pipe', 'ammonia_algeria', 'methanol_algeria', 'hydrogen_algeria']

    # Prepare the total cost data
    demand_in_twh = round(int(time_horizon) / 8760 * 10)

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
    ax.axhline(y=153.59, color='darkred', linestyle='--', label='Reference Case (Algeria)')

    
    # Labeling and aesthetics
    ax.set_ylabel('Cost (€MWh)', color='black', fontsize=14)
    ax.tick_params(axis='y', labelcolor='black', labelsize=12)
    ax.set_xticks(index)
    ax.set_xticklabels(cost_data['Scenario'], fontsize=12)

    # Annotate the bars with their values
    for bar, cost in zip(rreh_bars, cost_data['price_mwh']):
        height = bar.get_height()
        ax.annotate(f'{cost:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", fontsize=12)
    if not report:
        plt.title('Total Costs per MWh by Scenario', fontsize=16)
    img_folder_path = f'scripts/results/img/{time_horizon}'
    fig.tight_layout()
    # Ensure the directory exists
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
        parts = csv_file.split('_')
        timehorizon = int(parts[-1].rstrip('.csv'))
        if timehorizon == args.timehorizon:
            print(f"Analyzing CSV file: {csv_file} with time horizon: {timehorizon}")
            
            timehorizon=str(timehorizon)
            # Check if 'latex.txt' exists in the folder
            latex_file_path = os.path.join(results_dir, 'img', timehorizon, 'latex.txt')
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

            plot_price_intervals(df, timehorizon, args.report)

            plot_technology_capacities_and_prices(df, timehorizon, args.report)

            plot_stacked_bar_costs(df, timehorizon, args.report)

            plot_cost_comparison_specific_scenarios(df, timehorizon, args.report)

            plot_installed_capacity_specific_technologies(df, timehorizon, args.report)
    

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

