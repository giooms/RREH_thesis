import argparse
import os
import sys
from datetime import datetime
import json

from gboml import GbomlGraph
import gboml.compiler.classes as gcc

# Define the specific scenarios and their respective .gboml files
SCENARIO_FILES_STANDARD_WACC  = {
    'wind_onshore_stdwacc': 'greenland_wind_on.gboml',
    'wind_offshore_stdwacc': 'greenland_wind_off.gboml',
    'wave_stdwacc': 'greenland_wave.gboml',
    'hydro_stdwacc': 'greenland_hydro.gboml',
    'hydro_wind_stdwacc': 'greenland_hydro_wind.gboml',
    'combined_stdwacc': 'greenland_combined.gboml'
}

SCENARIO_FILES_VARIABLE_WACC = {
    'wind_onshore_varwacc': 'greenland_wind_on_varwacc.gboml',
    'wind_offshore_varwacc': 'greenland_wind_off_varwacc.gboml',
    'wave_varwacc': 'greenland_wave_varwacc.gboml',
    'hydro_varwacc': 'greenland_hydro_varwacc.gboml',
    'hydro_wind_varwacc': 'greenland_hydro_wind_varwacc.gboml',
    'combined_varwacc': 'greenland_combined_varwacc.gboml'
}

def run_scenario(scenario, timehorizon, scenario_files):
    """
    Runs the GBOML model for a given scenario based on its .gboml file and saves the results.
    """
    print(f"Running scenario: {scenario}")
    gboml_file_path = scenario_files.get(scenario)
    
    # Determine the base scenario name without WACC type for directory naming
    base_scenario_name = scenario.rsplit('_', 1)[0]  # Splits on the last underscore

    # Construct the full path to the .gboml model file
    gboml_model_full_path = os.path.join("models", base_scenario_name, gboml_file_path)

    gboml_model = GbomlGraph(timehorizon=timehorizon)
    nodes, edges, param = gboml_model.import_all_nodes_and_edges(gboml_model_full_path)
    gboml_model.add_global_parameters(param)
    gboml_model.add_nodes_in_model(*nodes)
    gboml_model.add_hyperedges_in_model(*edges)
    gboml_model.build_model()
    solution, obj, status, solver_info, constr_info, _ = gboml_model.solve_gurobi(opt_file="scripts/analysis/gurobi.txt")
    
    gathered_data = gboml_model.turn_solution_to_dictionary(solver_info, status, solution, obj, constr_info)

    # Construct the path to the scenario-specific results directory
    results_dir_path = f"models/{base_scenario_name}/results"

    # Check if the directory exists; if not, create it
    if not os.path.exists(results_dir_path):
        os.makedirs(results_dir_path)

    # Define the path for the results JSON file
    result_path = f"{results_dir_path}/{scenario}_{timehorizon}_results.json"

    # Save the gathered data to the JSON file
    with open(result_path, "w") as fp:
        json.dump(gathered_data, fp, indent=4)

    print(f"Results saved to {result_path}")


def main():
    parser = argparse.ArgumentParser(description="Run GBOML models for various energy scenarios.")
    parser.add_argument('-s', '--scenario', help="Specific energy scenario to run, or 'all' to run all scenarios.", 
                        default='all', choices=['all'] + list(SCENARIO_FILES_STANDARD_WACC.keys()) + list(SCENARIO_FILES_VARIABLE_WACC.keys()))
    parser.add_argument('-w', '--wacc', help="Type of Weighted Average Cost of Capital (WACC) to use.", 
                        default='all', choices=['standard', 'variable', 'all'])
    parser.add_argument('-t', '--timehorizon', help="Time horizon for the model", 
                        type=int, default=17544, choices=[8760, 17544, 26304, 35064, 43824])  # Default to one year
    args = parser.parse_args()

    # Decide which scenarios to run based on the WACC argument
    if args.wacc == 'standard':
        scenario_files = SCENARIO_FILES_STANDARD_WACC
    elif args.wacc == 'variable':
        scenario_files = SCENARIO_FILES_VARIABLE_WACC
    else:
        scenario_files = {**SCENARIO_FILES_STANDARD_WACC, **SCENARIO_FILES_VARIABLE_WACC}

    # Run scenarios based on the chosen WACC and scenario type
    scenarios_to_run = scenario_files.keys() if args.scenario == 'all' else [args.scenario]
    for scenario in scenarios_to_run:
        run_scenario(scenario, args.timehorizon, scenario_files)

if __name__ == "__main__":
    main()