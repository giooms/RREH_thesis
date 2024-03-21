import argparse
import os
import sys
from datetime import datetime
import json

from gboml import GbomlGraph
import gboml.compiler.classes as gcc

# Define the specific scenarios and their respective .gboml files
SCENARIO_FILES = {
    'wind_onshore': 'greenland_wind_on.gboml',
    'wind_offshore': 'greenland_wind_off.gboml',
    'wave': 'greenland_wave.gboml',
    'hydro': 'greenland_hydro.gboml',
    'combined': 'greenland_combined.gboml'
}

def run_scenario(scenario, timehorizon):
    """
    Runs the GBOML model for a given scenario based on its .gboml file and saves the results.
    """
    print(f"Running scenario: {scenario}")
    gboml_file_path = SCENARIO_FILES[scenario]
    # Assuming the script is run from the 'scripts' directory
    gboml_model_full_path = f"models/{scenario}/{gboml_file_path}"

    gboml_model = GbomlGraph(timehorizon=timehorizon)
    nodes, edges, param = gboml_model.import_all_nodes_and_edges(gboml_model_full_path)
    gboml_model.add_global_parameters(param)
    gboml_model.add_nodes_in_model(*nodes)
    gboml_model.add_hyperedges_in_model(*edges)
    gboml_model.build_model()
    solution, obj, status, solver_info, constr_info, _ = gboml_model.solve_gurobi(opt_file="scripts/analysis/gurobi.txt")
    
    gathered_data = gboml_model.turn_solution_to_dictionary(solver_info, status, solution, obj, constr_info)

    # Construct the path to the scenario-specific results directory
    results_dir_path = f"models/{scenario}/results"

    # Check if the scenario-specific results directory exists; if not, create it
    if not os.path.exists(results_dir_path):
        os.makedirs(results_dir_path)

    # Define the path for the results JSON file within the scenario-specific results directory
    result_path = f"{results_dir_path}/{scenario}_{timehorizon}_results.json"

    # Save the gathered data to the JSON file
    with open(result_path, "w") as fp:
        json.dump(gathered_data, fp, indent=4)

    print(f"Results saved to {result_path}")


def main():
    parser = argparse.ArgumentParser(description="Run GBOML models for various energy scenarios.")
    parser.add_argument('-s', '--scenario', help="Specific energy scenario to run, or 'all' to run all scenarios.", 
                        default='all', choices=list(SCENARIO_FILES.keys()) + ['all'])
    parser.add_argument('-t', '--timehorizon', help="Time horizon for the model", 
                        type=int, default=17544, choices=[8760, 17544, 26304, 35064, 43824])  # Default to one year (365 days * 24 hours)
    args = parser.parse_args()

    if args.scenario == 'all':
        # Iterate over and run all scenarios sequentially
        for scenario in SCENARIO_FILES:
            run_scenario(scenario, args.timehorizon)
    else:
        # Run the specified scenario
        run_scenario(args.scenario, args.timehorizon)

if __name__ == "__main__":
    main()