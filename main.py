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
    'hydro_wind': 'greenland_hydro_wind.gboml',
    'combined': 'greenland_combined.gboml',
    'spain': 'spain.gboml',
    'algeria': 'algeria.gboml',
    'germany': 'germany.gboml'
}


def run_scenario(scenario, timehorizon, wacc_case):
    """
    Runs the GBOML model for a given scenario based on its .gboml file with specified WACC values and saves the results.
    Handles specific country parameter adjustments based on the wacc_case.
    """
    print(f"Running scenario: {scenario} with WACC case: {wacc_case}")
    gboml_file_path = SCENARIO_FILES[scenario]
    gboml_model_full_path = os.path.join("models", scenario, gboml_file_path)
    gboml_model = GbomlGraph(timehorizon=timehorizon)
    nodes, edges, param = gboml_model.import_all_nodes_and_edges(gboml_model_full_path)
    gboml_model.add_global_parameters(param)

    if wacc_case == 'constant':
        gboml_model.add_global_parameter('wacc', 0.07)
        gboml_model.add_global_parameter('wacc_be', 0.07)
        gboml_model.add_global_parameter('onshore_wacc', 0.07)
        gboml_model.add_global_parameter('offshore_wacc', 0.07)
        gboml_model.add_global_parameter('hydro_wacc', 0.07)
        gboml_model.add_global_parameter('solar_wacc', 0.07)
    else:  # 'diff' case
        if scenario == 'spain':
            gboml_model.add_global_parameter('wacc', 0.0587)
            gboml_model.add_global_parameter('wacc_be', 0.0441)
            gboml_model.add_global_parameter('onshore_wacc', 0.031)
            gboml_model.add_global_parameter('solar_wacc', 0.036)
        elif scenario == 'germany':
            gboml_model.add_global_parameter('wacc', 0.0353)
            gboml_model.add_global_parameter('wacc_be', 0.0441)
            gboml_model.add_global_parameter('onshore_wacc', 0.013)
            gboml_model.add_global_parameter('solar_wacc', 0.013)
        elif scenario == 'algeria':
            gboml_model.add_global_parameter('wacc', 0.1011)
            gboml_model.add_global_parameter('wacc_be', 0.0441)
            gboml_model.add_global_parameter('onshore_wacc', 0.114)
            gboml_model.add_global_parameter('solar_wacc', 0.11)
        else:
            gboml_model.add_global_parameter('wacc', 0.0353)
            gboml_model.add_global_parameter('wacc_be', 0.0441)
            gboml_model.add_global_parameter('onshore_wacc', 0.015)
            gboml_model.add_global_parameter('offshore_wacc', 0.015)
            gboml_model.add_global_parameter('hydro_wacc', 0.0421)

    gboml_model.add_nodes_in_model(*nodes)
    gboml_model.add_hyperedges_in_model(*edges)
    gboml_model.build_model()
    solution, obj, status, solver_info, constr_info, _ = gboml_model.solve_gurobi(opt_file="scripts/analysis/gurobi.txt")

    gathered_data = gboml_model.turn_solution_to_dictionary(solver_info, status, solution, obj, constr_info)
    results_dir_path = os.path.join("models", scenario, "results")
    os.makedirs(results_dir_path, exist_ok=True)

    result_filename = f"{scenario}_{wacc_case}_{timehorizon}_results.json"
    result_path = os.path.join(results_dir_path, result_filename)
    with open(result_path, "w") as fp:
        json.dump(gathered_data, fp, indent=4)

    print(f"Results saved to {result_path}")

def main():
    parser = argparse.ArgumentParser(description="Run GBOML models for various energy scenarios.")
    parser.add_argument('-s', '--scenario', help="Specific energy scenario to run, or 'all' to run all scenarios.", 
                        default='all')
    parser.add_argument('-t', '--timehorizon', help="Time horizon for the model", type=int, default=17544)
    args = parser.parse_args()

    # All scenarios are defined here
    scenarios_to_run = SCENARIO_FILES.keys() if args.scenario == 'all' else [args.scenario]
    
    for scenario in scenarios_to_run:
        run_scenario(scenario, args.timehorizon, 'constant')
        run_scenario(scenario, args.timehorizon, 'diff')

if __name__ == "__main__":
    main()
