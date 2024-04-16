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
    'comparison': 'comparison.gboml'
}

COUNTRY_PARAMS = {
    'spain': {
        'SOLAR_PV_PLANTS': [('solar_wacc', 0.036),('capacity_factor_PV', 'import "../../data/pv_capacity_factors_ES.csv"')],
        'WIND_PLANTS': [('wind_wacc', 0.031),('capacity_factor_wind', 'import "../../data/wind_capacity_factors_ES.csv"')],
        'LIQUEFIED_METHANE_CARRIERS': [('travel_time', 39.22)]
    },
    'germany': {
        'SOLAR_PV_PLANTS': [('solar_wacc', 0.013),('capacity_factor_PV', 'import "../../data/pv_capacity_factors_DE.csv"')],
        'WIND_PLANTS': [('wind_wacc', 0.013),('capacity_factor_wind', 'import "../../data/wind_capacity_factors_DE.csv"')],
        'LIQUEFIED_METHANE_CARRIERS': [('travel_time', 16.63)]
    },
    'algeria': {
        'SOLAR_PV_PLANTS': [('solar_wacc', 0.11), ('capacity_factor_PV', 'import "../../data/pv_capacity_factors_DZ.csv"')],
        'WIND_PLANTS': [('wind_wacc', 0.114),('capacity_factor_wind', 'import "../../data/wind_capacity_factors_DZ.csv"')],
        'LIQUEFIED_METHANE_CARRIERS': [('travel_time', 116)]
    }
}


def run_scenario(scenario, timehorizon, wacc_value, wacc_label):
    """
    Runs the GBOML model for a given scenario based on its .gboml file with specified WACC and saves the results.
    Handles specific country parameter adjustments for the 'comparison' scenario.
    """
    print(f"Running scenario: {scenario} with {wacc_label} WACC: {wacc_value}")
    gboml_file_path = SCENARIO_FILES[scenario]
    gboml_model_full_path = os.path.join("models", scenario, gboml_file_path)

    # Determine the countries to iterate over based on the scenario
    countries = ['spain', 'germany', 'algeria'] if scenario == 'comparison' else [None]

    for country in countries:
        gboml_model = GbomlGraph(timehorizon=timehorizon)
        nodes, edges, param = gboml_model.import_all_nodes_and_edges(gboml_model_full_path)
        gboml_model.add_global_parameters(param)
        gboml_model.add_global_parameter('wacc', wacc_value)  # Adjust WACC here

        if country:
            # Redefine parameters specifically for each country in the comparison scenario
            country_specific_params = COUNTRY_PARAMS.get(country, [])
            for component, changes in country_specific_params.items():
                parameters, values = zip(*changes)  # Unpack parameters and values
                print(f"Modified parameters for {country}: {component}, {list(parameters)}, {list(values)}")
                # gboml_model.redefine_parameters_from_list(component, list(parameters), list(values))
                gboml_model.redefine_parameters_from_list(
                    component,
                    ['solar_wacc','full_capex'],
                    [0.036, 500.0]
                )
            print(f"Modified parameters for {country}")

        gboml_model.add_nodes_in_model(*nodes)
        gboml_model.add_hyperedges_in_model(*edges)
        gboml_model.build_model()
        solution, obj, status, solver_info, constr_info, _ = gboml_model.solve_gurobi(opt_file="scripts/analysis/gurobi.txt")

        gathered_data = gboml_model.turn_solution_to_dictionary(solver_info, status, solution, obj, constr_info)
        results_dir_path = os.path.join("models", scenario, "results")
        os.makedirs(results_dir_path, exist_ok=True)

        # Construct the result filename based on whether the run is country-specific or not
        result_filename = f"{scenario}_{country}_{wacc_label}_{timehorizon}_results.json" if country else f"{scenario}_{wacc_label}_{timehorizon}_results.json"
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

    wacc_scenarios = {
        'High': 0.10,
        'Mid': 0.07,
        'Low': 0.04
    }

    # All scenarios are defined here
    scenarios_to_run = SCENARIO_FILES.keys() if args.scenario == 'all' else [args.scenario]
    
    for scenario in scenarios_to_run:
        for label, wacc in wacc_scenarios.items():
            run_scenario(scenario, args.timehorizon, wacc, label)

if __name__ == "__main__":
    main()
