# Remote Renewable Energy Hub (RREH) Master Thesis Repository

This repository contains the code and resources used for my master's thesis titled "Economic Viability of Carbon-Neutral Synthetic Fuel Production in Greenland’s Remote Renewable Energy Hub". The thesis explores the technical feasibility and economic viability of creating a remote renewable energy hub in Greenland, utilizing its untapped wind and hydro resources to produce synthetic fuels like methane, methanol, hydrogen, and ammonia.

This repository provides the necessary tools to reproduce the results presented in the thesis.

## Reproducing Results
### Solving the models

To reproduce the results, you need to run the 'main.py' script with the following arguments:

```bash
RREH_thesis
├───models
│   ├───algeria
│   ├───ammonia
│   ├───ammonia_algeria
│   ├───combined
│   ├───germany
│   ├───germany_pipe
│   ├───hydro
│   ├───hydrogen
│   ├───hydrogen_algeria
│   ├───hydro_wind
│   ├───methanol
│   ├───methanol_algeria
│   ├───spain
│   ├───spain_pipe
│   ├───wave
│   ├───wind_offshore
│   └───wind_onshore
```
#### Example Command

To run the `main.py` script for all scenarios with a time horizon of 3 years (26,280 hours):

```bash
python main.py -s all -t 17544
```
> **Note:** The results obtained in the paper were done with `-t 43800` (5 years), but it can be very long to run them all! The time horizon is in hours, so 1 year is 8760, 2 years is 17544, 3 years is 26280, and 5 years is 43800.

### Storing Results

After running the `main.py` script, the results JSON files will be added to a new folder named `results` inside each model's folder.

### Analyzing Results

Once the results are generated, you can run the `analysis.py` script to analyze the data.

#### Analyzing JSON Files

To analyze all the JSON files and create a CSV of the most important data linked to the optimization:


```bash
python analysis.py -m analyze_json -t <timehorizon>
``` 
 >**Note:** Replace `<timehorizon>` with the time horizon used in the `main.py` script.

#### Analyzing CSV Files

To analyze the generated CSV file:


```bash
python analysis.py -m analyze_csv -t <timehorizon>
``` 
 >**Note:** Replace `<timehorizon>` with the time horizon used in the `main.py` script.