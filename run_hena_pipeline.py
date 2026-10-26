"""
run_hena_pipeline.py
====================
Master Execution Pipeline for Part 1: Data and Simulation.
Lead Author: Hena Parween
Project: Deep Learning Based Defect Detection and Rule-Based Correction Framework for FDM 3D Printing

This script runs all steps assigned to Hena under the Team Work Plan:
1. Ingests, filters, and cleans the dataset for the 6 target FDM defects.
2. Produces a stratified 70/15/15 train/val/test split with fixed seed=42.
3. Materializes the dataset and exports `data_split_manifest.csv`.
4. Computes inverse-frequency class weights for training.
5. Generates publication-grade EDA figures in `figures/`.
6. Compiles the 18 Predefined Simulation Scenarios (RQ3 testbed).
7. Compiles the Independent Reference Benchmark from external AM literature.
8. Executes the Simulated Rule-Based Recommendation evaluation and calculates
   Recommendation Agreement Rate and Cohen's Kappa score.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ingest_roboflow import ingest
from src.data_loader import FDMDatasetPipeline
from src.eda_visualizer import generate_eda_figures
from src.simulation_scenarios import export_scenarios_to_markdown, PREDEFINED_SCENARIOS
from src.independent_reference import INDEPENDENT_REFERENCE_BENCHMARK
from src.agreement_evaluator import evaluate_recommendation_agreement
from src.class_audit import make_audit_sheet
from src.rq3_figures import generate_rq3_figures
from src.report_writer import write_data_report
from src.model_report import write_model_report
from src.check_inputs import check_audit, check_rule_table, check_predictions


def export_independent_reference_markdown(output_file: str = "docs/INDEPENDENT_REFERENCE_TABLE.md"):
    """Exports independent benchmark reference table as Markdown for the team and report."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Independently Prepared Reference Recommendations Benchmark (RQ3)\n\n")
        f.write("**Prepared by: Hena Parween (Part 1: Data and Simulation)**  \n")
        f.write("*Note: Prepared independently from external AM literature to guarantee unbiased evaluation of the rule table.*\n\n")
        f.write("## 1. Methodological Independence Notice\n\n")
        f.write("In accordance with the project proposal and team work plan, Research Question 3 evaluates whether ")
        f.write("the rule table can reliably match expert manufacturing recommendations. To avoid self-referential bias ")
        f.write("(where the author of the rule table also provides the answers), this reference benchmark was authored ")
        f.write("independently using established international additive manufacturing standards and published troubleshooting guides:\n")
        f.write("- **Prusa Research**: *3D Printing Handbook & Knowledge Base* (2023)\n")
        f.write("- **Simplify3D**: *Comprehensive Print Quality Troubleshooting Guide* (2022)\n")
        f.write("- **All3DP**: *Additive Manufacturing Diagnostics and Print Recovery* (2023)\n")
        f.write("- **Gibson, Rosen, Stucker**: *Additive Manufacturing Technologies* (Springer, 2021)\n")
        f.write("- **ISO/ASTM 52900:2021**: *Additive manufacturing — General principles*\n\n")
        f.write("## 2. Independent Reference Recommendation Table\n\n")
        f.write("| Defect Category | Root Physical Cause | Primary Corrective Action | Secondary Action | Hardware / Mechanical Check | Source References |\n")
        f.write("|---|---|---|---|---|---|\n")
        
        for k, ref in INDEPENDENT_REFERENCE_BENCHMARK.items():
            prim = f"**{ref['primary_action']['parameter']}** ({ref['primary_action']['direction']}): `{ref['primary_action']['recommended_delta']}`"
            sec = f"**{ref['secondary_action']['parameter']}** ({ref['secondary_action']['direction']}): `{ref['secondary_action']['recommended_delta']}`"
            refs = "<br>".join(ref["primary_citations"])
            f.write(f"| **{ref['defect_name']}** | {ref['physical_root_cause']} | {prim} | {sec} | {ref['hardware_check']} | {refs} |\n")
            
    print(f"[+] Independent reference table exported to: {output_file}")


def _stop_if_problems(problems):
    """Stops the run so malformed teammate files never reach the RQ3 results."""
    if not problems:
        print("[+] Received files look valid (or have not arrived yet).")
        return
    print(f"[!] {len(problems)} problem(s) in files received from teammates. Send these back and re-run:")
    for p in problems:
        print(f"    - {p}")
    sys.exit(1)


def main():
    print("=" * 70)
    print(" FDM DEFECT DETECTION: PART 1 (HENA - DATA & SIMULATION PIPELINE)")
    print("=" * 70)
    
    # 0. Validate files received from Agya before they change the data
    print("\n>>> STEP 0: Checking files received from teammates...")
    _stop_if_problems(check_audit() + check_rule_table())

    # 1. Run Data Ingestion, Cleaning & Splitting
    print("\n>>> STEP 1: Running Data Ingestion, Cleaning & Stratified Splitting...")
    ingest()
    pipeline = FDMDatasetPipeline()
    cleaned = pipeline.scan_and_clean_raw()
    split_records = pipeline.create_stratified_split(cleaned)
    pipeline.materialize_dataset(split_records)
    
    # 2. Generate EDA Visualizations
    print("\n>>> STEP 2: Generating Exploratory Data Visualizations...")
    generate_eda_figures()
    
    # 3. Export Predefined Scenarios
    print("\n>>> STEP 3: Compiling 18 Predefined Defect Scenarios (RQ3)...")
    export_scenarios_to_markdown()
    
    # 4. Export Independent Reference Benchmark
    print("\n>>> STEP 4: Compiling Independent Reference Benchmark...")
    export_independent_reference_markdown()
    
    # 5. Evaluate Recommendation Agreement Rate
    print("\n>>> STEP 5: Running Recommendation Agreement Rate Evaluation...")
    _stop_if_problems(check_predictions())
    evaluate_recommendation_agreement()
    generate_rq3_figures()

    print("\n>>> STEP 6: Preparing the class audit sheet for Agya...")
    make_audit_sheet()

    print("\n>>> STEP 7: Writing the Data and Experimental Setup report sections...")
    write_data_report()
    write_model_report()
    
    print("\n" + "=" * 70)
    print(" PART 1 PIPELINE FINISHED. Check the RQ3 status above: it needs the")
    print(" rule table (data/rules/rule_table.json) and model predictions (reports/predictions_<model>.csv).")
    print("=" * 70)


if __name__ == "__main__":
    main()
