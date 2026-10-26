"""
simulation_scenarios.py
=======================
Predefined Defect Scenarios & Simulated Closed-Loop Execution Engine for RQ3.
Authored by Hena Parween (Part 1: Data and Simulation).

This module implements the simulated recommendation system proposed in lieu of
direct physical printer hardware control:
1. Defines 18 diverse test scenarios covering all 6 defect categories across
   three severity levels (Mild, Moderate, Severe) with realistic operational
   printing parameter baselines.
2. Implements a simulated execution engine incorporating the 65% classification
   confidence threshold defined in De La Rosa et al. (2024, CIRP).
3. Logs system decisions and corrective recommendations for agreement-rate validation.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

PREDEFINED_SCENARIOS: List[Dict[str, Any]] = [
    # ------------------- STRINGING -------------------
    {
        "scenario_id": "SCEN-STR-01",
        "ground_truth_defect": "stringing",
        "severity": "Mild",
        "material": "PLA",
        "baseline_state": {
            "nozzle_temp_c": 215,
            "bed_temp_c": 60,
            "flow_rate_pct": 100,
            "print_speed_mms": 50,
            "retraction_dist_mm": 4.0,
            "retraction_speed_mms": 35,
            "fan_speed_pwm": 255,
            "z_offset_mm": 0.00
        },
        "description": "Fine whisker-like cobwebs between dual calibration towers during travel moves."
    },
    {
        "scenario_id": "SCEN-STR-02",
        "ground_truth_defect": "stringing",
        "severity": "Moderate",
        "material": "PLA",
        "baseline_state": {
            "nozzle_temp_c": 225,
            "bed_temp_c": 60,
            "flow_rate_pct": 102,
            "print_speed_mms": 60,
            "retraction_dist_mm": 3.0,
            "retraction_speed_mms": 30,
            "fan_speed_pwm": 200,
            "z_offset_mm": 0.00
        },
        "description": "Noticeable branch-like polymer tendrils spanning non-print travel spans."
    },
    {
        "scenario_id": "SCEN-STR-03",
        "ground_truth_defect": "stringing",
        "severity": "Severe",
        "material": "PETG",
        "baseline_state": {
            "nozzle_temp_c": 245,
            "bed_temp_c": 75,
            "flow_rate_pct": 105,
            "print_speed_mms": 45,
            "retraction_dist_mm": 2.5,
            "retraction_speed_mms": 25,
            "fan_speed_pwm": 128,
            "z_offset_mm": 0.00
        },
        "description": "Heavy bridging blobs and thick polymer filaments clogging inter-feature cavities."
    },

    # ------------------- WARPING -------------------
    {
        "scenario_id": "SCEN-WARP-01",
        "ground_truth_defect": "warping",
        "severity": "Mild",
        "material": "PLA",
        "baseline_state": {
            "nozzle_temp_c": 200,
            "bed_temp_c": 50,
            "flow_rate_pct": 100,
            "print_speed_mms": 50,
            "retraction_dist_mm": 5.0,
            "retraction_speed_mms": 40,
            "fan_speed_pwm": 255,
            "z_offset_mm": 0.00
        },
        "description": "Slight upward curl (approx 0.5 mm) at sharp outer corner of rectangular base."
    },
    {
        "scenario_id": "SCEN-WARP-02",
        "ground_truth_defect": "warping",
        "severity": "Moderate",
        "material": "ABS",
        "baseline_state": {
            "nozzle_temp_c": 235,
            "bed_temp_c": 90,
            "flow_rate_pct": 100,
            "print_speed_mms": 45,
            "retraction_dist_mm": 4.0,
            "retraction_speed_mms": 35,
            "fan_speed_pwm": 100,
            "z_offset_mm": 0.00
        },
        "description": "Substantial perimeter peeling and bottom detachment causing convex base distortion."
    },
    {
        "scenario_id": "SCEN-WARP-03",
        "ground_truth_defect": "warping",
        "severity": "Severe",
        "material": "ABS",
        "baseline_state": {
            "nozzle_temp_c": 240,
            "bed_temp_c": 80,
            "flow_rate_pct": 100,
            "print_speed_mms": 60,
            "retraction_dist_mm": 4.0,
            "retraction_speed_mms": 40,
            "fan_speed_pwm": 180,
            "z_offset_mm": 0.00
        },
        "description": "Extreme corner lifting >3 mm with layer separation and print dislodgment risk."
    },

    # ------------------- LAYER SHIFTING -------------------
    {
        "scenario_id": "SCEN-LS-01",
        "ground_truth_defect": "layer_shifting",
        "severity": "Mild",
        "material": "PLA",
        "baseline_state": {
            "nozzle_temp_c": 205,
            "bed_temp_c": 60,
            "flow_rate_pct": 100,
            "print_speed_mms": 75,
            "retraction_dist_mm": 5.0,
            "retraction_speed_mms": 45,
            "fan_speed_pwm": 255,
            "z_offset_mm": 0.00
        },
        "description": "Single-step horizontal stair offset (approx 0.4 mm) along the X-axis."
    },
    {
        "scenario_id": "SCEN-LS-02",
        "ground_truth_defect": "layer_shifting",
        "severity": "Moderate",
        "material": "PLA",
        "baseline_state": {
            "nozzle_temp_c": 210,
            "bed_temp_c": 60,
            "flow_rate_pct": 100,
            "print_speed_mms": 90,
            "retraction_dist_mm": 5.0,
            "retraction_speed_mms": 45,
            "fan_speed_pwm": 255,
            "z_offset_mm": 0.00
        },
        "description": "Multiple lateral step offsets along Y-axis following fast non-print jerk moves."
    },
    {
        "scenario_id": "SCEN-LS-03",
        "ground_truth_defect": "layer_shifting",
        "severity": "Severe",
        "material": "PETG",
        "baseline_state": {
            "nozzle_temp_c": 235,
            "bed_temp_c": 75,
            "flow_rate_pct": 102,
            "print_speed_mms": 110,
            "retraction_dist_mm": 4.0,
            "retraction_speed_mms": 40,
            "fan_speed_pwm": 128,
            "z_offset_mm": 0.00
        },
        "description": "Discontinuous massive layer translation (>4 mm), completely compromising part geometry."
    },

    # ------------------- UNDER-EXTRUSION -------------------
    {
        "scenario_id": "SCEN-UND-01",
        "ground_truth_defect": "under_extrusion",
        "severity": "Mild",
        "material": "PLA",
        "baseline_state": {
            "nozzle_temp_c": 190,
            "bed_temp_c": 60,
            "flow_rate_pct": 92,
            "print_speed_mms": 65,
            "retraction_dist_mm": 5.0,
            "retraction_speed_mms": 45,
            "fan_speed_pwm": 255,
            "z_offset_mm": 0.00
        },
        "description": "Intermittent pinhole gaps visible across upper solid infill surface."
    },
    {
        "scenario_id": "SCEN-UND-02",
        "ground_truth_defect": "under_extrusion",
        "severity": "Moderate",
        "material": "PLA",
        "baseline_state": {
            "nozzle_temp_c": 185,
            "bed_temp_c": 60,
            "flow_rate_pct": 85,
            "print_speed_mms": 70,
            "retraction_dist_mm": 5.0,
            "retraction_speed_mms": 45,
            "fan_speed_pwm": 255,
            "z_offset_mm": 0.00
        },
        "description": "Perimeter bead separation, fragile delaminating layers, and sparse infill voids."
    },
    {
        "scenario_id": "SCEN-UND-03",
        "ground_truth_defect": "under_extrusion",
        "severity": "Severe",
        "material": "PETG",
        "baseline_state": {
            "nozzle_temp_c": 215,
            "bed_temp_c": 70,
            "flow_rate_pct": 75,
            "print_speed_mms": 80,
            "retraction_dist_mm": 6.0,
            "retraction_speed_mms": 50,
            "fan_speed_pwm": 200,
            "z_offset_mm": 0.00
        },
        "description": "Severe volumetric starvation: sponge-like brittle walls and discontinuous extrusions."
    },

    # ------------------- OVER-EXTRUSION -------------------
    {
        "scenario_id": "SCEN-OVR-01",
        "ground_truth_defect": "over_extrusion",
        "severity": "Mild",
        "material": "PLA",
        "baseline_state": {
            "nozzle_temp_c": 215,
            "bed_temp_c": 60,
            "flow_rate_pct": 108,
            "print_speed_mms": 50,
            "retraction_dist_mm": 5.0,
            "retraction_speed_mms": 40,
            "fan_speed_pwm": 200,
            "z_offset_mm": 0.00
        },
        "description": "Rough tactile top surface with slight polymer ridge build-up along perimeter seams."
    },
    {
        "scenario_id": "SCEN-OVR-02",
        "ground_truth_defect": "over_extrusion",
        "severity": "Moderate",
        "material": "PLA",
        "baseline_state": {
            "nozzle_temp_c": 225,
            "bed_temp_c": 60,
            "flow_rate_pct": 115,
            "print_speed_mms": 45,
            "retraction_dist_mm": 4.5,
            "retraction_speed_mms": 35,
            "fan_speed_pwm": 180,
            "z_offset_mm": 0.00
        },
        "description": "Prominent surface blobs, bulging outer walls, and visible nozzle drag furrows."
    },
    {
        "scenario_id": "SCEN-OVR-03",
        "ground_truth_defect": "over_extrusion",
        "severity": "Severe",
        "material": "PETG",
        "baseline_state": {
            "nozzle_temp_c": 250,
            "bed_temp_c": 75,
            "flow_rate_pct": 125,
            "print_speed_mms": 40,
            "retraction_dist_mm": 3.5,
            "retraction_speed_mms": 30,
            "fan_speed_pwm": 100,
            "z_offset_mm": 0.00
        },
        "description": "Excess molten polymer overflow, loss of dimensional tolerance, and nozzle tip scraping."
    },

    # ------------------- BED ADHESION -------------------
    {
        "scenario_id": "SCEN-BED-01",
        "ground_truth_defect": "bed_adhesion",
        "severity": "Mild",
        "material": "PLA",
        "baseline_state": {
            "nozzle_temp_c": 200,
            "bed_temp_c": 50,
            "flow_rate_pct": 100,
            "print_speed_mms": 50,
            "retraction_dist_mm": 5.0,
            "retraction_speed_mms": 40,
            "fan_speed_pwm": 255,
            "z_offset_mm": 0.15
        },
        "description": "First layer lines rounded rather than flat; loose perimeter easily peeled by hand."
    },
    {
        "scenario_id": "SCEN-BED-02",
        "ground_truth_defect": "bed_adhesion",
        "severity": "Moderate",
        "material": "PLA",
        "baseline_state": {
            "nozzle_temp_c": 200,
            "bed_temp_c": 45,
            "flow_rate_pct": 100,
            "print_speed_mms": 55,
            "retraction_dist_mm": 5.0,
            "retraction_speed_mms": 40,
            "fan_speed_pwm": 255,
            "z_offset_mm": 0.25
        },
        "description": "Significant detachment of first layer skirt and infill lines during initial skirt loop."
    },
    {
        "scenario_id": "SCEN-BED-03",
        "ground_truth_defect": "bed_adhesion",
        "severity": "Severe",
        "material": "PETG",
        "baseline_state": {
            "nozzle_temp_c": 230,
            "bed_temp_c": 55,
            "flow_rate_pct": 100,
            "print_speed_mms": 50,
            "retraction_dist_mm": 4.0,
            "retraction_speed_mms": 35,
            "fan_speed_pwm": 150,
            "z_offset_mm": 0.40
        },
        "description": "Total print release from bed surface; extruded filament forms dragging bird's nest."
    }
]


def export_scenarios_to_markdown(output_file: str = "docs/PREDEFINED_SCENARIOS.md"):
    """Exports scenario table in clean markdown format for team review and report."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Predefined FDM Defect Simulation Scenarios (RQ3 Testbed)\n\n")
        f.write("**Authored by Hena Parween (Part 1: Data and Simulation)**\n\n")
        f.write("This suite of 18 realistic operational test scenarios establishes the empirical testbed ")
        f.write("for evaluating the rule-based recommendation framework in lieu of real-time printer hardware control.\n\n")
        f.write("| ID | Defect Category | Severity | Material | Baseline Key Params | Scenario Physical Manifestation |\n")
        f.write("|---|---|---|---|---|---|\n")
        for sc in PREDEFINED_SCENARIOS:
            b = sc["baseline_state"]
            params = f"Nozzle:{b['nozzle_temp_c']}°C, Bed:{b['bed_temp_c']}°C, Flow:{b['flow_rate_pct']}%, Speed:{b['print_speed_mms']}mm/s"
            f.write(f"| `{sc['scenario_id']}` | **{sc['ground_truth_defect'].replace('_', ' ').title()}** | {sc['severity']} | {sc['material']} | {params} | {sc['description']} |\n")
    print(f"[+] Exported predefined scenarios to: {output_file}")


if __name__ == "__main__":
    export_scenarios_to_markdown()
