"""
independent_reference.py
========================
Independently Prepared Reference Recommendations Benchmark for RQ3.
Authored by Hena Parween (Part 1: Data and Simulation).

CRITICAL METHODOLOGICAL GUARANTEE:
Per the team work plan, the Manufacturing Teammate develops the model's Rule Table
based on Jin et al. (2019), Brion et al. (2022), and De La Rosa et al. (2024).
To prevent circular evaluation and guarantee experimental independence for RQ3,
this reference benchmark was authored INDEPENDENTLY by Hena using an external set of
standard manufacturing literature and established additive troubleshooting handbooks:
- Prusa Research 3D Printing Knowledge Base & Handbook (2023)
- Simplify3D Comprehensive Print Quality Troubleshooting Guide (2022)
- Gibson, Rosen, & Stucker, Additive Manufacturing Technologies (Springer, 2021)
- All3DP Additive Diagnostic Guide (2023)
- ISO/ASTM 52900:2021 General AM Principles
"""

from typing import Dict, Any, List


INDEPENDENT_REFERENCE_BENCHMARK: Dict[str, Dict[str, Any]] = {
    "stringing": {
        "defect_name": "Stringing / Oozing",
        "physical_root_cause": (
            "Residual nozzle pressure and molten polymer fluidity during non-print travel moves, "
            "allowing low-viscosity filament to weep out of the nozzle orifice."
        ),
        "primary_action": {
            "parameter": "retraction_distance",
            "direction": "INCREASE",
            "recommended_delta": "+1.0 to +2.0 mm (Bowden) or +0.5 to +1.0 mm (Direct Drive)",
            "normalized_adjustment": {"retraction_distance_delta_mm": +1.0}
        },
        "secondary_action": {
            "parameter": "nozzle_temperature",
            "direction": "DECREASE",
            "recommended_delta": "-5 to -10 °C",
            "normalized_adjustment": {"nozzle_temperature_delta_c": -5}
        },
        "tertiary_action": {
            "parameter": "retraction_speed",
            "direction": "INCREASE",
            "recommended_delta": "+10 to +20 mm/s",
            "normalized_adjustment": {"retraction_speed_delta_mms": +15}
        },
        "hardware_check": "Inspect nozzle tip for partial erosion; ensure filament is dried (hygroscopic moisture boils and forces oozing).",
        "primary_citations": [
            "Simplify3D Print Quality Guide: Stringing and Oozing (2022)",
            "Prusa Knowledge Base: Stringing and Oozing Solutions (2023)"
        ]
    },
    
    "warping": {
        "defect_name": "Warping / Thermal Shrinkage",
        "physical_root_cause": (
            "Differential thermal contraction and internal cooling stress gradients between "
            "newly deposited hot upper layers and cooling lower layers in contact with the print bed."
        ),
        "primary_action": {
            "parameter": "bed_temperature",
            "direction": "INCREASE",
            "recommended_delta": "+5 to +10 °C (below polymer glass transition Tg)",
            "normalized_adjustment": {"bed_temperature_delta_c": +5}
        },
        "secondary_action": {
            "parameter": "cooling_fan_speed",
            "direction": "DECREASE",
            "recommended_delta": "-20% to -50% (turn off completely for first layers)",
            "normalized_adjustment": {"fan_speed_factor": -0.30}
        },
        "tertiary_action": {
            "parameter": "print_speed_first_layer",
            "direction": "DECREASE",
            "recommended_delta": "-10 to -20 mm/s for initial layer adhesion",
            "normalized_adjustment": {"first_layer_speed_delta_mms": -10}
        },
        "hardware_check": "Clean bed surface with Isopropyl Alcohol (IPA); apply adhesive brim/raft; eliminate ambient room drafts/enclosure draft.",
        "primary_citations": [
            "Gibson, Rosen, & Stucker, Additive Manufacturing Technologies (Springer, 2021)",
            "Prusa Research: Warping Prevention in Extrusion Printing (2023)"
        ]
    },
    
    "layer_shifting": {
        "defect_name": "Layer Shifting / Step Misalignment",
        "physical_root_cause": (
            "Stepper motor torque failure, belt tooth skipping, mechanical binding along X/Y axes, "
            "or nozzle collision with curled overhangs causing skipped stepper micro-steps."
        ),
        "primary_action": {
            "parameter": "travel_and_print_speed",
            "direction": "DECREASE",
            "recommended_delta": "-15% to -25% (reduce acceleration and jerk)",
            "normalized_adjustment": {"feedrate_percentage_delta": -20}
        },
        "secondary_action": {
            "parameter": "stepper_motor_current_or_jerk",
            "direction": "DECREASE_JERK",
            "recommended_delta": "Lower maximum jerk from 10 to 7 mm/s",
            "normalized_adjustment": {"jerk_delta": -3}
        },
        "tertiary_action": {
            "parameter": "z_hop_retraction",
            "direction": "ENABLE",
            "recommended_delta": "Enable 0.2 mm Z-hop during travel to prevent nozzle strike",
            "normalized_adjustment": {"z_hop_mm": 0.2}
        },
        "hardware_check": "Tighten loose timing belts (X/Y axis); check set screws on pulleys; lubricate linear rods; inspect stepper driver overheating.",
        "primary_citations": [
            "All3DP: 3D Printer Layer Shift – How to Fix It (2023)",
            "Simplify3D Troubleshooting: Layer Shifting and Misalignment (2022)"
        ]
    },
    
    "under_extrusion": {
        "defect_name": "Under-Extrusion",
        "physical_root_cause": (
            "Volumetric material delivery rate is less than the commanded toolpath volume, "
            "resulting in inter-bead voids, weak perimeters, and porous infill."
        ),
        "primary_action": {
            "parameter": "flow_rate_extrusion_multiplier",
            "direction": "INCREASE",
            "recommended_delta": "+5% to +10%",
            "normalized_adjustment": {"flow_rate_percentage_delta": +7}
        },
        "secondary_action": {
            "parameter": "nozzle_temperature",
            "direction": "INCREASE",
            "recommended_delta": "+5 to +10 °C (lowers melt viscosity and backpressure)",
            "normalized_adjustment": {"nozzle_temperature_delta_c": +5}
        },
        "tertiary_action": {
            "parameter": "print_speed",
            "direction": "DECREASE",
            "recommended_delta": "-10% to -20% (relieves volumetric flow limit)",
            "normalized_adjustment": {"feedrate_percentage_delta": -15}
        },
        "hardware_check": "Check for partial nozzle clog (perform cold pull); verify extruder drive gear tension; ensure filament diameter matches slicer setting.",
        "primary_citations": [
            "Simplify3D Print Quality Guide: Under-Extrusion (2022)",
            "Prusa 3D Printing Handbook: Extrusion and Flow Troubleshooting (2023)"
        ]
    },
    
    "over_extrusion": {
        "defect_name": "Over-Extrusion",
        "physical_root_cause": (
            "Excessive volumetric deposition exceeding the sliced bead envelope, "
            "causing plastic bulging, dimensional inaccuracy, nozzle dragging, and surface blobbing."
        ),
        "primary_action": {
            "parameter": "flow_rate_extrusion_multiplier",
            "direction": "DECREASE",
            "recommended_delta": "-5% to -10%",
            "normalized_adjustment": {"flow_rate_percentage_delta": -7}
        },
        "secondary_action": {
            "parameter": "nozzle_temperature",
            "direction": "DECREASE",
            "recommended_delta": "-5 °C (curbs excessive thermal expansion and runniness)",
            "normalized_adjustment": {"nozzle_temperature_delta_c": -5}
        },
        "tertiary_action": {
            "parameter": "cooling_fan_speed",
            "direction": "INCREASE",
            "recommended_delta": "+20% to +30% (solidifies deposited bead rapidly)",
            "normalized_adjustment": {"fan_speed_factor": +0.25}
        },
        "hardware_check": "Calibrate extruder E-steps/mm; measure actual filament diameter with digital calipers (e.g. 1.78mm vs 1.75mm).",
        "primary_citations": [
            "Prusa Knowledge Base: Over-Extrusion Calibration (2023)",
            "All3DP: Over-Extrusion in 3D Printing – Causes and Solutions (2023)"
        ]
    },
    
    "bed_adhesion": {
        "defect_name": "Bed Adhesion Failure",
        "physical_root_cause": (
            "Insufficient intermolecular bonding between the first thermoplastic layer and the build plate, "
            "caused by improper nozzle-to-bed gap (Z-offset), contamination, or non-uniform leveling."
        ),
        "primary_action": {
            "parameter": "z_offset_nozzle_gap",
            "direction": "DECREASE",
            "recommended_delta": "-0.05 to -0.10 mm (lower nozzle closer to bed for proper squish)",
            "normalized_adjustment": {"z_offset_delta_mm": -0.05}
        },
        "secondary_action": {
            "parameter": "bed_temperature",
            "direction": "INCREASE",
            "recommended_delta": "+5 to +10 °C",
            "normalized_adjustment": {"bed_temperature_delta_c": +5}
        },
        "tertiary_action": {
            "parameter": "first_layer_extrusion_width_and_speed",
            "direction": "ADJUST_FIRST_LAYER",
            "recommended_delta": "Increase first layer flow (+5%), reduce first layer speed (-30%)",
            "normalized_adjustment": {"first_layer_speed_delta_mms": -15, "first_layer_flow_delta": +5}
        },
        "hardware_check": "Re-tram/level build plate corners; degrease PEI/glass sheet with 90%+ IPA or warm dish soap; apply adhesion promoter (glue stick/PVP).",
        "primary_citations": [
            "Simplify3D Quality Guide: Not Sticking to the Bed (2022)",
            "Prusa 3D Printing Handbook: First Layer Calibration (2023)"
        ]
    }
}


def get_reference_for_defect(defect_name: str) -> Dict[str, Any]:
    """Retrieves the independently prepared reference recommendation for a given defect."""
    key = defect_name.lower().replace("-", "_").replace(" ", "_")
    if key not in INDEPENDENT_REFERENCE_BENCHMARK:
        raise ValueError(f"Defect '{defect_name}' not found in independent reference benchmark.")
    return INDEPENDENT_REFERENCE_BENCHMARK[key]
