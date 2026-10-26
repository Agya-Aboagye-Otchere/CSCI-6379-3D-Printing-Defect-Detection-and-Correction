# Predefined FDM Defect Simulation Scenarios (RQ3 Testbed)

**Authored by Hena Parween (Part 1: Data and Simulation)**

This suite of 18 realistic operational test scenarios establishes the empirical testbed for evaluating the rule-based recommendation framework in lieu of real-time printer hardware control.

| ID | Defect Category | Severity | Material | Baseline Key Params | Scenario Physical Manifestation |
|---|---|---|---|---|---|
| `SCEN-STR-01` | **Stringing** | Mild | PLA | Nozzle:215°C, Bed:60°C, Flow:100%, Speed:50mm/s | Fine whisker-like cobwebs between dual calibration towers during travel moves. |
| `SCEN-STR-02` | **Stringing** | Moderate | PLA | Nozzle:225°C, Bed:60°C, Flow:102%, Speed:60mm/s | Noticeable branch-like polymer tendrils spanning non-print travel spans. |
| `SCEN-STR-03` | **Stringing** | Severe | PETG | Nozzle:245°C, Bed:75°C, Flow:105%, Speed:45mm/s | Heavy bridging blobs and thick polymer filaments clogging inter-feature cavities. |
| `SCEN-WARP-01` | **Warping** | Mild | PLA | Nozzle:200°C, Bed:50°C, Flow:100%, Speed:50mm/s | Slight upward curl (approx 0.5 mm) at sharp outer corner of rectangular base. |
| `SCEN-WARP-02` | **Warping** | Moderate | ABS | Nozzle:235°C, Bed:90°C, Flow:100%, Speed:45mm/s | Substantial perimeter peeling and bottom detachment causing convex base distortion. |
| `SCEN-WARP-03` | **Warping** | Severe | ABS | Nozzle:240°C, Bed:80°C, Flow:100%, Speed:60mm/s | Extreme corner lifting >3 mm with layer separation and print dislodgment risk. |
| `SCEN-LS-01` | **Layer Shifting** | Mild | PLA | Nozzle:205°C, Bed:60°C, Flow:100%, Speed:75mm/s | Single-step horizontal stair offset (approx 0.4 mm) along the X-axis. |
| `SCEN-LS-02` | **Layer Shifting** | Moderate | PLA | Nozzle:210°C, Bed:60°C, Flow:100%, Speed:90mm/s | Multiple lateral step offsets along Y-axis following fast non-print jerk moves. |
| `SCEN-LS-03` | **Layer Shifting** | Severe | PETG | Nozzle:235°C, Bed:75°C, Flow:102%, Speed:110mm/s | Discontinuous massive layer translation (>4 mm), completely compromising part geometry. |
| `SCEN-UND-01` | **Under Extrusion** | Mild | PLA | Nozzle:190°C, Bed:60°C, Flow:92%, Speed:65mm/s | Intermittent pinhole gaps visible across upper solid infill surface. |
| `SCEN-UND-02` | **Under Extrusion** | Moderate | PLA | Nozzle:185°C, Bed:60°C, Flow:85%, Speed:70mm/s | Perimeter bead separation, fragile delaminating layers, and sparse infill voids. |
| `SCEN-UND-03` | **Under Extrusion** | Severe | PETG | Nozzle:215°C, Bed:70°C, Flow:75%, Speed:80mm/s | Severe volumetric starvation: sponge-like brittle walls and discontinuous extrusions. |
| `SCEN-OVR-01` | **Over Extrusion** | Mild | PLA | Nozzle:215°C, Bed:60°C, Flow:108%, Speed:50mm/s | Rough tactile top surface with slight polymer ridge build-up along perimeter seams. |
| `SCEN-OVR-02` | **Over Extrusion** | Moderate | PLA | Nozzle:225°C, Bed:60°C, Flow:115%, Speed:45mm/s | Prominent surface blobs, bulging outer walls, and visible nozzle drag furrows. |
| `SCEN-OVR-03` | **Over Extrusion** | Severe | PETG | Nozzle:250°C, Bed:75°C, Flow:125%, Speed:40mm/s | Excess molten polymer overflow, loss of dimensional tolerance, and nozzle tip scraping. |
| `SCEN-BED-01` | **Bed Adhesion** | Mild | PLA | Nozzle:200°C, Bed:50°C, Flow:100%, Speed:50mm/s | First layer lines rounded rather than flat; loose perimeter easily peeled by hand. |
| `SCEN-BED-02` | **Bed Adhesion** | Moderate | PLA | Nozzle:200°C, Bed:45°C, Flow:100%, Speed:55mm/s | Significant detachment of first layer skirt and infill lines during initial skirt loop. |
| `SCEN-BED-03` | **Bed Adhesion** | Severe | PETG | Nozzle:230°C, Bed:55°C, Flow:100%, Speed:50mm/s | Total print release from bed surface; extruded filament forms dragging bird's nest. |
