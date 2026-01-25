# ICE-signal-processing-pipeline
This repository contains the code and a technical report for the post-processing of a **Compression Ignition engine (CI)** endowed with **Exhaust Gas Recirculation (EGR)**. This project was developed
as a part of the Propulsion Systems and their Applications to Vehicle course at Politecnico di Torino. 
## Objectives and Project Overview
The aim of this work is the analysis of experimental data from steady-state condition tests and in-clylinder pressure data using **MATLAB** to evaluate performance parameters, combustion metrics, to realize the Fourier analysis and the dimensioning
of a flywheel.
This project is divided in four parts:
- **Steady-State Test Evaluation and In-Cylinder Pressure Analysis**. The initial dataset, comprising variables such as engine speed, torque, fuel mass flow,
and intake conditions, was imported from the datasheet for processing corrected according to the ISO 1585 standard. From these data some parameters, such as full load characteristic, brake specific fuel consumption, fuel conversion efficiency and volumetric efficiency are calculated and plotted against engine speed.
 In-cylinder pressure data are then filtered testing different types of filters. The correlation between the filtered pressure and volume was visualized through p–V diagrams. These were plotted in linear coordinates to visualize the work loop. In addition, The Brake Mean Effective Pressure (BMEP) was derived from the corrected dynamometer torque and the gross mechanical efficiency ($η_m$) is evaluated.
- **Fourier analysis**. Initially, the in-cylinder pressure signal is reconstructed using the Fourier Transform. The In-cylinder pressure harmonic spectrum is described by their amplitude and their
order. Then, instantaneous torque is computed as a function of the crank angles for both single-cylinder and multi-cylinder engines. Finally, the rotating vectors of the different orders or the four cylinders are plotted to understand the global contribution of each harmonic.
- **Flywheel dimensioning** This section addresses the dimensioning of a flywheel to regulate angular velocity fluctuations and ensure smooth engine operation. By analyzing rotational dynamics, the study calculates the moment of inertia required to maintain a kinematic irregularity below 0.01.
- **Heat Release Rate (HRR) analysis** an analysis of the Heat Release Rate (HRR) was performed
based on experimental in-cylinder pressure filtered data  The analysis is conducted using a pressure-based
Rassweiler-Withrow (RW) method and a single-zone model, assuming the gas inside the combustion chamber as homogeneous. The purpose of this analysis is to compute the curve of the mass
fraction burned ($X_b$) and to identify key combustion parameters such as the Start of
Combustion (SOC), the Start of Ignition (SOI), the Ignition Delay (ID) and the mass
fraction burned at 10% of the crank angle (MFB10), at 50% of crank angle (MFB50)
and at 90% of crank angle (MFB90).

## Key results and visualizations
### Thermodynamic Performance
| Parameter | Value |
| :--- | :--- |
| **Gross IMEP** | 19.63 bar |
| **Net IMEP** | 19.33 bar |
| **Brake Mean Effective Pressure (BMEP)** | 18.10 bar |
| **Mechanical Efficiency** ($\eta_{m}$) | 0.92 |

![filters](imgs/filters.pdf) 
Comparison between different filters.

![p-V diagram](imgs/pV.pdf)
In-cylinder pressure vs Volume (log scale).

### Fourier analysis
![Torque approximation using FFT](imgs/T-Fourier.pdf)
visual representation of the torque approximation using the FFT.

![Amplitude spectrum for the multi-cylinder engine](imgs/Fourier_spectrum.pdf)
Amplitude spectrum for the multi-cylinder engine. Only the harmonics of order n=$\frac{i}{2}$ are in phase.

### Flywheel dimensioning 
| Parameter | Value |
| :--- | :--- |
|**Required Flywheel Diameter (Single-Cylinder)** | 0.565 m |
|**Required Flywheel Diameter (Multi-Cylinder)** | 0.471 m |

![Work and pressure over crank angle](imgs/Flywheel_dimensioning.pdf)
Shaft and resistant Work, tangential and resistant pressure trend over crank angle.

![Angular speed with flywheel](imgs/With_flywheel.pdf)
![Angular speed without flywheel](imgs/Without_flywheel.pdf)
Effect of the implementation of the flywheel on the engine speed for the multi-cylinder engine.

### Combustion Metrics
| Parameter | Value |
| :--- | :--- |
|**Start of Injection (SOI)** | 345.40° CA |
|**Start of Combustion (SOC)** |356.50° CA |
|**Ignition Delay (ID)** | 11.10° CA |
|**Combustion Phasing (MFB50)** | 376.00° CA |

![MFB trend](imgs/MFB.pdf)
Comparison between the in_cylinder pressure and the MFB over crank angle. Important MFB stages are highlighted.

## How to run
- Ensure **MATLAB** is installed with the **Signal processing toolbox**.
- Add 'datasheet_2025.xlsx', 'ifile_2000FL.mat' to the working directory.
- Run 'main_analysis.mat'. 

## File Descriptions
* `main_analysis.m`: The primary script containing data import, performance correction, filtering, Fourier analysis, and HRR calculations.
* `Technical_report.pdf`: Detailed documentation including the mathematical framework, methodology, and graphical results.
* `datasheet_2025.xlsx`: Datasheet containing experimental data relative to steady-state tests and engine parameters.
* `ifile_2000FL.mat`: Data relative to in-cylinder pressure analysis.
* `imgs`: folder including the most important plots obtained from this project.

