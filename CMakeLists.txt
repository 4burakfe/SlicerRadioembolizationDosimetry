cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(RadioembolizationDosimetry)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/4burakfe/SlicerRadioembolizationDosimetry")
set(EXTENSION_CATEGORY "Nuclear Medicine")
set(EXTENSION_CONTRIBUTORS "Burak Demir")
set(EXTENSION_DESCRIPTION "Taranis - Radioembolization Dosimetry Suite. A collection of integrated modules and workflows that allow users to perform voxel-based dosimetry with patient-relative and absolute calibration methods for treatments of liver tumors with Y-90 microspheres. Supports registration, LSF calculation, segmentation with many tools.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/4burakfe/SlicerRadioembolizationDosimetry/refs/heads/main/taranis_logo.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/4burakfe/SlicerRadioembolizationDosimetry/refs/heads/main/ss_relative.jpg https://raw.githubusercontent.com/4burakfe/SlicerRadioembolizationDosimetry/refs/heads/main/ss_absolut.jpg https://raw.githubusercontent.com/4burakfe/SlicerRadioembolizationDosimetry/refs/heads/main/ss_segment.jpg https://raw.githubusercontent.com/4burakfe/SlicerRadioembolizationDosimetry/refs/heads/main/ss_lsf.jpg")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(Taranis)
add_subdirectory(easy_reg)
add_subdirectory(LSFcalc)
add_subdirectory(CBCTPlanning)
add_subdirectory(RadioembolizationDosimetryAbsolute)
add_subdirectory(RadioembolizationDosimetryRelative)
add_subdirectory(TaranisSegmentation)
add_subdirectory(TaranisDataRoles)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
