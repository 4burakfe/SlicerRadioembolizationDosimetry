cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(RadioembolizationDosimetry)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/4burakfe/SlicerRadioembolizationDosimetry")
set(EXTENSION_CATEGORY "Nuclear Medicine")
set(EXTENSION_CONTRIBUTORS "Burak Demir")
set(EXTENSION_DESCRIPTION "Taranis - Radiembolization Dosimetry. A simple module that allows users to perform voxel-based dosimetry with patient-relative and absolute calibration methods for treatments of liver tumors with Y-90 microspheres.")
set(EXTENSION_ICONURL "https://github.com/4burakfe/SlicerRadioembolizationDosimetry/blob/3e18fb523c923a330f4e347263ff3bbae9c5eae8/taranis_logo.png?raw=true")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/4burakfe/SlicerRadioembolizationDosimetry/3e18fb523c923a330f4e347263ff3bbae9c5eae8/Screenshot1.jpg")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(easy_reg)
add_subdirectory(LSFcalc)
add_subdirectory(RadioembolizationDosimetry)
add_subdirectory(RadioembolizationDosimetryabs)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
