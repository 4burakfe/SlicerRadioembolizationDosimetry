#-----------------------------------------------------------------------------
set(MODULE_NAME Taranis)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  TaranisLib/__init__.py
  TaranisLib/case.py
  TaranisLib/lsf.py
  TaranisLib/memory.py
  TaranisLib/controller.py
  TaranisLib/doseguard.py
  TaranisLib/dosimetry.py
  TaranisLib/ai.py
  TaranisLib/roles.py
  TaranisLib/segmentops.py
  TaranisLib/segtools.py
  TaranisLib/timing.py
  TaranisLib/toolbar.py
  TaranisLib/views.py
  TaranisLib/visibility.py
  TaranisLib/workflow.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/Icons/Epona.png
  Resources/banner.png
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)
endif()
