project(vtkSlicer${MODULE_NAME}ModuleLogic)

set(KIT ${PROJECT_NAME})

set(${KIT}_EXPORT_DIRECTIVE "VTK_SLICER_${MODULE_NAME_UPPER}_MODULE_LOGIC_EXPORT")

set(${KIT}_INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/SurfFeatures/includes 
  )

set(${KIT}_SRCS
  vtkSlicer${MODULE_NAME}Logic.cxx
  vtkSlicer${MODULE_NAME}Logic.h
  )
  


set(${KIT}_TARGET_LIBRARIES
  ${ITK_LIBRARIES}
  )

message(STATUS "nlopt include: ${NLOPT_INCLUDE_DIRECTORIES}")
message(STATUS "nlopt library: ${NLOPT_LIBRARIES}")
message(STATUS "nlopt dir: ${NLOPT_DIR}")

#-----------------------------------------------------------------------------
SlicerMacroBuildModuleLogic(
  NAME ${KIT}
  EXPORT_DIRECTIVE ${${KIT}_EXPORT_DIRECTIVE}
  INCLUDE_DIRECTORIES ${${KIT}_INCLUDE_DIRECTORIES} ${OpenCV_INCLUDE_DIRS}
  SRCS ${${KIT}_SRCS}
  TARGET_LIBRARIES ${${KIT}_TARGET_LIBRARIES} ${OpenCV_LIBS}
)
  
get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
  message(STATUS "${_variableName}=${${_variableName}}")
endforeach()

add_executable(surfcmd surfcmd.cxx logicUtility.hpp surfLogicUtility.hpp)
target_link_libraries(surfcmd ${KIT})

add_executable(testcmd testcmd.cxx surfLogicUtility.hpp)
target_link_libraries(testcmd ${KIT})
