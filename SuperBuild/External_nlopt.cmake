set(extProjName "nlopt")
if(DEFINED NLOPT_DIR AND NOT EXISTS ${NLOPT_DIR})
 message(FATAL_ERROR "${extProjName}_DIR variable is defined but corresponds to non-existing directory (${${extProjName}_DIR})")
endif()

message(STATUS "External nlopt")
set(OPENCV_GIT_REPO "https://dkostro@bitbucket.org/dkostro/nlopt.git")

if(NOT DEFINED NLOPT_DIR)
  set(proj nlopt)
  
  set(OPENCV_EXTERNAL_PROJECT_ARGS)
  if(NOT CMAKE_CONFIGURATION_TYPES)
    list(APPEND OPENCV_EXTERNAL_PROJECT_ARGS -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE})
  endif()

message(STATUS "nlopt external proj")
  ExternalProject_add(${proj}
    GIT_REPOSITORY ${OPENCV_GIT_REPO}
    SOURCE_DIR ${proj}
    BINARY_DIR ${proj}-build
    CMAKE_ARGS
      ${CMAKE_OSX_EXTERNAL_PROJECT_ARGS}
      ${COMMON_EXTERNAL_PROJECT_ARGS}
      ${NLOPT_EXTERNAL_PROJECT_ARGS}
      -DNLOPT_BUILD_SHARED:BOOL=OFF
      -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_BINARY_DIR}/${proj}-build
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
    )
  
  message(STATUS "Setting vars")
  set(NLOPT_DIR ${CMAKE_BINARY_DIR}/${proj}-build)
  # Hack to get the proper environment variables
  set(NLOPT_INCLUDE_DIRECTORIES
    ${NLOPT_DIR}/../nlopt
    ${NLOPT_DIR}
    ${NLOPT_DIR}/../nlopt/stogo 
    ${NLOPT_DIR}/../nlopt/util 
    ${NLOPT_DIR}/../nlopt/direct 
    ${NLOPT_DIR}/../nlopt/cdirect 
    ${NLOPT_DIR}/../nlopt/praxis 
    ${NLOPT_DIR}/../nlopt/luksan 
    ${NLOPT_DIR}/../nlopt/crs 
    ${NLOPT_DIR}/../nlopt/mlsl 
    ${NLOPT_DIR}/../nlopt/mma 
    ${NLOPT_DIR}/../nlopt/cobyla 
    ${NLOPT_DIR}/../nlopt/newuoa 
    ${NLOPT_DIR}/../nlopt/neldermead 
    ${NLOPT_DIR}/../nlopt/auglag 
    ${NLOPT_DIR}/../nlopt/bobyqa 
    ${NLOPT_DIR}/../nlopt/isres 
    ${NLOPT_DIR}/../nlopt/slsqp
    ${NLOPT_DIR}/../nlopt/api)
  set(NLOPT_LIBRARIES ${NLOPT_DIR}/libnlopt.a)
  
  function(list_to_string separator input_list output_string_var)
    set(_string "")
    cmake_policy(PUSH)
    cmake_policy(SET CMP0007 OLD)
    # Get list length
    list(LENGTH input_list list_length)
    # If the list has 0 or 1 element, there is no need to loop over.
    if(list_length LESS 2)
      set(_string  "${input_list}")
    else()
      math(EXPR last_element_index "${list_length} - 1")
      foreach(index RANGE ${last_element_index})
        # Get current item_value
        list(GET input_list ${index} item_value)
        # .. and append to output string
        set(_string  "${_string}${item_value}")
        # Append separator if current element is NOT the last one.
        if(NOT index EQUAL last_element_index)
          set(_string  "${_string}${separator}")
        endif()
      endforeach()
    endif()
    set(${output_string_var} ${_string} PARENT_SCOPE)
    cmake_policy(POP)
  endfunction()
  list_to_string(${ep_list_separator} "${NLOPT_INCLUDE_DIRECTORIES}" ep_NLOPT_INCLUDE_DIRECTORIES)
  message(STATUS "include string ${ep_NLOPT_INCLUDE_DIRECTORIES}")
  
  
endif(NOT DEFINED NLOPT_DIR)
