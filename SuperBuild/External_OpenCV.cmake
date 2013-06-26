set(extProjName "OpenCV")
if(DEFINED OpenCV_DIR AND NOT EXISTS ${OpenCV_DIR})
 message(FATAL_ERROR "${extProjName}_DIR variable is defined but corresponds to non-existing directory (${${extProjName}_DIR})")
endif()

set(OPENCV_GIT_REPO "https://github.com/Itseez/opencv.git")
set(OPENCV_GIT_TAG "2.4.5")

if(NOT DEFINED OpenCV_DIR)
  set(OpenCV_DEPEND OpenCV)
  set(proj OpenCV)
  
  set(OPENCV_EXTERNAL_PROJECT_ARGS)
  if(NOT CMAKE_CONFIGURATION_TYPES)
    list(APPEND OPENCV_EXTERNAL_PROJECT_ARGS -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE})
  endif()

  ExternalProject_add(${proj}
    SOURCE_DIR ${proj}
    BINARY_DIR ${proj}-build
    
    GIT_REPOSITORY ${OPENCV_GIT_REPO}
    GIT_TAG ${OPENCV_GIT_TAG}
    CMAKE_ARGS
      ${CMAKE_OSX_EXTERNAL_PROJECT_ARGS}
      ${COMMON_EXTERNAL_PROJECT_ARGS}
      ${OPENCV_EXTERNAL_PROJECT_ARGS}
      -DBUILD_EXAMPLES:BOOL=OFF
      -DBUILD_TESTING:BOOL=OFF
      -DBUILD_NEW_PYTHON_SUPPORT:BOOL=OFF
      -DBUILD_TESTS:BOOL=OFF
      -DWITH_FFMPEG:BOOL=ON
      -DWITH_JASPER:BOOL=OFF
      -DWITH_OPENEXR:BOOL=OFF
      -DWITH_PVAPI:BOOL=OFF
      -DWITH_JPEG:BOOL=ON
      -DWITH_TIFF:BOOL=OFF
      -DWITH_PNG:BOOL=ON
## The following might cause build issues, here for testing
      -DENABLE_SSE:BOOL=ON
      -DENABLE_SSE2:BOOL=ON
      -DENABLE_SSE3:BOOL=ON
      -DENABLE_SSE41:BOOL=ON
      -DENABLE_SSE42:BOOL=ON
      -DENABLE_SSSE3:BOOL=ON
## OpenCV Options
      -DBUILD_opencv_calib3d:BOOL=ON
      -DBUILD_opencv_contrib:BOOL=ON
      -DBUILD_opencv_core:BOOL=ON
      -DBUILD_opencv_features2d:BOOL=ON
      -DBUILD_opencv_flann:BOOL=ON
      -DBUILD_opencv_highgui:BOOL=ON
      -DBUILD_opencv_imgproc:BOOL=ON
      -DBUILD_opencv_legacy:BOOL=ON
      -DBUILD_opencv_ml:BOOL=ON
      -DBUILD_opencv_nonfree:BOOL=ON
      -DBUILD_opencv_objdetect:BOOL=ON
      -DBUILD_opencv_photo:BOOL=ON
      -DBUILD_opencv_python:BOOL=ON
      -DBUILD_opencv_stitching:BOOL=ON
      -DBUILD_opencv_ts:BOOL=ON
      -DBUILD_opencv_video:BOOL=ON
      -DBUILD_opencv_videostab:BOOL=ON
      -DBUILD_SHARED_LIBS:BOOL=ON
      -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_BINARY_DIR}/${proj}-build
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
    )
  set(OpenCV_DIR ${CMAKE_BINARY_DIR}/${proj}-build)
endif(NOT DEFINED OpenCV_DIR)
