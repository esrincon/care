set(CONTROLLER_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DCONTROLLER_FOLDER="${CONTROLLER_FOLDER}")

set(APP_NAME panda_care)
# create an executable
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CARE_BINARY_DIR}//${APP_NAME})
add_executable(${APP_NAME}-simviz simviz.cpp)
add_executable(${APP_NAME}-controller controller.cpp)

# and link the library against the executable
TARGET_LINK_LIBRARIES (${APP_NAME}-simviz ${CARE_COMMON_LIBRARIES})
TARGET_LINK_LIBRARIES (${APP_NAME}-controller ${CARE_COMMON_LIBRARIES})
