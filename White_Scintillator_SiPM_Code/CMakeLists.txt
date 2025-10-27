cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(sim
    VERSION 1.0
    DESCRIPTION "Geant4 Simulation for muCT"
    LANGUAGES CXX
)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(sources ${PROJECT_SOURCE_DIR}/src/DetectorConstruction.cc

    # ${PROJECT_SOURCE_DIR}/src/PhysicsList.cc
    ${PROJECT_SOURCE_DIR}/src/ActionInitialization.cc
    ${PROJECT_SOURCE_DIR}/src/PrimaryGeneratorAction.cc
    ${PROJECT_SOURCE_DIR}/src/SensitiveDetector.cc
    ${PROJECT_SOURCE_DIR}/src/RunAction.cc
    ${PROJECT_SOURCE_DIR}/src/EventAction.cc
    ${PROJECT_SOURCE_DIR}/src/SteppingAction.cc
)

find_package(Geant4 REQUIRED ui_all vis_all)

add_executable(${PROJECT_NAME} sim.cc ${sources})

# include directories
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME} PRIVATE ${Geant4_LIBRARIES})

target_compile_definitions(${PROJECT_NAME} PRIVATE WIN32)
