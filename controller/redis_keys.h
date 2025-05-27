/**
 * @file redis_keys.h
 * @brief Contains all redis keys for simulation and control.
 * 
 */

std::string JOINT_ANGLES_KEY = "sai::sim::panda::sensors::q";
std::string JOINT_VELOCITIES_KEY = "sai::sim::panda::sensors::dq";
std::string JOINT_TORQUES_COMMANDED_KEY = "sai::sim::panda::actuators::fgc";


std::string SCRUB_POINTS_KEY = "sai::commands::Sponge::scrub_points";