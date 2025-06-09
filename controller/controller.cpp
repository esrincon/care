/**
 * @file controller.cpp
 * @brief Controller file
 * 
 */

 #include <SaiModel.h>
 #include "SaiPrimitives.h"
 #include "redis/RedisClient.h"
 #include "timer/LoopTimer.h"
 #include <signal.h>
  #include "redis_keys.h"
 #include <signal.h>
 #include <Eigen/Geometry>
 #include <Eigen/Dense>
 #include <cmath>
 #include <iostream>
 #include <string>
 #include <deque>
 #include <vector>
 #include <sstream>  


 using namespace std;
 using namespace Eigen;
 using namespace SaiPrimitives;
 
 bool runloop = false;
 void sighandler(int) {runloop = false;}
 
 bool simulation = false;
 
 bool reached_vel = false;

 bool inertia_regularization = true;

 int region = 0;

 
 // States 
 enum State {
	 POSTURE = 0, 
	 INITIAL_ROTATION,	
	 INITIAL_APPROACH1, 
	 INITIAL_APPROACH2,
	 CLEAN_VERTICAL, 
	 CLEAN_HORIZONTAL,
	 SHORT_TRANSITION,
	 LONG_TRANSITION, 
	 PAUSE,
	 DONE,
	 STOP
 };

 int prev_state = -1;

#include <sstream>   // for istringstream


double triangleWave(double time,
                    double frequency,
                    double torso_bottom,
                    double amplitude) {
    static long counter = 0;
    double period      = 1.0 / frequency;
    double x           = fmod(time, period);     // position within [0, period)
    double half_period = period / 2.0;
    double slope       = amplitude / half_period; // rise (and fall) rate

    ++counter;
    if (x < half_period) {
        // Rising phase: from torso_bottom → torso_bottom + amplitude
        double y = torso_bottom + slope * x;
        if (counter % 100 == 0) {
            std::cout << "position (rising): " << y << std::endl;
        }
        return y;
    } else {
        // Falling phase: from torso_bottom + amplitude → torso_bottom
        double x_rel = x - half_period; 
        double y = torso_bottom + amplitude - slope * x_rel;
        if (counter % 100 == 0) {
            std::cout << "position (falling): " << y << std::endl;
        }
        return y;
    }
}


bool parseScrubBlob(
    const std::string &blob,
    std::vector<Eigen::Vector3d> &positions,
    std::vector<Eigen::Vector3d> &orientations)
{
    // find the separator
    auto bar = blob.find('|');
    if (bar == std::string::npos) return false;    std::string pos_block = blob.substr(0, bar);
    std::string ori_block = blob.substr(bar + 1);    // helper: split 'a;b;c' → {"a","b","c"}
    auto split_semis = [&](const std::string &in){
        std::vector<std::string> out;
        size_t start = 0;
        while (start < in.size()) {
            auto sep = in.find(';', start);
            if (sep == std::string::npos) sep = in.size();
            out.push_back(in.substr(start, sep - start));
            start = sep + 1;
        }
        return out;
    };    // parse one block of "a,b,c" entries into a Vector3d list
    auto parse_vec3 = [&](const std::string &block,
                          std::vector<Eigen::Vector3d> &out_list)
    {
        for (auto &entry : split_semis(block)) {
            if (entry.empty()) continue;
            std::istringstream ss(entry);
            double a, b, c; char comma1, comma2;
            if (!(ss >> a >> comma1 >> b >> comma2 >> c)
                || comma1 != ',' || comma2 != ',')
            {
                return false;
            }
            out_list.emplace_back(a, b, c);
        }
        return true;
    };    // actually parse both halves
    if (!parse_vec3(pos_block, positions))      return false;
    if (!parse_vec3(ori_block, orientations))   return false;    return true;
}



 int main() {
	if (simulation) {
		cout << "SIMULATION TRUE" << endl;
		JOINT_ANGLES_KEY = "sai::sim::panda::sensors::q";
		JOINT_VELOCITIES_KEY = "sai::sim::panda::sensors::dq";
		JOINT_TORQUES_COMMANDED_KEY = "sai::sim::panda::actuators::fgc";
	} else {
		cout << "SIMULATION FALSE" << endl;
		JOINT_ANGLES_KEY = "sai::sensors::FrankaRobot::joint_positions";
		JOINT_VELOCITIES_KEY = "sai::sensors::FrankaRobot::joint_velocities";
		JOINT_TORQUES_COMMANDED_KEY = "sai::commands::FrankaRobot::control_torques";
		SCRUB_POINTS_KEY = "sai::commands::Sponge::scrub_points";
		MASSMATRIX_KEY = "sai::sensors::FrankaRobot::model::mass_matrix";
		ROBOT_SENSED_FORCE_KEY = "sai2::ATIGamma_Sensor::Romeo::force_torque";
	}

	 // Location of URDF files specifying world and robot information
	 static const string robot_file = "../../urdf/panda_arm_sponge.urdf";
 
	 // initial state 
	 int state = POSTURE;
	 string controller_status = "1";
	 
	 // start redis client
	 auto redis_client = SaiCommon::RedisClient();
	 redis_client.connect();
 
	 // set up signal handler
	 signal(SIGABRT, &sighandler);
	 signal(SIGTERM, &sighandler);
	 signal(SIGINT, &sighandler);
 
	 // load robots, read current state and update the model
	 auto robot = std::make_shared<SaiModel::SaiModel>(robot_file, false);
	 robot->setQ(redis_client.getEigen(JOINT_ANGLES_KEY));
	 robot->setDq(redis_client.getEigen(JOINT_VELOCITIES_KEY));
	 robot->updateModel();
 
	 // prepare controller
	 int dof = robot->dof();
	 VectorXd command_torques = VectorXd::Zero(dof);  // panda torques 
	 MatrixXd N_prec = MatrixXd::Identity(dof, dof);
 
	 // Postion of EE intilization 
	 const string control_link   = "link7";
	 const Vector3d control_point = Vector3d(0.0, 0.0, 0.125);
	 //     visual origin (0.1) + half the cylinder length (0.015) = 0.115
	 Affine3d compliant_frame = Affine3d::Identity();
	 compliant_frame.translation() = control_point;
	 auto pose_task = std::make_shared<SaiPrimitives::MotionForceTask>(robot, control_link, compliant_frame);
	 // Will tune depending on the task
	 pose_task->setPosControlGains(100, 10, 0);
	 pose_task->setOriControlGains(100, 20, 10);
 
	 Vector3d ee_pos;
	 Matrix3d ee_ori;

	Matrix3d ee_ori_current;

	// force sensing
	// set sensor frame transform in end-effector frame
	Affine3d sensor_transform_in_link = Affine3d::Identity();
	const Vector3d sensor_pos_in_link = Eigen::Vector3d(0, 0, 0.0406);
	Matrix3d R_link_sensor = Matrix3d::Identity();
	sensor_transform_in_link.translation() = sensor_pos_in_link;
	sensor_transform_in_link.linear() = R_link_sensor;
	pose_task->setForceSensorFrame(control_link, sensor_transform_in_link);

	VectorXd sensed_force_moment_local_frame = VectorXd::Zero(6);
	VectorXd sensed_force_moment_world_frame = VectorXd::Zero(6);
	Vector3d sensed_force_world_frame = Vector3d::Zero();
	Vector3d sensed_moment_world_frame = Vector3d::Zero();
	// VectorXd force_bias = VectorXd::Zero(6);
	double tool_mass = 0.060;
	Vector3d tool_com = Vector3d::Zero();

	Vector3d init_force = Vector3d::Zero();
	Vector3d init_moment = Vector3d::Zero();
	bool first_loop = true;

	if (!simulation) {
		// force_bias << 0.705852, 3.17762, 0.828429, -0.000728745, 0.209324, 0.073354;
		tool_mass = 0.06;
		tool_com = Vector3d(0.0, 0.0, 0.03);
	}

	// remove inertial forces from tool
	Vector3d tool_velocity = Vector3d::Zero();
	Vector3d prev_tool_velocity = Vector3d::Zero();
	Vector3d tool_acceleration = Vector3d::Zero();
	Vector3d tool_inertial_forces = Vector3d::Zero();

	Vector3d desired_force = Vector3d::Zero();
	Vector3d prev_desired_force = Vector3d::Zero();
 
	 // joint task
	 auto joint_task = std::make_shared<SaiPrimitives::JointTask>(robot);
	 joint_task->setGains(250, 20, 30);


    // Desired null space posture for scrubbing
	 VectorXd q_desired(dof);
	 q_desired << 0.581292, 0.0975641, 0.0502793,  -1.77865,  -1.25744,   1.03827,  0.392796;
	 //q_desired *= M_PI / 180.0;

	 
 
	 joint_task->setGoalPosition(q_desired);

	 Vector3d ee_pos_current;

	 Vector3d ee_pos_clean_start;

	 Vector3d ee_pos_desired;

	 Vector3d init_pos;

	pose_task->disableInternalOtg();
	pose_task->enableVelocitySaturation(0.20, M_PI / 3.0);

	pose_task->setForceControlGains(0.05, 10.0, 1.0);
	pose_task->setClosedLoopForceControl(false);
	pose_task->enablePassivity();

	pose_task->setClosedLoopMomentControl(false);
	pose_task->setMomentControlGains(0.05, 10.0, 1.0);
	 // FOR THE VELOCITY HANDLING
	 std::deque<double>   vel_buf;    // will hold up to 100 samples
	 double               vel_sum = 0;
 
	 // create a loop timer
	 runloop = true;
	 double control_freq = 1000;
	 SaiCommon::LoopTimer timer(control_freq, 1e6);

	 int next_state = 0;
 
	 // Timer for clean 1
	 double start_time = -1.0;

	 double pause_time = 0.0;

	unsigned long long controller_counter = 0;

	std::string blob = redis_client.get(SCRUB_POINTS_KEY);
	std::cout << blob << std::endl;
	std::vector<Eigen::Vector3d> scrub_positions, scrub_orientations;

	if (!parseScrubBlob(blob, scrub_positions, scrub_orientations)) {
		std::cerr << "[controller] ERROR parsing scrub_points blob\n";
	} else {
		for (size_t i = 0; i < scrub_positions.size(); ++i) {
			std::cout << "Position " << i << ": " << scrub_positions[i].transpose() << std::endl;
		}
		for (size_t i = 0; i < scrub_orientations.size(); ++i) {
			std::cout << "Orientation " << i << ": " << scrub_orientations[i].transpose() << std::endl;
		}
	}

	 while (runloop) {
		
		timer.waitForNextLoop();
		const double time = timer.elapsedSimTime();
 
		// update robot 
		robot->setQ(redis_client.getEigen(JOINT_ANGLES_KEY));
		robot->setDq(redis_client.getEigen(JOINT_VELOCITIES_KEY));
		MatrixXd M = robot->M();
		if(!simulation) {
			M = redis_client.getEigen(MASSMATRIX_KEY);
			if (inertia_regularization)	{
				M(4, 4) += 0.2;
				M(5, 5) += 0.2;
				M(6, 6) += 0.2;
			}
		}
		robot->updateModel(M);

		// read kinematic data
		ee_pos = robot->position(control_link, control_point);
		ee_ori = robot->rotation(control_link);

		// add bias and ee weight to sensed forces
		sensed_force_moment_local_frame = redis_client.getEigen(ROBOT_SENSED_FORCE_KEY);
		// sensed_force_moment_local_frame -= force_bias;
		Matrix3d R_world_sensor = robot->rotation(control_link);
		R_world_sensor = R_world_sensor * R_link_sensor;
		Vector3d p_tool_local_frame = tool_mass * R_world_sensor.transpose() * Vector3d(0, 0, -9.81);
		sensed_force_moment_local_frame.head(3) += p_tool_local_frame;
		sensed_force_moment_local_frame.tail(3) += tool_com.cross(p_tool_local_frame);

		if (first_loop) {
			init_force = sensed_force_moment_local_frame.head(3);
			init_moment = sensed_force_moment_local_frame.tail(3);
			first_loop = false;
		}
		sensed_force_moment_local_frame.head(3) -= init_force;
		sensed_force_moment_local_frame.tail(3) -= init_moment;
	 
	 	// update forces for pose task
		pose_task->updateSensedForceAndMoment(sensed_force_moment_local_frame.head(3), sensed_force_moment_local_frame.tail(3));
		sensed_force_moment_world_frame.head(3) = R_world_sensor * sensed_force_moment_local_frame.head(3);
		sensed_force_moment_world_frame.tail(3) = R_world_sensor * sensed_force_moment_local_frame.tail(3);

		sensed_force_world_frame = sensed_force_moment_world_frame.head(3);
		sensed_moment_world_frame = sensed_force_moment_world_frame.tail(3);

		 if (state == POSTURE) {
			 // update task model 
			 N_prec.setIdentity();
			 joint_task->updateTaskModel(N_prec);
 
			 command_torques = joint_task->computeTorques();
 
			 if ((robot->q() - q_desired).norm() < 18e-2) {
				 cout << "Posture To Motion" << endl;
				 pose_task->reInitializeTask();
				 joint_task->reInitializeTask();

				 ee_ori_current = ee_ori;
 
				 state = INITIAL_ROTATION;
			 }
 
		 } else if (state == INITIAL_ROTATION) {
			 // TEST
			if (start_time < 0.0) {
				start_time = time;	
				init_pos = robot->position(control_link, control_point);
			 }
			 double t_elapsed = time - start_time;

					Matrix3d sponge_ori;
				sponge_ori <<
						1.0,  0.0,  0.0,
						0.0,  0.0,  -1.0,
						0.0, 1.0,  0.0;
   				pose_task->setGoalOrientation(sponge_ori);
			

			    
			VectorXd kp_gain_orientation(3);
			kp_gain_orientation << 100.0, 100.0, 100.0;
			VectorXd kv_gain_orientation = VectorXd::Constant(3, 10.0);
			VectorXd ki_gain_orientation = VectorXd::Zero(3);
			pose_task->setOriControlGains(kp_gain_orientation, kv_gain_orientation, ki_gain_orientation);
 
			 N_prec.setIdentity();
			 pose_task->updateTaskModel(N_prec);
			 joint_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());
 
			 command_torques = pose_task->computeTorques() + joint_task->computeTorques();

			 // check if orientation goal reached
			 if (t_elapsed > 3.0) {
				 cout << "Orientation Achieved" << endl;
				 //pause_time = 0.0;

				start_time = -1.0;
				 
				 // print out position before approaching
				 ee_pos_current = robot->position(control_link, control_point);
				 cout << "EE position: " << ee_pos_current.transpose() << endl;
				 
				 pose_task->reInitializeTask();
				 joint_task->reInitializeTask();

				 if (region == 16) state = STOP;
				 else state = INITIAL_APPROACH1;

			 }
		 } else if (state == INITIAL_APPROACH1) {
			double offset = 0.1;

			ee_pos_desired = scrub_positions[region];
			
			ee_pos_desired[1] += offset; // add offset along Y (depth) before approaching slowly
			pose_task->setGoalPosition(ee_pos_desired);
			 
			 N_prec.setIdentity();
			 pose_task->updateTaskModel(N_prec);
			 joint_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());
 
			 command_torques = pose_task->computeTorques() + joint_task->computeTorques();
 
			 // Print position				
			 ee_pos_current = robot->position(control_link, control_point);
			 cout << "INITIAL_APPROACH1 | Current:  "
				 << ee_pos_current.transpose()
				 << "  Desired:  " << ee_pos_desired.transpose() 
				 << "  Err: " << (ee_pos_current - ee_pos_desired).norm()
				 << endl;
 
			 const double thresh = 5e-2;  // 1 cm on XY maybe?
			 if ((ee_pos_current - ee_pos_desired).norm() < thresh) {
				pose_task->reInitializeTask();
				joint_task->reInitializeTask();
				pause_time = 0.0;
				state = INITIAL_APPROACH2;
			 }

			} else if (state == INITIAL_APPROACH2) {
				
				ee_pos_desired = scrub_positions[region];
				pose_task->setGoalPosition(ee_pos_desired);

				VectorXd kp_gain_orientation(3);
				kp_gain_orientation << 100.0, 100.0, 100.0;
				VectorXd kv_gain_orientation = VectorXd::Constant(3, 10.0);
				VectorXd ki_gain_orientation = VectorXd::Zero(3);
			 	pose_task->setOriControlGains(kp_gain_orientation, kv_gain_orientation, ki_gain_orientation);

				// desired orientation for approach
				
				AngleAxisd A_roll  (scrub_orientations[region][0], Vector3d::UnitX());
				AngleAxisd A_pitch (scrub_orientations[region][1], Vector3d::UnitY());
				AngleAxisd A_yaw   (scrub_orientations[region][2], Vector3d::UnitZ());
				Matrix3d sponge_ori = (A_yaw * A_pitch * A_roll).toRotationMatrix();
				pose_task->setGoalOrientation(sponge_ori);
			

				Vector3d force_axis_compliant_frame = Vector3d::UnitZ();
				Vector3d force_axis_base_frame = ee_ori * force_axis_compliant_frame;

				if (controller_counter % 100 == 0) {
					pose_task->parametrizeForceMotionSpaces(1, force_axis_base_frame);
				}

				double desired_force_norm = 3.0;
				desired_force = desired_force_norm * force_axis_base_frame;
				pose_task->setGoalForce(desired_force);

				N_prec.setIdentity();
				pose_task->updateTaskModel(N_prec);

				joint_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());

				command_torques = pose_task->computeTorques() + joint_task->computeTorques();

				const double thresh = -0.001; 

				if (sensed_force_moment_local_frame[2] > 3.0) {
					cout << "Aligned at center height—starting CLEAN_VERTICAL/HORIZONTAL\n";
					prev_state = state;
					
					// reset tasks here
					start_time = -1.0;
					pause_time = 0.0;
					ee_pos_current = robot->position(control_link, control_point);
					pose_task->reInitializeTask();
					joint_task->reInitializeTask();

					state = CLEAN_VERTICAL;
				}
 
		 } else if (state == CLEAN_VERTICAL) {
			 // initialize start time
			 double z_des;
			 bool UP;
			 if (start_time < 0.0) {
				start_time = time;
				UP = true;
			 }
			 double t_elapsed = time - start_time;
 
			 // ■ compute amplitude so sponge never crosses top/bottom
			 const double amplitude_z =  0.10;

			double z_top = ee_pos_current[2] + amplitude_z;
			double z_bottom = ee_pos_current[2];

			if (UP) {
				z_des = z_top;
			} else {
				z_des = z_bottom;
			}
			
			if (abs(z_des - ee_pos[2]) < 0.03) {
				UP = !UP;
			}
		
 			ee_pos_desired << ee_pos_current[0], ee_pos_current[1], z_des;
			pose_task->setGoalPosition(ee_pos_desired);

			Vector3d force_axis_compliant_frame = Vector3d::UnitZ();
			Vector3d force_axis_base_frame = ee_ori * force_axis_compliant_frame;
			if (controller_counter % 100 == 0) {
					pose_task->parametrizeForceMotionSpaces(1, force_axis_base_frame);
				}

			double desired_force_norm = 5.0;
			desired_force = desired_force_norm * force_axis_base_frame;
			pose_task->setGoalForce(desired_force);
			pose_task->setGoalMoment(VectorXd::Zero(3));
		
			 VectorXd kp_gain_orientation(3);
			 kp_gain_orientation << 100.0,100.0, 100.0;
			 VectorXd kv_gain_orientation = VectorXd::Constant(3, 10.0);
			 VectorXd ki_gain_orientation = VectorXd::Zero(3);
			 pose_task->setOriControlGains(kp_gain_orientation, kv_gain_orientation, ki_gain_orientation);

			 VectorXd kp_gain(3);
			 kp_gain << 100.0, 100.0, 100.0;
			 VectorXd kv_gain = VectorXd::Constant(3, 20.0);
			 VectorXd ki_gain = VectorXd::Zero(3);
			 pose_task->setPosControlGains(kp_gain, kv_gain, ki_gain);
 
			 // ■ task hierarchy + torque
			 N_prec.setIdentity();
			 pose_task->updateTaskModel(N_prec);
			 joint_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());
			 command_torques = pose_task->computeTorques()
							 + joint_task->computeTorques();

			 if (t_elapsed > 5.0) {
				cout << "CLEAN VERTICAL complete, switching to TRANSITION\n";

				std::size_t maxIdx = scrub_positions.size(); // assume they’re same length
				if (region + 1 >= maxIdx) {
					std::cout << "Reached last region; going to POSTURE\n";
					state = DONE;
				} else {
					region += 1;
					 if (region == 9) region++;
					state = SHORT_TRANSITION;
					pose_task->reInitializeTask();
				joint_task->reInitializeTask();
				
    			}
			 }
		 }  else if (state == SHORT_TRANSITION) {
			// 1) set your desired goal
			ee_pos_current = robot->position(control_link, control_point);

			//  ee_pos_desired << 0.554677,-0.168536, 0.169195+ 0.02;
			cout << "SHORT TRANSITION (region=" << region << ")\n";
			ee_pos_desired = scrub_positions[region];
			pose_task->setGoalPosition(ee_pos_desired);

			VectorXd kp_gain_orientation(3);
			kp_gain_orientation << 100.0, 100.0, 100.0;
			VectorXd kv_gain_orientation = VectorXd::Constant(3, 10.0);
			VectorXd ki_gain_orientation = VectorXd::Zero(3);
			pose_task->setOriControlGains(kp_gain_orientation, kv_gain_orientation, ki_gain_orientation);

			if (region < 12) {
				AngleAxisd A_roll  (scrub_orientations[region][0], Vector3d::UnitX());
				AngleAxisd A_pitch (scrub_orientations[region][1], Vector3d::UnitY());
				AngleAxisd A_yaw   (scrub_orientations[region][2], Vector3d::UnitZ());
				Matrix3d sponge_ori = (A_yaw * A_pitch * A_roll).toRotationMatrix();
				pose_task->setGoalOrientation(sponge_ori);
			} else {
				Matrix3d sponge_ori;
				sponge_ori <<
						1.0,  0.0,  0.0,
						0.0,  0.0,  -1.0,
						0.0, 1.0,  0.0;
   				pose_task->setGoalOrientation(sponge_ori);
			}

			Vector3d force_axis_compliant_frame = Vector3d::UnitZ();
			Vector3d force_axis_base_frame = ee_ori * force_axis_compliant_frame;
			if (controller_counter % 100 == 0) {
					pose_task->parametrizeForceMotionSpaces(1, force_axis_base_frame);
			}

			double desired_force_norm = 2.0;
			desired_force = desired_force_norm * force_axis_base_frame;
			pose_task->setGoalForce(desired_force);

			 N_prec.setIdentity();
			 pose_task->updateTaskModel(N_prec);
			 joint_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());
 
			 // 4) compute torques (now with sat’d vels)
			 command_torques = pose_task->computeTorques() + joint_task->computeTorques();

			 cout << "error: " << abs(ee_pos_current[0] - ee_pos_desired[0]) << "/t" << abs(ee_pos_current[2] - ee_pos_desired[2]) << endl;
  
			 const double thresh = 4e-2;  // 1 cm on XY maybe
			 if (abs(ee_pos_current[0] - ee_pos_desired[0]) < thresh && abs(ee_pos_current[2] - ee_pos_desired[2]) < thresh) {

				cout << "TRANSITION complete, moving back to CLEAN VERTICAL" << endl;

				start_time = -1.0;
				state = CLEAN_VERTICAL;

				ee_pos_current = robot->position(control_link, control_point);

				pose_task->reInitializeTask();
				joint_task->reInitializeTask();
			 }
		} else if (state == LONG_TRANSITION) {
			 cout << "LONG TRANSITION" << endl;
			 ee_pos_desired = scrub_positions[region];
			 ee_pos_desired[1] += 0.15;
			//  ee_pos_desired[2] += 0.03;

			 pose_task->setGoalPosition(ee_pos_desired);

			VectorXd kp_gain_orientation(3);
			kp_gain_orientation << 100.0, 100.0, 100.0;
			VectorXd kv_gain_orientation = VectorXd::Constant(3, 10.0);
			VectorXd ki_gain_orientation = VectorXd::Zero(3);
			pose_task->setOriControlGains(kp_gain_orientation, kv_gain_orientation, ki_gain_orientation);
			AngleAxisd A_roll  (scrub_orientations[region][0], Vector3d::UnitX());
			AngleAxisd A_pitch (scrub_orientations[region][1], Vector3d::UnitY());
			AngleAxisd A_yaw   (scrub_orientations[region][2], Vector3d::UnitZ());

			Matrix3d sponge_ori = (A_yaw * A_pitch * A_roll).toRotationMatrix();
			pose_task->setGoalOrientation(sponge_ori);

			 N_prec.setIdentity();
			 pose_task->updateTaskModel(N_prec);
			 joint_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());
 
			 // 4) compute torques (now with sat’d vels)
			 command_torques = pose_task->computeTorques() + joint_task->computeTorques();

			 cout << "Current Position: " << (ee_pos - ee_pos_desired).norm() << endl;
 
			 const double thresh = 5e-2;  // 2 cm on XY maybe
			 if ((ee_pos - ee_pos_desired).norm() < thresh) {

				cout << "TRANSITION complete, moving back to CLEAN_1" << endl;

				start_time = -1.0;

				state = INITIAL_APPROACH2;
				ee_pos_current = robot->position(control_link, control_point);

		
				pose_task->reInitializeTask();
				joint_task->reInitializeTask();
			 }
		} else if (state == STOP) {
			// stop the robot			
			pose_task->parametrizeForceMotionSpaces(0); // turn off force control along the Z (and all other axes as well)
		
			N_prec.setIdentity();
			pose_task->updateTaskModel(N_prec);
			joint_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());

			command_torques = pose_task->computeTorques() + joint_task->computeTorques();
		 } else if (state == DONE) {
			// stop the robot			
			pose_task->parametrizeForceMotionSpaces(0); // turn off force control along the Z (and all other axes as well)
			pose_task->setGoalPosition(init_pos);
		
			N_prec.setIdentity();
			pose_task->updateTaskModel(N_prec);
			joint_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());

			command_torques = pose_task->computeTorques() + joint_task->computeTorques();

			const double thresh = 5e-2;  // 2 cm on XY maybe
			 if ((ee_pos - init_pos).norm() < thresh) {

				cout << "DONE" << endl;

				state = STOP;
		
				pose_task->reInitializeTask();
				joint_task->reInitializeTask();
			 }
		 }

		 // execute redis write callback
		 redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, command_torques);

		 controller_counter++;
	 }
 
	 timer.stop();
	 cout << "\nSimulation loop timer stats:\n";
	 timer.printInfoPostRun();
	 redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, 0 * command_torques);  // back to floating
 
	 return 0;
 }