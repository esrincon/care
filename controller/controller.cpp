/**
 * @file controller.cpp
 * @brief Controller file
 * 
 */

 #include <SaiModel.h>
 #include "SaiPrimitives.h"
 #include "redis/RedisClient.h"
 #include "timer/LoopTimer.h"
 #include <Eigen/Geometry>
 
 #include <iostream>
 #include <string>
 
 using namespace std;
 using namespace Eigen;
 using namespace SaiPrimitives;
 
 #include <signal.h>
 bool runloop = false;
 void sighandler(int) {runloop = false;}
 
 #include "redis_keys.h"

 bool simulation = false;

 bool reached_vel = false;
 
 // States 
 enum State {
	 POSTURE = 0, 
	 INITIAL_ROTATION,	
	 INITIAL_APPROACH1, 
	 INITIAL_APPROACH2,
	 CLEAN_1, 
	 RETRACT, 
	 STOP
 };

 int prev_state = -1;

#include <sstream>   // for istringstream

// vector<Vector3d> getScrubPoints(SaiCommon::RedisClient& redis_client,
//                                 const string& key) {
//     vector<Vector3d> points;
//     auto scrub_points_str = redis_client.lrange(key, 0L, -1L);
//     for (auto const& s : scrub_points_str) {
//         istringstream ss(s);
//         double x, y, z;
//         ss >> x >> y >> z;
//         points.emplace_back(x, y, z);
//     }
//     return points;
// }


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
	 const string control_link   = "sponge";
	 const Vector3d control_point = Vector3d(0.0, 0.0, 0.106);
	 //     visual origin (0.1) + half the cylinder length (0.015) = 0.115
	 Affine3d compliant_frame = Affine3d::Identity();
	 compliant_frame.translation() = control_point;
	 auto pose_task = std::make_shared<SaiPrimitives::MotionForceTask>(robot, control_link, compliant_frame);
	 // Will tune depending on the task
	 pose_task->setPosControlGains(100, 20, 0);
	 pose_task->setOriControlGains(100, 20, 0);
 
	 Vector3d ee_pos;
	 Matrix3d ee_ori;
 
	 // joint task
	 auto joint_task = std::make_shared<SaiPrimitives::JointTask>(robot);
	 joint_task->setGains(100, 10, 0);
 
	 VectorXd q_desired(dof);
	 q_desired << 30.0, -5.0, -10.0, -100.0, 0.0, 90.0, 90.0;
	 q_desired *= M_PI / 180.0;
 
	 joint_task->setGoalPosition(q_desired);

	 Vector3d ee_pos_current;

	 Vector3d ee_pos_clean_start;

	 Vector3d ee_pos_desired;

	// get list of scrub points
	// auto scrub_points = getScrubPoints(redis_client, SCRUB_POINTS_KEY);

	// cout << "scrub points:" << endl;
	// for (auto const& point : scrub_points) {
	// 	cout << point.transpose() << endl;
	// }
 
	 // create a loop timer
	 runloop = true;
	 double control_freq = 1000;
	 SaiCommon::LoopTimer timer(control_freq, 1e6);
 
	 // Timer for clean 1
	 double clean1_start_time = -1.0;
	 while (runloop) {
		 timer.waitForNextLoop();
		 const double time = timer.elapsedSimTime();
 
		 // update robot 
		 robot->setQ(redis_client.getEigen(JOINT_ANGLES_KEY));
		 robot->setDq(redis_client.getEigen(JOINT_VELOCITIES_KEY));
		 robot->updateModel();
	 
		 if (state == POSTURE) {
			 // update task model 
			 N_prec.setIdentity();
			 joint_task->updateTaskModel(N_prec);
 
			 command_torques = joint_task->computeTorques();

			 //cout << (robot->q() - q_desired).norm() << endl;
 
			 if ((robot->q() - q_desired).norm() < 8e-2) {
				 cout << "Posture To Motion" << endl;
				 pose_task->reInitializeTask();
				 joint_task->reInitializeTask();
 
				 state = INITIAL_ROTATION;
			 }
 
		 } else if (state == INITIAL_ROTATION) {
			 // update goal position and orientation
 
			 // align sponge cylinder axis (local Z) to world Y (faces XZ plane)
			 Eigen::Matrix3d sponge_ori = Eigen::AngleAxisd(M_PI/2, Vector3d::UnitX()).toRotationMatrix();
			 pose_task->setGoalOrientation(sponge_ori);
			 
 
			 N_prec.setIdentity();
			 // set pose task as priority 1
			 pose_task->updateTaskModel(N_prec);
			 // secondary priority: joint task in nullspace of pose
			 joint_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());
 
			 // set priority 1 as velocity saturation
			 // set priority 2 as posture
 
			 command_torques = pose_task->computeTorques() + joint_task->computeTorques();
			 cout << (ee_ori - sponge_ori).norm() << endl;
 		 // 
			 // check if orientation goal reached
			 ee_ori = robot->rotation(control_link);
			 if ((ee_ori - sponge_ori).norm() < 1e-1) {
				 cout << "Orientation Achieved" << endl;
				 state = INITIAL_APPROACH1;

				 // print out position before approaching
				 ee_pos_current = robot->position(control_link, control_point);
				 cout << "EE position: " << ee_pos_current.transpose() << endl;
				 
				 pose_task->reInitializeTask();
				 joint_task->reInitializeTask();
			 }
		 } else if (state == INITIAL_APPROACH1) {
			 // update goal position and orientation

			 // 1) select a point from list of scrub_pooints 
			//  size_t scrub_index = 2;
			//  if (scrub_points.size() > scrub_index) {
			double offset = 0.20;
			// 	ee_pos_desired = scrub_points[scrub_index];
			// 	ee_pos_desired(1) += offset; // Add offset to y position
			// 	pose_task->setGoalPosition(ee_pos_desired);
			//  } else {
			// 	cout << "[controller] ⚠️ Not enough scrub points available!" << endl;
			//  }
			ee_pos_desired << 0.482223, -0.081536 + offset, 0.328810;
			pose_task->setGoalPosition(ee_pos_desired);
			 // 2) turn on velocity saturation (linear , angular )
			pose_task->enableVelocitySaturation(0.1, 0.5);
 
			 // 3) build your task hierarchy as usual
			 N_prec.setIdentity();
			 pose_task->updateTaskModel(N_prec);
			 joint_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());
 
			 // 4) compute torques (now with sat’d vels)
			 command_torques = pose_task->computeTorques() + joint_task->computeTorques();
 
			 // Print position				
			 ee_pos_current = robot->position(control_link, control_point);
			 cout << "INITIAL_APPROACH1 | Current:  "
				 << ee_pos_current.transpose()
				 << "  Desired:  " << ee_pos_desired.transpose() 
				 << "  Err: " << (ee_pos_current - ee_pos_desired).norm()
				 << endl;
 
			 const double thresh = 8e-2;  // 1 cm on XY maybe?
			 if ((ee_pos_current - ee_pos_desired).norm() < thresh) {
				pose_task -> disableVelocitySaturation();
				pose_task->reInitializeTask();
				joint_task->reInitializeTask();
				prev_state = state;
				state = INITIAL_APPROACH2;
				
			 }

			} else if (state == INITIAL_APPROACH2) {
				double desired_y_vel = -0.05;

				VectorXd kp_gain(3);
			    kp_gain << 100.0, 0.0, 100.0;
				VectorXd kv_gain = VectorXd::Constant(3, 30.0);
			    VectorXd ki_gain = VectorXd::Zero(3);
				pose_task->setPosControlGains(kp_gain, kv_gain, ki_gain);
				
				pose_task->setGoalLinearVelocity(Eigen::Vector3d(0.0, desired_y_vel, 0.0));

				ee_pos_desired << 0.482223, -0.081536 , 0.328810;
				pose_task->setGoalPosition(ee_pos_desired);


				N_prec.setIdentity();
				pose_task->updateTaskModel(N_prec);
				joint_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());

				command_torques = pose_task->computeTorques() + joint_task->computeTorques();

				// Print position				
				Vector3d ee_curr_vel= robot->linearVelocity(control_link, control_point);
				ee_pos_current = robot->position(control_link, control_point);
				cout << "INITIAL_APPROACH2 | Current:  "
				 << ee_pos_current.transpose()  
				 << ee_curr_vel.transpose() << endl;

				if (ee_curr_vel[1] <= desired_y_vel) {
					reached_vel = true;
					cout << "changed" << endl;
				}

				const double thresh = -0.005; 
				if (ee_curr_vel.y() > thresh && reached_vel) {
					cout << "Aligned at center height—starting CLEAN_1\n";
					prev_state = state;
					state = CLEAN_1;
					// reset tasks here
					clean1_start_time = -1.0;
					ee_pos_clean_start= robot->position(control_link, control_point);
					pose_task->reInitializeTask();
					joint_task->reInitializeTask();
				}
 
		 } else if (state == CLEAN_1) {
			 // initialize start time
			 if (clean1_start_time < 0.0) {
				 clean1_start_time = time;
			 }
			 double t_elapsed = time - clean1_start_time;
 
			 // get these points from redis keys
			 const double torso_center_z    = ee_pos_clean_start[2];     // from <origin xyz=…>
			 const double torso_half_height = 0.2 / 2; // box size z = 0.5
 
			 // ■ compute amplitude so sponge never crosses top/bottom
			 double amplitude_z = torso_half_height;
 
			 // ■ pick a frequency (e.g. 0.2 Hz ⇒ one full up+down every 5 s)
			 const double freq1  = 0.01;
			 double omega = 2.0 * M_PI * freq1;
 
			 // ■ desired Z = center + amplitude * sin(ω t)
			 double z_des = torso_center_z + amplitude_z * sin(omega * t_elapsed);
			//  double zdot_des = amplitude_z * cos(omega * t_elapsed) * omega;
			//  double zdotdot_des = -amplitude_z * sin(omega * t_elapsed) * omega * omega;

	 		double y_des = ee_pos_clean_start[1] - 0.01;
			double x_des = ee_pos_clean_start[0];
 
			 // ■ build the full 3D goal (keep your contact Y fixed)
			 // Karim: We were using this ee_cur but don't think we need it. Leads to drift
			 Vector3d ee_cur_pos = robot->position(control_link, control_point);
			 Vector3d ee_cur_vel = robot->linearVelocity(control_link, control_point);
			 Vector3d goal;
			 goal << x_des, y_des, z_des;

			 VectorXd kp_gain_orientation(3);
			 kp_gain_orientation << 10.0, 100.0, 10.0;
			 VectorXd kv_gain_orientation = VectorXd::Constant(3, 10.0);
			 VectorXd ki_gain_orientation = VectorXd::Zero(3);
			 pose_task->setOriControlGains(kp_gain_orientation, kv_gain_orientation, ki_gain_orientation);

			 VectorXd kp_gain(3);
			 kp_gain << 100.0, 0.0, 100.0;
			 VectorXd kv_gain = VectorXd::Constant(3, 20.0);
			 VectorXd ki_gain = VectorXd::Zero(3);
			 pose_task->setPosControlGains(kp_gain, kv_gain, ki_gain);
			 double des_y_vel = -0.025;
			 pose_task->setGoalLinearVelocity(Eigen::Vector3d(0.0, des_y_vel, 0.0));
			 pose_task->setGoalPosition(goal);
			 //pose_task->setGoalLinearVelocity(Eigen::Vector3d(ee_cur_vel.x(), ee_cur_vel.y(), zdot_des));
			 //pose_task->setGoalLinearAcceleration(Vector3d(0.0, 0.0, zdotdot_des));
 
			 // ■ task hierarchy + torque
			 N_prec.setIdentity();
			 pose_task->updateTaskModel(N_prec);
			 joint_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());
			 command_torques = pose_task->computeTorques()
							 + joint_task->computeTorques();
 
			 // ■ debug print
			 cout << "CLEAN_1 | z_cur=" << ee_cur_pos.z()
				 << " z_des=" << z_des
				 << " err=" << fabs(ee_cur_pos.z()-z_des) << endl;
 
			 // ■ after 15 s, go to RETRACT
			 if (t_elapsed > 20.0) {
				 cout << "CLEAN_1 complete, switching to RETRACT\n";
				 state = STOP;
				 pose_task->reInitializeTask();
				 joint_task->reInitializeTask();
			 }
 
		 } else if (state == RETRACT) {
			 // 1) set your desired goal
			 Vector3d ee_cur_pos = robot->position(control_link, control_point);
			 Vector3d ee_pos_desired;
			 ee_pos_desired << ee_cur_pos.x(), 0.125, ee_cur_pos.z();
			 pose_task->setGoalPosition(ee_pos_desired);
 
			 // 2) turn on velocity saturation (linear , angular )
			 pose_task->enableVelocitySaturation(0.1, 0.5);
 
			 // 3) build your task hierarchy as usual
			 N_prec.setIdentity();
			 pose_task->updateTaskModel(N_prec);
			 joint_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());
 
			 // 4) compute torques (now with sat’d vels)
			 command_torques = pose_task->computeTorques() + joint_task->computeTorques();
 
			 // Print position				
			 Vector3d ee_pos_current = robot->position(control_link, control_point);
			 cout << "INITIAL_APPROACH1 | Current:  "
				 << ee_pos_current.transpose()
				 << "  Desired:  " << ee_pos_desired.transpose() 
				 << "  Err: " << (ee_pos_current - ee_pos_desired).norm()
				 << endl;
 
			 const double thresh = 1e-2;  // 1 cm on XY maybe?
			 if ((ee_pos_current - ee_pos_desired).norm() < thresh) {
				pose_task -> disableVelocitySaturation();
				pose_task->reInitializeTask();
				joint_task->reInitializeTask();
				prev_state = state;
				state = STOP;
				
			 }
		 }

		 else if (state == STOP) {
			// stop the robot
			N_prec.setIdentity();
			pose_task->updateTaskModel(N_prec);
			joint_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());

			command_torques = pose_task->computeTorques() + joint_task->computeTorques();
		 }
 
		 // execute redis write callback
		 redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, command_torques);
	 }
 
	 timer.stop();
	 cout << "\nSimulation loop timer stats:\n";
	 timer.printInfoPostRun();
	 redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, 0 * command_torques);  // back to floating
 
	 return 0;
 }