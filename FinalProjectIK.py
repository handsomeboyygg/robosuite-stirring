import robosuite as suite
from robosuite.controllers import load_controller_config
import numpy as np
import time
import matplotlib.pyplot as plt

radius = 0.1
def rot(point):
    theta = np.radians(-90)    
    transformation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], 
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    rot_pos = transformation_matrix @ point
    rot_pos[2] = 0
    return rot_pos


if __name__ == "__main__":
    controller_config = load_controller_config(default_controller="IK_POSE")
    env = suite.make(
        env_name="Lift", 
        robots="Panda", 
        controller_configs=controller_config,
        has_renderer=True,  
        render_camera="frontview",
    )
    obs = env.reset()
    ##################################################################################################
    robot_name = env.robots[0].name
    mujoco_model = env.sim.model

    eef_name = 'gripper0_grip_site'
    ee_link_name = "gripper0_eef"

    ee_link_id = mujoco_model.body_name2id(ee_link_name)

    obs = env.reset()
    target_position = [0, 0, 0]
    target_orientation = [0.0, 0.0, 0.0, 1.0]
    addfor = 0
    action = np.concatenate([target_position, target_orientation])
    for _ in range(10):
        env.step(action)
        env.render()
        joint_torques = env.sim.data.actuator_force[ :7]
        jacp, jacr = env.sim.data.get_body_jacp(ee_link_name)[:, :7], env.sim.data.get_body_jacr(ee_link_name)[:, :7]
        jac = np.vstack((jacp, jacr))
        jac = np.linalg.pinv(jac.T)
        force = jac @ joint_torques
        print("force", force[ :3])
        addfor = force[ :3]

    print(addfor)

    target_position = [0, 0, 0]
    target_orientation = [0.0, 0.0, 0.0, 1.0]
    action = np.concatenate([target_position, target_orientation])
    for _ in range(30):
        env.step(action)
        env.render()
        joint_torques = env.sim.data.actuator_force[ :7]
        jacp, jacr = env.sim.data.get_body_jacp(ee_link_name)[:, :7], env.sim.data.get_body_jacr(ee_link_name)[:, :7]
        jac = np.vstack((jacp, jacr))
        force = jac @ joint_torques
        jac = np.linalg.pinv(jac.T)
        force = jac @ joint_torques
        print("force", force[ :3] - addfor)



    eef_pos = obs['robot0_eef_pos']
    center = np.copy(eef_pos) + np.array([0.1, 0, 0])
    kavg = []
    print(eef_pos)
    print(center)
    for i in range(800):
        tar = obs['robot0_eef_pos'] - center
        tar_ori = rot(tar)
        action = np.concatenate([0.05*tar_ori, target_orientation])
        obs, reward, done, info = env.step(action)
        env.render()

        joint_torques = env.sim.data.actuator_force[ :7]
        jacp, jacr = env.sim.data.get_body_jacp(ee_link_name)[:, :7], env.sim.data.get_body_jacr(ee_link_name)[:, :7]
        print(jacp)
        jac = np.vstack((jacp, jacr))
        force = jac @ joint_torques
        jac = np.linalg.pinv(jac.T)
        force = jac @ joint_torques
        force = force[ :3] - addfor
        eef_vel = env.sim.data.get_site_xvelp(eef_name)
        #print(np.array2string(eef_vel, formatter={'float_kind':lambda x: f"{x:.6f}"}))
        
        nor_vec = (obs['robot0_eef_pos'] - center)[:2] / np.linalg.norm((obs['robot0_eef_pos'] - center)[:2])
        force_nor = np.dot(force[:2], nor_vec)
        force_tan = np.sqrt(np.linalg.norm(force)**2 - force_nor**2)

        # eef_vel_vec = eef_vel[:2] / np.linalg.norm(eef_vel[:2])
        # force_tan = np.dot(force[:2], eef_vel_vec)
        # force_nor = np.sqrt(np.linalg.norm(force)**2 - force_tan**2)
        k = abs(np.linalg.norm(force_tan)) / abs(np.linalg.norm(eef_vel[:2]))
        if(i >= 200):
            kavg.append(k)
        print("force", force)
        print("normal_force", force_tan)
        print("tangential_force", force_nor)
        print("k", k)

    
    time_points = np.arange(0, 600, 1)
    plt.plot(time_points, kavg, marker='o', linestyle='-', color='b')
    plt.title('Drag Coefficient over Time')
    plt.xlabel('Time')
    plt.ylabel('Drag Coefficient')
    plt.grid(True)
    plt.show()
    kavg = np.mean(kavg)
    print("kavg = ", kavg)

    env.close()