import genesis as gs
import numpy as np
import os
########################## init ##########################
gs.init(
    seed                = None,
    precision           = '32',
    debug               = False,
    eps                 = 1e-12,
    logging_level       = None,
    backend             = gs.gpu,
    theme               = 'dark',
    logger_verbose_time = False
)


########################## create a scene ##########################

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt       = 1e-3,
        substeps = 10,
    ),
    sph_options=gs.options.SPHOptions(
        lower_bound   = (-1.0, -1.0, 0.0),
        upper_bound   = (1.0, 1.0, 1),
        particle_size = 0.01,
    ),
    vis_options=gs.options.VisOptions(
        visualize_sph_boundary = True,
    ),
    show_viewer = True,
)

########################## entities ##########################
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
)

liquid = scene.add_entity(
    # viscous liquid
    # material=gs.materials.SPH.Liquid(mu=0.02, gamma=0.02),
    material=gs.materials.SPH.Liquid(),
    morph=gs.morphs.Cylinder(
        pos  = (0.55, 0.0, 0.34),
        height = 0.3,
        radius = 0.08,
    ),
    surface=gs.surfaces.Default(
        color    = (0.4, 0.8, 1.0),
    ),
)

cup = scene.add_entity(
    material=gs.materials.Rigid(rho=1000,
    sdf_cell_size=0.001),
    morph=gs.morphs.Mesh(file='/home/irol/workspace/genesis_ai/src/asset/Cup.obj',
    scale=0.002,
    pos=(0.45, 0.1, 0.05),
    euler=(90.0, 0.0, 0.0),
    convexify=False,),
)

franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

#############,
# decimate_face_num = 10,############# build ##########################
scene.build()


motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

# set control gains
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

end_effector = franka.get_link("hand")


# move to pre-grasp pose
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.55, 0.2, 0.2]),
    quat=np.array([0, 0.5, -0.5, 0]),
)
# gripper open pos
qpos[-2:] = 0.04
path = franka.plan_path(
    qpos_goal=qpos,
    num_waypoints=200 if "PYTEST_VERSION" not in os.environ else 10,  # 2s duration
)
# draw the planned path
path_debug = scene.draw_debug_path(path, franka)

# execute the planned path
for waypoint in path:
    franka.control_dofs_position(waypoint)
    scene.step()

# remove the drawn path
scene.clear_debug_object(path_debug)

# allow robot to reach the last waypoint
for i in range(100 if "PYTEST_VERSION" not in os.environ else 1):
    scene.step()

# reach
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.55, -0.2, 0.2]),
    quat=np.array([0, 0.5, 0.5, 0]),
)

franka.control_dofs_position(qpos[:-2], motors_dof)
for i in range(1000 if "PYTEST_VERSION" not in os.environ else 1):
    scene.step()



# get particle positions
# particles = liquid.get_state()
