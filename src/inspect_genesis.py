import genesis as gs
import inspect

print("Genesis location:", gs.__file__)

try:
    # Attempt to find SPHEntity in gs or submodules
    if hasattr(gs, 'SPHEntity'):
        cls = gs.SPHEntity
        print("Found SPHEntity directly in gs")
    else:
        # It seems SPHEntity might be dynamically created or in a submodule
        # Based on liquid_test.py: liquid = scene.add_entity(...)
        # liquid_test.py implies scene.add_entity returns an SPHEntity when material is SPH.Liquid?
        # Let's inspect what scene.add_entity returns given the arguments in liquid_test.py
        pass

    # Let's try to run a minimal setup similar to liquid_test.py up to creation
    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        sph_options=gs.options.SPHOptions(
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1),
            particle_size=0.01,
        )
    )
    liquid = scene.add_entity(
        material=gs.materials.SPH.Liquid(),
        morph=gs.morphs.Box(pos=(0,0,0.65), size=(0.4, 0.4, 0.4)),
        surface=gs.surfaces.Default(vis_mode='particle')
    )
    
    print("Type of liquid:", type(liquid))
    print("Attributes of liquid:", dir(liquid))
    
    if hasattr(liquid, 'get_particles'):
        print("liquid has get_particles")
    else:
        print("liquid DOES NOT have get_particles")
        # Look for similar methods
        state_methods = [m for m in dir(liquid) if 'state' in m or 'particle' in m or 'pos' in m]
        print("Possible relevant methods:", state_methods)

except Exception as e:
    print("Error during inspection:", e)
