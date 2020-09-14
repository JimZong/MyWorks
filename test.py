import argparse
import random
from datetime import datetime
import os
from tqdm import trange
import numpy as np
from PIL import Image
import platform
from subprocess import call
try:
    from manta import *
except ImportError:
    pass

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default='data/smoke_gun')
parser.add_argument("--num_param", type=int, default=3)
parser.add_argument("--path_format", type=str, default='%03d.%s')
parser.add_argument("--npz2vdb_dir", type=str, default='data\\npz2vdb')

parser.add_argument("--src_x_pos", type=float, default=0.5)
parser.add_argument("--src_z_pos", type=float, default=0.5)
parser.add_argument("--src_y_pos", type=float, default=0.2)
parser.add_argument("--src_inflow", type=float, default=5)     #flow
parser.add_argument("--strength", type=float, default=0.05)        #strength of particle spread
parser.add_argument("--src_radius", type=float, default=0.1)    #src radius
parser.add_argument("--num_frames", type=int, default=240)
parser.add_argument("--obstacle", type=bool, default=False)

parser.add_argument("--resolution_x", type=int, default=128)
parser.add_argument("--resolution_y", type=int, default=196)
parser.add_argument("--resolution_z", type=int, default=128)
parser.add_argument("--buoyancy", type=float, default=4e-8)
parser.add_argument("--bWidth", type=int, default=0)
parser.add_argument("--open_bound", type=bool, default=True)
parser.add_argument("--time_step", type=float, default=0.4)
parser.add_argument("--adv_order", type=int, default=2)
parser.add_argument("--clamp_mode", type=int, default=2)

args = parser.parse_args()

def main():
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    field_type = ['d', 'v']
    for field in field_type:
        field_path = os.path.join(args.data_dir,field)
        if not os.path.exists(field_path):
            os.mkdir(field_path)

    args_file = os.path.join(args.data_dir, 'args.txt')
    with open(args_file, 'w') as f:
        print('%s: arguments' % datetime.now())
        for k, v in vars(args).items():
            print('  %s: %s' % (k, v))
            f.write('%s: %s\n' % (k, v))

    res_x = args.resolution_x
    res_y = args.resolution_y
    res_z = args.resolution_z
    d_ = np.zeros([res_z,res_y,res_x], dtype=np.float32)
    v_ = np.zeros([res_z,res_y,res_x,3], dtype=np.float32)
    
    # solver params
    gs = vec3(res_x, res_y, res_z)
    buoyancy = vec3(0,args.buoyancy,0)

    s = Solver(name='main', gridSize=gs, dim=3)
    s.timestep = args.time_step

    flags = s.create(FlagGrid)
    vel = s.create(MACGrid)
    density = s.create(RealGrid)
    pressure = s.create(RealGrid)

    flags.initDomain(boundaryWidth=args.bWidth)
    flags.fillGrid()
    if args.open_bound:
        setOpenBound(flags, args.bWidth,'xXyYzZ', FlagOutflow|FlagEmpty)

    radius = gs.x*args.src_radius
    center = gs*vec3(args.src_x_pos,args.src_y_pos,args.src_z_pos)
    source = s.create(Sphere, center=gs*vec3(args.src_x_pos,args.src_y_pos,args.src_z_pos), radius=radius)
    source1 = s.create(Sphere, center=gs*vec3(args.src_x_pos+0.15,args.src_y_pos,args.src_z_pos), radius=radius*0.4)
    source2 = s.create(Sphere, center=gs*vec3(args.src_x_pos-0.15,args.src_y_pos,args.src_z_pos), radius=radius*0.4)
    source3 = s.create(Sphere, center=gs*vec3(args.src_x_pos,args.src_y_pos,args.src_z_pos+0.15), radius=radius*0.4)
    source4 = s.create(Sphere, center=gs*vec3(args.src_x_pos,args.src_y_pos,args.src_z_pos-0.15), radius=radius*0.4)
    source5 = s.create(Sphere, center=gs*vec3(args.src_x_pos+0.106,args.src_y_pos,args.src_z_pos+0.106), radius=radius*0.4)
    source6 = s.create(Sphere, center=gs*vec3(args.src_x_pos-0.106,args.src_y_pos,args.src_z_pos+0.106), radius=radius*0.4)
    source7 = s.create(Sphere, center=gs*vec3(args.src_x_pos+0.106,args.src_y_pos,args.src_z_pos-0.106), radius=radius*0.4)
    source8 = s.create(Sphere, center=gs*vec3(args.src_x_pos-0.106,args.src_y_pos,args.src_z_pos-0.106), radius=radius*0.4)

    if args.obstacle:
        obs_radius = gs.x*0.15
        obs_center = gs*vec3(0.7, 0.5, 0.5)
        obs = s.create(Sphere, center=obs_center, radius=obs_radius)
        obs.applyToGrid(grid=flags, value=FlagObstacle)
    
    if (GUI):
        gui = Gui()
        gui.show(True)
        #gui.pause()

    for t in trange(args.num_frames, desc='sim'):
        sideStrength=2
        source.applyToGrid(grid=density, value=1.0)
        source.applyToGrid(grid=vel,value=vec3(0,1,0))
        source1.applyToGrid(grid=density, value=1.0)
        #source.applyToGrid(grid=vel,value=vec3(random.uniform(-1,1)*args.src_inflow,args.src_inflow,random.uniform(-1,1)*args.src_inflow))
        source1.applyToGrid(grid=vel,value=sideStrength*vec3(0.1,1,0))
        source2.applyToGrid(grid=density, value=1.0)
        source2.applyToGrid(grid=vel,value=sideStrength*vec3(-0.1,1,0))
        source3.applyToGrid(grid=density, value=1.0)
        source3.applyToGrid(grid=vel,value=sideStrength*vec3(0,1,0.1))
        source4.applyToGrid(grid=density, value=1.0)
        source4.applyToGrid(grid=vel,value=sideStrength*vec3(0,1,-0.1))
        source5.applyToGrid(grid=density, value=1.0)
        source5.applyToGrid(grid=vel,value=sideStrength*vec3(0.07,1,0.07))
        source6.applyToGrid(grid=density, value=1.0)
        source6.applyToGrid(grid=vel,value=sideStrength*vec3(-0.07,1,0.07))
        source7.applyToGrid(grid=density, value=1.0)
        source7.applyToGrid(grid=vel,value=sideStrength*vec3(0.07,1,-0.07))
        source8.applyToGrid(grid=density, value=1.0)
        source8.applyToGrid(grid=vel,value=sideStrength*vec3(-0.07,1,-0.07))

        # save density
        copyGridToArrayReal(density, d_)
        d_file_path = os.path.join(args.data_dir, 'd', args.path_format % (t, 'npz'))
        np.savez_compressed(d_file_path, x=d_)

        if 'Windows' in platform.system():
            manta_path = os.path.join(args.npz2vdb_dir, 'manta.exe')
            py_path = os.path.join(args.npz2vdb_dir, 'npz2vdb.py')
            sh = [manta_path, py_path, '--src_path='+d_file_path]
            call(sh, shell=True)

        d_file_path = os.path.join(args.data_dir, 'd', args.path_format % (t, 'png'))
        im = np.sum(d_, axis=0)
        im = im[::-1]/im.max()*255
        im = Image.fromarray(im.astype(np.uint8))
        im.save(d_file_path)

        # save velocity
        v_file_path = os.path.join(args.data_dir, 'v', args.path_format % (t, 'npz'))
        copyGridToArrayMAC(vel, v_)
        np.savez_compressed(v_file_path, x=v_)

        advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2,
                            openBounds=args.open_bound, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)

        advectSemiLagrange(flags=flags, vel=vel, grid=vel, order=2,
                            openBounds=args.open_bound, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)

        vorticityConfinement(vel=vel, flags=flags, strength=args.strength)

        setWallBcs(flags=flags, vel=vel)
        addBuoyancy(density=density, vel=vel, gravity=buoyancy, flags=flags)
        solvePressure(flags=flags, vel=vel, pressure=pressure, cgMaxIterFac=10.0, cgAccuracy=0.0001)
        setWallBcs(flags=flags, vel=vel)

        s.step()

if __name__ == '__main__':
    main()
