import os
import pickle
import cloudpickle as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from matplotlib import rcParams
from matplotlib.animation import PillowWriter
from sympy import lambdify
from scipy.interpolate import CubicSpline
from src.Optimization.opt_container import DataStorage  # adjust if needed
from src.Optimization.opt_utils import _plot_ground  # adapt imports to your structure
from sympy.physics.mechanics import Point
from symbrim.utilities.plotting import Plotter

import PIL
from matplotlib.animation import FuncAnimation, HTMLWriter, PillowWriter, FFMpegWriter
from scipy.interpolate import CubicSpline
from symmeplot.matplotlib import PlotBody, PlotVector
import sys
sys.path.append("C:/THESIS")
sys.path.append("..")


def bike_following_animation(data, output_name: str, angly=None, elevv=None) -> tuple[
    plt.Figure, plt.Axes, FuncAnimation]:
    # Step 1: Resolve path to data.pkl and solution_data.pkl
    if isinstance(data, str):
        if os.path.isdir(data):
            folder = data
            data_path = None
            for file in os.listdir(folder):
                if file == "data.pkl":
                    data_path = os.path.join(folder, file)
                    break
            if data_path is None:
                raise FileNotFoundError("No data.pkl file found in directory.")
        elif data.endswith(".pkl"):
            data_path = data
            folder = os.path.dirname(data_path)
        else:
            raise ValueError("Provided path must be a folder or a .pkl file.")

        # Load the structure (DataStorage) from data.pkl
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        print("📂 Loaded:", data_path)

        # If wrapped in dict, unwrap
        if isinstance(data, dict):
            for v in data.values():
                if hasattr(v, "solution_state"):
                    data = v
                    break

        # Step 2: If available, load solution_data.pkl and overwrite .solution
        solution_path = os.path.join(folder, "solution_data.pkl")
        if os.path.isfile(solution_path):
            print(f"📥 Loading solution array from: {solution_path}")
            with open(solution_path, "rb") as f:
                solution_array = pickle.load(f)
            if hasattr(data, "solution"):
                data.solution = solution_array
                print("✅ Solution array injected.")
            else:
                print("⚠️ No .solution attribute to overwrite.")

    # Step 3: Check we now have a valid DataStorage object
    if not hasattr(data, "solution_state"):
        raise TypeError(f"❌ Loaded object is type {type(data)}, and does not have `solution_state`.")

    # Set up animation output path
    output_dir = os.path.join("AnimationSims")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name if output_name.endswith(".gif") else f"{output_name}.gif")

    rcParams['animation.convert_path'] = r"C:\ImageMagick-7.1.1-47-Q16-HDRI-x64-dll\magick.exe"
    rcParams['lines.linewidth'] = 4

    C = 2
    rider_angles = [3, 7, 8, 9]
    state_modified = data.solution_state.copy()
    state_modified[rider_angles, :] *= C

    x_eval = CubicSpline(data.time_array, state_modified.T)
    r_eval = CubicSpline(data.time_array, [[cf(t, x) for cf in data.simulator.inputs.values()] for t, x in
                                           zip(data.time_array, data.solution_state.T)])
    p, p_vals = zip(*data.constants.items())

    rear_cp = data.bicycle.rear_tire.contact_point
    front_cp = data.bicycle.front_tire.contact_point
    N = data.system.frame

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 6))
    plotter = Plotter.from_model(data.bicycle_rider, ax=ax)

    queue = [plotter]
    while queue:
        parent = queue.pop()
        if isinstance(parent, PlotBody):
            parent.plot_frame.visible = False
            parent.plot_masscenter.visible = False
        elif isinstance(parent, PlotVector):
            parent.visible = False
        else:
            queue.extend(parent.children)

    plotter.lambdify_system((data.system.q[:] + data.system.u[:], data.simulator.inputs.keys(), p))
    plotter.evaluate_system(x_eval(0), r_eval(0), p_vals)
    plotter.plot()

    ax.axis("off")
    _plot_ground(data, plotter)

    ax.invert_zaxis()
    ax.invert_yaxis()
    ax.set_aspect("equal")

    fps = 50
    track_x, track_y, track_z, FWCP_x, FWCP_y, FWCP_z = [], [], [], [], [], []

    def get_args(ti):
        x_val = x_eval(ti)
        r_val = r_eval(ti)
        x_pos, y_pos, yaw_ang, roll_ang = x_val[0], x_val[1], x_val[2], x_val[3]

        track_x.append(x_pos)
        track_y.append(y_pos)
        track_z.append(0)

        vec_fwcp = front_cp.pos_from(rear_cp)
        x_expr = vec_fwcp.dot(N.x)
        y_expr = vec_fwcp.dot(N.y)
        z_expr = vec_fwcp.dot(N.z)

        subs_dict = dict(zip(data.system.q[:] + data.system.u[:], x_val))
        subs_dict.update(zip(data.simulator.inputs.keys(), r_val))
        subs_dict.update(zip(p, p_vals))

        dx = float(x_expr.subs(subs_dict))
        dy = float(y_expr.subs(subs_dict))
        dz = float(z_expr.subs(subs_dict))

        FWCP_x.append(x_pos + dx)
        FWCP_y.append(y_pos + dy)
        FWCP_z.append(0)

        ax.plot(FWCP_x, FWCP_y, FWCP_z, color='red', linewidth=1, linestyle='dashed')
        ax.plot(track_x, track_y, track_z, color='blue', linewidth=1, linestyle='dashed')
        ax.set_xlim(x_pos - 0.5, x_pos + 0.5)
        ax.set_ylim(y_pos - 1.5, y_pos + 1.5)
        ax.set_aspect("equal")
        azim = np.degrees(yaw_ang)
        roll = np.degrees(roll_ang)
        ax.view_init(elev=elevv, azim=(azim - 180 + angly), roll=0)

        return x_val, r_val, p_vals

    ani = plotter.animate(
        get_args,
        frames=np.arange(0, data.time_array[175], 1 / fps),
        blit=False
    )

    ani.save(output_path, writer="imagemagick", fps=20, dpi=150)
    plt.close()

    return fig, ax, ani


pkl_path = "output_90degTurn/result6_NOCS_triplepen_sidelean_dur5_w1000_dt0.02_no_violations/data.pkl"

with open(pkl_path, "rb") as f:
    obj = cp.load(f)

print("Type of loaded object:", type(obj))

# If it's a dict, list keys
if isinstance(obj, dict):
    print("Top-level keys:", obj.keys())
elif hasattr(obj, "__dict__"):
    print("Attributes:", vars(obj).keys())
else:
    print("Content (sample):", str(obj)[:500])


bike_following_animation("output_90degTurn/result6_NOCS_triplepen_sidelean_dur5_w1000_dt0.02_no_violations", "result6_turn")
