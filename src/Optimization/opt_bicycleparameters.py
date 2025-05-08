import bicycleparameters as bp
from bicycleparameters import tables, plot
from bicycleparameters.models import Meijaard2007Model
from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
from uncertainties import nominal_value

import numpy as np
import matplotlib.pyplot as plt


bicycle = bp.Bicycle('Browser', pathToData='C:\THESIS\src\Optimization\data', forceRawCalc=True)

#bicycle.plot_bicycle_geometry(show=True)
speeds = np.linspace(0, 10, 100)
#bicycle.plot_eigenvalues_vs_speed(speeds, show=True, show_legend=True)
print(Meijaard2007ParameterSet.par_strings)

bicycle.add_rider('Jason', reCalc=True)

print(bicycle.parameters)
print(Meijaard2007ParameterSet.par_strings.keys())
cm = 1/2.54


bicycle.plot_bicycle_geometry(show=True)
bicycle.plot_eigenvalues_vs_speed(speeds, show=True, show_legend=False)
plt.axvspan(5.15, 7.7, color='grey', alpha=0.3)  # alpha controls transparency
plt.ylim(-15, 7.5)
plt.xlabel("Velocity [m/s]")
plt.grid()
benchmark_params = {
    key: float(nominal_value(val))
    for key, val in bicycle.parameters['Benchmark'].items()
    if key in Meijaard2007ParameterSet.par_strings  # only include expected params
}
benchmark_params['v'] = 0.0


par_set = Meijaard2007ParameterSet(parameters=benchmark_params, includes_rider=True)
model = Meijaard2007Model(par_set)


#model = Meijaard2007Model
v = np.linspace(0, 10, 100)

# Plot using the provided parameters and axes
model.plot_eigenvalue_parts(v=v,
    colors=['C0', 'C0', 'C2', 'C3'],  # Provide 4 colors for the 4 modes
    show_stable_regions=True,  # Optional, enable stable regions shading
    hide_zeros=False,  # Optional, can hide zeros if needed
    show_legend=True
)
plt.ylim(-10, 7.5)
plt.xlabel("Velocity [m/s]")
plt.show()