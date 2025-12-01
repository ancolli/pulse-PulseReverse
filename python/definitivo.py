import numpy as np
from scipy.integrate import solve_ivp
import scipy.integrate as spi
import scipy.optimize as opt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Definimos las constantes
c = 0.5
j0 = 10
bTc = -0.05
bTa = 0.017
tfwd = 0.00001 #0.00005 #0.00005 #
trev = 0.00001 #0.0002 #0.00005 #0.00005 #

I0 = -1500 #-825 #-2856 #-272 #
I1 = 0 #1000

ciclos = 1
period = tfwd + trev
tend = ciclos * period

def DutyCycle(tfwd,trev):
    return tfwd / (tfwd + trev)

def I(t):
    cycle_time = t % period
    if t >= ciclos * period:
        return 0.0
    return I0 if cycle_time < tfwd else I1

def nonlinear_term(V):
    return j0 * (np.exp(V / bTa) - np.exp(V / bTc))

def dVdt(t, V):
    return (I(t) - nonlinear_term(V)) / c

    
    # Definir la función a maximizar (en este caso, una función cuadrática negativa)
def funcionInteres(tfwd, trev):
    Ifwd, _ = spi.quad(lambda t: nonlinear_term(V_interp(t)), 0, tfwd)
    Irev, _ = spi.quad(lambda t: nonlinear_term(V_interp(t)), tfwd, tfwd+trev)  
    return (Irev)/(Ifwd )#I - 0.99 * (t_f - tfwd) * DutyCycle(t_f - tfwd) * I0

# Rango de búsqueda para V0
#V0_range = np.linspace(-0.2, -0.2, 10)  # Intenta con 10 valores entre -0.5 y 0.5
V0_guess = -.05
tol = 1e-4
max_iter = 500
min_iter = 1

solution_found = False

#for V0_guess in V0_range:
#    print(f"Probando con V0 inicial = {V0_guess:.4f}")
    
error = 1.0
iteration = 0
    
while (error > tol or iteration < min_iter) and iteration < max_iter:
        # Resolver ODE con el valor actual de V0_guess
        V0 = [V0_guess]
        sol = solve_ivp(dVdt, (0, tend), V0, t_eval=np.linspace(0, tend, 100), atol=1e-8, rtol=1e-8)

        # Obtener V(t) y t
        Pot = sol.y[0]
        t_vals = sol.t

        # Crear interpolador
        V_interp = interp1d(t_vals, Pot, kind='cubic', fill_value="extrapolate")
                

        result = funcionInteres(tfwd, trev)

        # Nuevo valor estimado de V0 basado en V_interp en t_f encontrado
        #V0_new = V_interp(t_f_solution.root)
        V0_new = V_interp(tend)

        # Calcular error
        error = abs(V0_new - V0_guess)

        # Actualizar V0_guess
        V0_guess = V0_new
        
        Vmax = min(Pot)

        iteration += 1
        print(f"Iteración {iteration}: V0 = {V0_guess:.6f}, error = {error:.2e}")


print(f"Convergió en {iteration} iteraciones: delta = {result:.6f}, V0 = {V0_guess:.6f}, Vmax = {Vmax:.6f}, dutyCycle = {DutyCycle(tfwd,trev):.6f}, jav = {DutyCycle(tfwd,trev)*I0:.6f}")

# Guardar datos en un archivo txt con separador de coma y encabezado personalizado
np.savetxt("datos.txt", np.column_stack((t_vals, Pot, nonlinear_term(Pot))), delimiter=",", header="tiempo,Pot,j", comments="")

# Reacondiciono para graficar

# Graficar resultados potencial    
plt.figure()  # Figura 1
plt.plot(t_vals, Pot, label='V(t)', color='black')    
plt.xlabel('Time (s)')
plt.ylabel(r'$\phi_w$ (V)')  
plt.legend()
plt.grid(True)
plt.title('Charge and discharge of the double layer')


# Graficar resultados corriente
plt.figure()  # Figura 1
plt.plot(t_vals, nonlinear_term(Pot), label='j(t)', color='black')
plt.xlabel('Time (s)')
plt.ylabel('j (A/m²)')  
plt.legend()
plt.grid(True)
plt.title('Charge and discharge of the double layer')

"""
t_test = np.linspace(0, tend, 100)
V_test = nonlinear_term(V_interp(t_test))
plt.figure()  # Figura 1
plt.plot(t_test, V_test, label="Interpolación de V")
plt.xlabel("t")
plt.ylabel("V_interp(t)")
plt.legend()
"""

plt.show()

