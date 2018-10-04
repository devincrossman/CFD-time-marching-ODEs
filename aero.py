"""This program solves the one-dimensional first order wave equation
with periodic boundary conditions on the domain 0 <= x <= 1"""
import tkinter as tk
from tkinter import messagebox
import numpy as np
import numexpr as ne
import matplotlib as mpl
#pylint: disable=wrong-import-position
mpl.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#pylint: enable=wrong-import-position

class App:
    """A tkinter app to demonstrate different methods of solving the
    first order wave equation (transport equation) with periodic
    boundary conditions.
    """
    def __init__(self):
        """Initialize the application."""

        # spatial domain
        self.xmin = 0
        self.xmax = 1

        # create the interface
        self.create_interface()

        # create the figure for the plots
        self.create_figure()

        # center the window on the screen
        self.center_window()

        # start the app
        self.root.mainloop()

    def create_interface(self):
        """Create the interface."""
        self.root = tk.Tk()
        self.root.title("AERO 455")

        time_stepping_options = [
            "Explicit Euler",
            "Implicit Euler",
            "Trapezoidal",
            "Adams-Bashforth 2",
            "Runge-Kutta 4"
        ]

        time_steppping_label = tk.Label(self.root,
                                        text="Time-marching method:")
        time_steppping_label.grid(row=0, column=0, sticky=tk.W)
        self.time_stepping = tk.StringVar(self.root)
        self.time_stepping.set(time_stepping_options[0])

        time_stepping_widget = tk.OptionMenu(
            self.root, self.time_stepping, *time_stepping_options)
        time_stepping_widget.grid(row=0, column=1, sticky=tk.W)

        spatial_derivative_options = [
            "2nd order",
            "4th order"
        ]
        spatial_derivative_label = tk.Label(
            self.root, text="Spatial derivative accuracy:")
        spatial_derivative_label.grid(row=1, column=0, sticky=tk.W)
        self.spatial_derivative_order = tk.StringVar(self.root)
        self.spatial_derivative_order.set(spatial_derivative_options[0])

        self.spatial_derivative_order_widget = tk.OptionMenu(
            self.root, self.spatial_derivative_order,
            *spatial_derivative_options)
        self.spatial_derivative_order_widget.grid(row=1, column=1,
                                                  sticky=tk.W)

        self.animate_btn = tk.Button(self.root, text="Animate",
                                     fg="blue", command=self.animate)
        self.animate_btn.grid(row=1, column=2, sticky=tk.W)

        quit_btn = tk.Button(self.root, text="QUIT",
                             fg="red", command=self.root.quit)
        quit_btn.grid(row=1, column=3, sticky=tk.W)

        init_condition_label_widget = tk.Label(
            self.root,
            text="Initial Condition:")
        init_condition_label_widget.grid(row=3, column=0, sticky=tk.W)

        self.init_condition_input = tk.Entry(
            self.root, validate="focusout", width=20,
            vcmd=(
                self.root.register(self.validate_init_condition),
                '%P'))
        self.init_condition_input.insert(0,
                                         "exp(-0.5*((x-0.5)/0.08)**2)")
        self.init_condition_input.grid(row=3, column=1, sticky=tk.W)

        wave_speed_label_widget = tk.Label(self.root,
                                           text="Wave speed:")
        wave_speed_label_widget.grid(row=4, column=0, sticky=tk.W)

        self.wave_speed_input = tk.Entry(
            self.root, validate="all", width=3,
            vcmd=(self.root.register(self.validate_wave_speed),
                  '%V', '%P'))
        self.wave_speed_input.insert(0, "1")
        self.wave_speed_input.grid(row=4, column=1, sticky=tk.W)

        cfl_label_widget = tk.Label(self.root, text="CFL number:")
        cfl_label_widget.grid(row=5, column=0, sticky=tk.W)

        self.cfl_input = tk.Entry(
            self.root, validate="all", width=3,
            vcmd=(self.root.register(self.validate_cfl), '%V', '%P'))
        self.cfl_input.insert(0, "0.1")
        self.cfl_input.grid(row=5, column=1, sticky=tk.W)

        grid_size_label_widget = tk.Label(
            self.root,
            text="Number of spatial nodes:")
        grid_size_label_widget.grid(row=6, column=0, sticky=tk.W)

        self.grid_size_input = tk.Entry(
            self.root, validate="all", width=3,
            vcmd=(self.root.register(self.validate_grid_size),
                  '%V', '%P'))
        self.grid_size_input.insert(0, "50")
        self.grid_size_input.grid(row=6, column=1, sticky=tk.W)

        timesteps_label_widget = tk.Label(self.root,
                                          text="Number of time steps:")
        timesteps_label_widget.grid(row=7, column=0, sticky=tk.W)

        self.timesteps_input = tk.Entry(
            self.root, validate="all", width=4,
            vcmd=(self.root.register(self.validate_timesteps),
                  '%V', '%P'))
        self.timesteps_input.insert(0, "490")
        self.timesteps_input.grid(row=7, column=1, sticky=tk.W)

        self.info_label = tk.StringVar(self.root)
        self.info_label.set("")

        info_label_widget = tk.Label(self.root,
                                     textvariable=self.info_label)
        info_label_widget.grid(row=8, columnspan=4)

        self.minmax_label = tk.StringVar(self.root)
        self.minmax_label.set("\n")

        minmax_label_widget = tk.Label(self.root,
                                       textvariable=self.minmax_label)
        minmax_label_widget.grid(row=9, columnspan=4)

        self.time_label = tk.StringVar(self.root)
        self.time_label.set("t = 0.00s")

        time_label_widget = tk.Label(self.root,
                                     textvariable=self.time_label)
        time_label_widget.grid(row=10, columnspan=4)

        self.error_label = tk.StringVar(self.root)
        self.error_label.set("error = 0")

        error_label_widget = tk.Label(self.root,
                                      textvariable=self.error_label)
        error_label_widget.grid(row=11, columnspan=4)

    def center_window(self):
        """Center the window on the screen."""
        self.root.withdraw()
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = self.root.winfo_reqwidth()
        win_h = self.root.winfo_reqheight()
        x = screen_w/2 - win_w/2
        y = screen_h/2 - win_h/2
        self.root.geometry("%dx%d+%d+%d" % (win_w, win_h, x, y))
        self.root.deiconify()

    def create_figure(self):
        """Create the figure for plots."""
        canvas_w = 600
        canvas_h = 350
        #plt.style.use('dark_background')
        mpl.rcParams.update({'figure.autolayout': True})
        self.fig = plt.figure(figsize=(1, 1))
        fig_w = canvas_w/self.fig.get_dpi()
        fig_h = canvas_h/self.fig.get_dpi()
        self.fig.set_size_inches(fig_w, fig_h)
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True)
        self.ax.set_xlim([self.xmin, self.xmax])
        self.ax.set_ylim([-0.5, 1.5])
        self.ax.legend(loc='upper right')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('u')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(columnspan=4)

    def validate_grid_size(self, operation, value):
        """Validate the grid size input
        must be an integer between 6 and 999.
        """
        if operation != 'focusout':
            if value == '':
                return True
        if len(value) > 3:
            return False
        if ' ' in value:
            return False
        try:
            int(value)
            if operation == 'focusout':
                if int(value) >= 6 and int(value) <= 999:
                    self.grid_size_input.configure(background='white')
                    return True
                raise ValueError
            return True
        except ValueError:
            messagebox.showerror(
                "Error",
                "The value for grid size must be an integer between "
                "6 and 999")
            self.grid_size_input.configure(background='#FFdddd')
            return False

    def validate_timesteps(self, operation, value):
        """Validate the grid size input
        must be an integer between 1 and 5000.
        """
        if operation != 'focusout':
            if value == '':
                return True
        if len(value) > 4:
            return False
        if ' ' in value:
            return False
        try:
            int(value)
            if operation == 'focusout':
                if int(value) >= 1 and int(value) <= 5000:
                    self.timesteps_input.configure(background='white')
                    return True
                raise ValueError
            return True
        except ValueError:
            messagebox.showerror(
                "Error",
                "The value for time steps must be an integer between "
                "1 and 5000")
            self.timesteps_input.configure(background='#FFdddd')
            return False

    def validate_wave_speed(self, operation, value):
        """Validate the wave speed input
        must be a floating point number between 0 and 10.
        """
        if operation != 'focusout':
            if value == '' or value == '-' or value == '.':
                return True
        if ' ' in value:
            return False
        try:
            float(value)
            if operation == 'focusout':
                if float(value) >= -10 and float(value) <= 10:
                    self.wave_speed_input.configure(background='white')
                    return True
                raise ValueError
            return True
        except ValueError:
            messagebox.showerror(
                "Error",
                "The value for wave speed must be a floating point "
                "number between -10 and 10")
            self.wave_speed_input.configure(background='#FFdddd')
            return False

    def validate_cfl(self, operation, value):
        """Validate the cfl input
        must be a floating point number between 0 and 10.
        """
        if operation != 'focusout':
            if value == '' or value == '.':
                return True
        if ' ' in value:
            return False
        try:
            float(value)
            if operation == 'focusout':
                if float(value) > 0 and float(value) <= 10:
                    self.cfl_input.configure(background='white')
                    return True
                raise ValueError
            return True
        except ValueError:
            messagebox.showerror(
                "Error",
                "The value for CFL must be a floating point number "
                "between 0 and 10")
            self.cfl_input.configure(background='#FFdddd')
            return False

    def validate_init_condition(self, value):
        """Validate the initial condition
        must be a function of x only.
        """
        try:
            x = np.linspace(self.xmin, self.xmax)
            ne.evaluate(value, {'x':x, 'pi':np.pi})
            self.init_condition_input.configure(background='white')
            return True
        except Exception:
            messagebox.showerror(
                "Error",
                "The value for the initial condition must be an "
                "evaluatable function of x")
            self.init_condition_input.configure(background='#FFdddd')
            return False

    def create_grid(self):
        """Create the solution grid based on user inputs and solve
        the exact equation.
        """
        self.n = int(self.grid_size_input.get())
        self.c = float(self.wave_speed_input.get())
        self.CFL = float(self.cfl_input.get())
        self.X, self.dx = np.linspace(self.xmin, self.xmax,
                                      self.n, retstep=True)
        # determine dt based on CFL
        self.dt = self.CFL * self.dx / np.abs(self.c)
        self.info_label.set(f"c = {self.c}, dx = {self.dx}, "
                            f"dt = {self.dt}, CFL = {self.CFL}")
        # solve U_exact for tmax time steps
        # using mod to make the function periodic
        self.U_exact = []
        for t in range(self.tmax + 1):
            self.U_exact.append(self.initial_u(
                np.mod(self.X - self.c * t * self.dt,
                       self.xmax - self.xmin)))

    def clear_plot(self):
        """Clear the previous solutions from the figure."""
        if hasattr(self, 'exact_line'):
            self.exact_line.remove()
        if hasattr(self, 'approximation_line'):
            self.approximation_line.remove()

    def current_timestep(self):
        """Return the current timestep for the animation."""
        self.timestep = 0
        while self.timestep < self.tmax:
            self.timestep += 1
            yield self.timestep

    def inputs_are_valid(self):
        """Check if user inputs are valid."""
        return (self.validate_init_condition(
            self.init_condition_input.get())
                and self.validate_cfl('focusout', self.cfl_input.get())
                and self.validate_wave_speed('focusout',
                                             self.wave_speed_input.get())
                and self.validate_timesteps('focusout',
                                            self.timesteps_input.get())
                and self.validate_grid_size('focusout',
                                            self.grid_size_input.get()))

    def animate(self):
        """Start the animation."""
        # ensure user input validation is performed
        if self.inputs_are_valid():
            self.solve_approximation()
            self.clear_plot()
            self.exact_line, = self.ax.plot(self.X, self.U_exact[0],
                                            color="grey",
                                            label='Exact')
            self.approximation_line, = self.ax.plot(
                self.X, self.U_exact[0], color="black",
                label='Approximation', linestyle='dashed')
            # stop previous animation by setting the timestep > tmax
            self.timestep = self.tmax + 1
            self.anim = animation.FuncAnimation(
                self.fig, self.do_animation,
                frames=self.current_timestep,
                repeat=False, interval=self.dt*1000)
            self.fig.canvas.draw()

    def do_animation(self, i):
        """Update the animation frame."""
        self.time_label.set(f"t = {i} * {self.dt}s = {i*self.dt:.2f}s")
        self.exact_line.set_ydata(self.U_exact[i])
        self.approximation_line.set_ydata(self.U[i])
        error = np.sqrt(np.sum(np.power(self.U_exact[i]-self.U[i], 2)))
        self.error_label.set(f'error = {error}')
        minmax = ('Exact min: ' + str(self.U_exact[i].min())
                  + ' Exact max: ' + str(self.U_exact[i].max())
                  + '\nNumerical min: ' + str(self.U[i].min())
                  + ' Numerical max: ' + str(self.U[i].max()))
        self.minmax_label.set(minmax)

    def solve_approximation(self):
        """Solve the differential equation
        using thecurrent settings."""
        selected_method = self.time_stepping.get()
        if selected_method == 'Explicit Euler':
            solution_method = self.euler_explicit
        elif selected_method == 'Implicit Euler':
            solution_method = self.euler_implicit
        elif selected_method == 'Trapezoidal':
            solution_method = self.trapezoidal
        elif selected_method == 'Adams-Bashforth 2':
            solution_method = self.ab2
        elif selected_method == 'Runge-Kutta 4':
            solution_method = self.rk4
        self.U = [] # reset the value of U
        self.tmax = int(self.timesteps_input.get())
        # create grid and solve exact solution
        self.create_grid()
        # solve approximation using solution method
        for t in range(self.tmax + 1):
            self.U.append(solution_method(t))

    def initial_u(self, _x):
        """Return the initial values for U."""
        return ne.evaluate(self.init_condition_input.get(),
                           {'x':_x, 'pi':np.pi})

    def euler_explicit(self, t):
        """Return the solution for the t-th time step
        using the euler explicit method.
        """
        U, c, dt = self.U, self.c, self.dt
        if t == 0: # initial condition
            return self.initial_u(self.X)
        return U[t-1] - c*dt*self.central_ux(t-1)

    def euler_implicit(self, t):
        """Return the solution for the t-th time step
        using the euler implicit method.
        """
        U, c, dt, dx, n = self.U, self.c, self.dt, self.dx, self.n
        if t == 0: # initial condition
            return self.initial_u(self.X)
        a = np.zeros((n, n))
        # set the corners of the matrix
        a[0][n-1] = -c*dt/(2*dx)
        a[n-1][0] = c*dt/(2*dx)
        for i in range(n):
            for j in range(n):
                if i == j:
                    a[i][j] = 1
                elif i == j-1:
                    a[i][j] = c*dt/(2*dx)
                elif i == j+1:
                    a[i][j] = -c*dt/(2*dx)
        return np.linalg.solve(a, U[t-1])

    def ab2(self, t):
        """Return the solution for the t-th time step using the
        Adams-Bashforth 2nd order method.
        """
        U, c, dt = self.U, self.c, self.dt
        if t == 0: # initial condition
            return self.initial_u(self.X)
        if t == 1: # first step with RK4 to start
            return self.rk4(t)
        return (U[t-1] - c*dt*((3/2)*self.central_ux(t-1)
                               - (1/2)*self.central_ux(t-2)))

    def trapezoidal(self, t):
        """Return the solution for the t-th time step using the
        Adams-Moulton 2nd order (trapezoidal) method.
        """
        U, c, dt, dx, n = self.U, self.c, self.dt, self.dx, self.n
        if t == 0: # initial condition
            return self.initial_u(self.X)
        a = np.zeros((n, n))
        b = np.zeros(n)
        # set the corners of the matrix
        a[0][n-1] = -c*dt/(4*dx)
        a[n-1][0] = c*dt/(4*dx)
        for i in range(n):
            for j in range(n):
                # set diagonals
                if i == j:
                    a[i][j] = 1
                elif i == j-1:
                    a[i][j] = c*dt/(4*dx)
                elif i == j+1:
                    a[i][j] = -c*dt/(4*dx)
        # set solution vector
        b = U[t-1]-c*dt/2*self.central_ux(t-1)
        return np.linalg.solve(a, b)

    def rk4(self, t):
        """Return the solution for the t-th time step using the
        classical 4th order Runge-Kutta method.
        """
        U, c, dt = self.U, self.c, self.dt
        if t == 0: # initial condition
            return self.initial_u(self.X)
        ux = self.central_ux(t-1)
        uxx = self.central_uxx(t-1)
        uxxx = self.central_uxxx(t-1)
        uxxxx = self.central_uxxxx(t-1)
        Rn = -c*ux
        R1 = -c*ux + (c**2)*dt/2*uxx
        R2 = (-c*ux + (c**2)*dt/2*uxx
              - (c**3)*(dt**2)/4*uxxx)
        R3 = (-c*ux + (c**2)*dt*uxx
              - (c**3)*(dt**2)/2*uxxx
              + (c**4)*(dt**3)/4*uxxxx)
        return U[t-1] + dt/6*(Rn+2*R1+2*R2+R3)

    def central_ux(self, t):
        """Return the central difference approximation of the first
        derivative of u with respect to x at time step t.
        """
        if self.spatial_derivative_order.get() == '2nd order':
            return self.central_ux_2(t)
        return self.central_ux_4(t)

    def central_uxx(self, t):
        """Return the central difference approximation of the second
        derivative of u with respect to x at time step t.
        """
        if self.spatial_derivative_order.get() == '2nd order':
            return self.central_uxx_2(t)
        return self.central_uxx_4(t)

    def central_uxxx(self, t):
        """Return the central difference approximation of the third
        derivative of u with respect to x at time step t.
        """
        if self.spatial_derivative_order.get() == '2nd order':
            return self.central_uxxx_2(t)
        return self.central_uxxx_4(t)

    def central_uxxxx(self, t):
        """Return the central difference approximation of the fourth
        derivative of u with respect to x at time step t.
        """
        if self.spatial_derivative_order.get() == '2nd order':
            return self.central_uxxxx_2(t)
        return self.central_uxxxx_4(t)

    def central_ux_2(self, t):
        """Return the 2nd order central difference approximation
        of the first derivative of U with respect to x at time step t.
        """
        U, X, dx, n = self.U, self.X, self.dx, self.n
        val = [] # values for the approximation at the t-th time step
        for j in range(len(X)):
            if j == 0: # left boundary
                val.append((U[t][j+1]-U[t][n-1])/(2*dx))
            elif j == n-1: # right boundary
                val.append((U[t][0]-U[t][j-1])/(2*dx))
            else:
                val.append((U[t][j+1]-U[t][j-1])/(2*dx))
        return np.array(val)

    def central_ux_4(self, t):
        """Return the 4th order central difference approximation
        of the first derivative of U with respect to x at time step t.
        """
        U, X, dx, n = self.U, self.X, self.dx, self.n
        val = [] # values for the approximation at the t-th time step
        for j in range(len(X)):
            if j == 0: # left boundary
                val.append((U[t][n-2]-8*U[t][n-1]+8*U[t][j+1]
                            -U[t][j+2])/(12*dx))
            elif j == 1:
                val.append((U[t][n-1]-8*U[t][j-1]+8*U[t][j+1]
                            -U[t][j+2])/(12*dx))
            elif j == n-2:
                val.append((U[t][j-2]-8*U[t][j-1]+8*U[t][j+1]
                            -U[t][0])/(12*dx))
            elif j == n-1: # right boundary
                val.append((U[t][j-2]-8*U[t][j-1]+8*U[t][0]
                            -U[t][1])/(12*dx))
            else:
                val.append((U[t][j-2]-8*U[t][j-1]+8*U[t][j+1]
                            -U[t][j+2])/(12*dx))
        return np.array(val)

    def central_uxx_2(self, t):
        """Return the 2nd order central difference approximation
        of the second derivative of U with respect to x at time step t.
        """
        U, X, dx, n = self.U, self.X, self.dx, self.n
        val = [] # values for the approximation at the t-th time step
        for j in range(len(X)):
            if j == 0: # left boundary
                val.append((U[t][j+1]-2*U[t][j]+U[t][n-1])/(dx**2))
            elif j == n-1: # right boundary
                val.append((U[t][0]-2*U[t][j]+U[t][j-1])/(dx**2))
            else:
                val.append((U[t][j+1]-2*U[t][j]+U[t][j-1])/(dx**2))
        return np.array(val)

    def central_uxx_4(self, t):
        """Return the 4th order central difference approximation
        of the second derivative of U with respect to x at time step t.
        """
        U, X, dx, n = self.U, self.X, self.dx, self.n
        val = [] # values for the approximation at the t-th time step
        for j in range(len(X)):
            if j == 0: # left boundary
                val.append((-U[t][n-2]+16*U[t][n-1]-30*U[t][j]
                            +16*U[t][j+1]-U[t][j+2])/(12*dx**2))
            elif j == 1:
                val.append((-U[t][n-1]+16*U[t][j-1]-30*U[t][j]
                            +16*U[t][j+1]-U[t][j+2])/(12*dx**2))
            elif j == n-2:
                val.append((-U[t][j-2]+16*U[t][j-1]-30*U[t][j]
                            +16*U[t][j+1]-U[t][0])/(12*dx**2))
            elif j == n-1: # right boundary
                val.append((-U[t][j-2]+16*U[t][j-1]-30*U[t][j]
                            +16*U[t][0]-U[t][1])/(12*dx**2))
            else:
                val.append((-U[t][j-2]+16*U[t][j-1]-30*U[t][j]
                            +16*U[t][j+1]-U[t][j+2])/(12*dx**2))
        return np.array(val)

    def central_uxxx_2(self, t):
        """Return the 2nd order central difference approximation
        of the third derivative of U with respect to x at time step t.
        """
        U, X, dx, n = self.U, self.X, self.dx, self.n
        val = [] # values for the approximation at the t-th time step
        for j in range(len(X)):
            if j == 0: # left boundary
                val.append((-U[t][n-2]+2*U[t][n-1]-2*U[t][j+1]
                            +U[t][j+2])/(2*dx**3))
            elif j == 1:
                val.append((-U[t][n-1]+2*U[t][j-1]-2*U[t][j+1]
                            +U[t][j+2])/(2*dx**3))
            elif j == n-2:
                val.append((-U[t][j-2]+2*U[t][j-1]-2*U[t][j+1]
                            +U[t][0])/(2*dx**3))
            elif j == n-1: # right boundary
                val.append((-U[t][j-2]+2*U[t][j-1]-2*U[t][0]
                            +U[t][1])/(2*dx**3))
            else:
                val.append((-U[t][j-2]+2*U[t][j-1]-2*U[t][j+1]
                            +U[t][j+2])/(2*dx**3))
        return np.array(val)

    def central_uxxx_4(self, t):
        """Return the 4th order central difference approximation
        of the third derivative of U with respect to x at time step t.
        """
        U, X, dx, n = self.U, self.X, self.dx, self.n
        val = [] # values for the approximation at the t-th time step
        for j in range(len(X)):
            if j == 0: # left boundary
                val.append((U[t][n-3]-8*U[t][n-2]+13*U[t][n-1]
                            -13*U[t][j+1]+8*U[t][j+2]
                            -U[t][j+3])/(8*dx**3))
            elif j == 1:
                val.append((U[t][n-2]-8*U[t][n-1]+13*U[t][j-1]
                            -13*U[t][j+1]+8*U[t][j+2]
                            -U[t][j+3])/(8*dx**3))
            elif j == 2:
                val.append((U[t][n-1]-8*U[t][j-2]+13*U[t][j-1]
                            -13*U[t][j+1]+8*U[t][j+2]
                            -U[t][j+3])/(8*dx**3))
            elif j == n-3:
                val.append((U[t][j-3]-8*U[t][j-2]+13*U[t][j-1]
                            -13*U[t][j+1]+8*U[t][j+2]
                            -U[t][0])/(8*dx**3))
            elif j == n-2:
                val.append((U[t][j-3]-8*U[t][j-2]+13*U[t][j-1]
                            -13*U[t][j+1]+8*U[t][0]
                            -U[t][1])/(8*dx**3))
            elif j == n-1: # right boundary
                val.append((U[t][j-3]-8*U[t][j-2]+13*U[t][j-1]
                            -13*U[t][0]+8*U[t][1]-U[t][2])/(8*dx**3))
            else:
                val.append((U[t][j-3]-8*U[t][j-2]+13*U[t][j-1]
                            -13*U[t][j+1]+8*U[t][j+2]
                            -U[t][j+3])/(8*dx**3))
        return np.array(val)

    def central_uxxxx_2(self, t):
        """Return the 2nd order central difference approximation
        of the fourth derivative of U with respect to x at time step t.
        """
        U, X, dx, n = self.U, self.X, self.dx, self.n
        val = [] # values for the approximation at the t-th time step
        for j in range(len(X)):
            if j == 0: # left boundary
                val.append((U[t][n-2]-4*U[t][n-1]+6*U[t][j]-4*U[t][j+1]
                            +U[t][j+2])/(dx**4))
            elif j == 1:
                val.append((U[t][n-1]-4*U[t][j-1]+6*U[t][j]-4*U[t][j+1]
                            +U[t][j+2])/(dx**4))
            elif j == n-2:
                val.append((U[t][j-2]-4*U[t][j-1]+6*U[t][j]-4*U[t][j+1]
                            +U[t][0])/(dx**4))
            elif j == n-1: # right boundary
                val.append((U[t][j-2]-4*U[t][j-1]+6*U[t][j]-4*U[t][0]
                            +U[t][1])/(dx**4))
            else:
                val.append((U[t][j-2]-4*U[t][j-1]+6*U[t][j]-4*U[t][j+1]
                            +U[t][j+2])/(dx**4))
        return np.array(val)

    def central_uxxxx_4(self, t):
        """Return the 4th order central difference approximation
        of the fourth derivative of U with respect to x at time step t.
        """
        U, X, dx, n = self.U, self.X, self.dx, self.n
        val = [] # values for the approximation at the t-th time step
        for j in range(len(X)):
            if j == 0: # left boundary
                val.append((-U[t][n-3]+12*U[t][n-2]-39*U[t][n-1]
                            +56*U[t][j]-39*U[t][j+1]+12*U[t][j+2]
                            -U[t][j+3])/(6*dx**4))
            elif j == 1:
                val.append((-U[t][n-2]+12*U[t][n-1]-39*U[t][j-1]
                            +56*U[t][j]-39*U[t][j+1]+12*U[t][j+2]
                            -U[t][j+3])/(6*dx**4))
            elif j == 2:
                val.append((-U[t][n-1]+12*U[t][j-2]-39*U[t][j-1]
                            +56*U[t][j]-39*U[t][j+1]+12*U[t][j+2]
                            -U[t][j+3])/(6*dx**4))
            elif j == n-3:
                val.append((-U[t][j-3]+12*U[t][j-2]-39*U[t][j-1]
                            +56*U[t][j]-39*U[t][j+1]+12*U[t][j+2]
                            -U[t][0])/(6*dx**4))
            elif j == n-2:
                val.append((-U[t][j-3]+12*U[t][j-2]-39*U[t][j-1]
                            +56*U[t][j]-39*U[t][j+1]+12*U[t][0]
                            -U[t][1])/(6*dx**4))
            elif j == n-1: # right boundary
                val.append((-U[t][j-3]+12*U[t][j-2]-39*U[t][j-1]
                            +56*U[t][j]-39*U[t][0]+12*U[t][1]
                            -U[t][2])/(6*dx**4))
            else:
                val.append((-U[t][j-3]+12*U[t][j-2]-39*U[t][j-1]
                            +56*U[t][j]-39*U[t][j+1]+12*U[t][j+2]
                            -U[t][j+3])/(6*dx**4))
        return np.array(val)
app = App()
