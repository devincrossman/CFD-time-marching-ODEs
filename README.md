This is a python application I wrote for a computational fluid dynamics class to study different time marching solutions to the one-dimensional first order wave equation a.k.a. the 1-D advection equation.

![application screenshot](https://devincrossman.com/aero/screenshot.png "application screenshot")

Standalone versions of the application tested on Windows 10 and Mac OS X 10.11 that don't require a python installation are available at https://www.devincrossman.com/aero/ along with the report containing the derivation of the equations and analysis.

The Python script was written using Python 3.6.5 but may work with earlier versions. The script depends on a few different modules that must be installed prior to running the program. These modules can be installed system wide or within a virtual environment. 

The program uses the NumPy module to perform efficient linear algebra calculations using vectors and matrices, and the Matplotlib module to plot the results. A graphical user interface (GUI) was created using the TkInter package and safe interpretation of user input for the initial conditions was achieved with the NumExpr package.

Assuming Python 3.x and the package manager pip are already installed and located some- where in your system path, the following procedure is used to run the program. After cloning the repository or downloading the aero.py source code, execute the following commands from a command line prompt in that directory.

Install virtualenv if not already installed.
```
$ pip install virtualenv
```
Create a new virtual environment in the current directory and activate it.
```
$ virtualenv .
$ source ./bin/activate # on Mac
$ .\Scripts\activate # on Windows
```
This creates an isolated environment for the Python program to run in without affecting or being affected by other modules installed on the system.

Next, install NumPy, Matplotlib, and NumExpr.
```
$ pip install numpy matplotlib numexpr
```
Run the program.
```
$ python aero.py
```
Use the command `deactivate` to exit the virtual environment.

Options to configure the numerical methods can be selected from drop downs and text inputs. A list of supported functions for the initial conditions can be found in the NumExpr documentation at the following url:
http://numexpr.readthedocs.io/en/latest/user_guide.html#supported-functions