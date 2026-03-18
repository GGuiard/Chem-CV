import matplotlib.pyplot as plt
import numpy as np

# TODO: 
#   - adapt the plots to the system studied
#   - set aspect ratio 1:1 to phipsi plots
#   - give the possibility to not show err in fes1D plots
#   - setup save option, location, name, dpi, transparent... and an option to not show the plots if saved
#   - make multiple plots with all the trj, all the fes

# def trj_phi(time, phi, transient=0):
#     if transient!=0:
#         plt.axhspan(0, time[transient], color='grey', alpha=0.3)
#     plt.plot(phi, time, 'o', ms=1)
#     plt.xlim((-np.pi,np.pi))
#     plt.xlabel(r"$\phi\ [rad]$")
#     plt.ylabel("t [ps]")
#     plt.show()

# def trj_psi(time, psi, transient=0):
#     plt.plot(psi, time, 'o', ms=1)
#     plt.xlim((-np.pi,np.pi))
#     plt.xlabel(r"$\psi\ [rad]$")
#     plt.ylabel("t [ns]")
#     plt.show()

# def trj_phipsi(phi, psi):
#     plt.plot(phi, psi, 'o', ms=1)
#     plt.xlabel(r"$\phi\ [rad]$")
#     plt.ylabel(r"$\psi\ [rad]$")
#     plt.xlim((-np.pi,np.pi))
#     plt.ylim((-np.pi,np.pi))
#     plt.show()

# def fes_phi(grid, fes, err, ref=True):
#     plt.fill_between(grid, fes-err, fes+err, alpha=0.3)
#     plt.plot(grid, fes)
#     plt.xlim((-np.pi,np.pi))
#     plt.ylim(-10,80)
#     plt.xlabel(r"$\phi\ [rad]$")
#     plt.ylabel(r"$F(\phi)\ [kJ/mol]$")
#     plt.show()

# def fes_psi(grid, fes, err, ref=False):
#     plt.fill_between(grid, fes-err, fes+err, alpha=0.3)
#     plt.plot(grid, fes)
#     plt.xlim((-np.pi,np.pi))
#     plt.ylim(-5,40)
#     plt.xlabel(r"$\psi\ [rad]$")
#     plt.ylabel(r"$F(\psi)\ [kJ/mol]$")
#     plt.show()

# def fes_phipsi(grid_1, grid_2, fes, ref=True, colorbar=False):
#     contours = np.linspace(-10, 80, 15)
#     im = plt.contourf(grid_1, grid_2, fes, levels=contours)
#     plt.xlim((-np.pi,np.pi))
#     plt.ylim((-np.pi,np.pi))
#     plt.xlabel(r"$\phi\ [rad]$")
#     plt.ylabel(r"$\psi\ [rad]$")
#     if colorbar:
#         plt.colorbar(im, label=r"$F(\phi,\psi)\ [kJ/mol]$")
#     plt.show()

# def err_fes_phipsi(grid_1, grid_2, err, ref=True, colorbar=False):
#     im = plt.contourf(grid_1, grid_2, err, levels=15)
#     plt.xlim((-np.pi,np.pi))
#     plt.ylim((-np.pi,np.pi))
#     plt.xlabel(r"$\phi\ [rad]$")
#     plt.ylabel(r"$\psi\ [rad]$")
#     if colorbar:
#         plt.colorbar(im, label=r"$Err(F(\phi,\psi))\ [kJ/mol]$")
#     plt.show()

# plt.plot(av_phi)
# plt.xlabel("number of frames")
# plt.ylabel(r"$\langle\phi\rangle\ [rad]$")
# plt.show()

# plt.axhspan(av_phiA-std_phiA, av_phiA+std_phiA, alpha=0.2)
# plt.axhline(av_phiA, color='C0', linestyle='--')
# plt.plot(phiA, 'o', ms=3)
# plt.xlabel("number of partitions")
# plt.ylabel(r"$\langle\phi\rangle_A\ [rad]$")
# plt.show()