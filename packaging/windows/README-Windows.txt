MDATK @VERSION@ for Windows 10 / 11 (64-bit) - BETA, untested
==============================================================

Molecular Dynamic Analysis Tool Kit: analysis and multi-panel video rendering of molecular
dynamics trajectories of RNA systems.

Start
  mdatk-gui.exe        the graphical interface (double-click; a project file can be dropped on it)
  mdatk.exe            the command line, e.g. in PowerShell or cmd:
                         .\mdatk.exe --help
                         .\mdatk.exe demo C:\Temp\mdatk-demo
                         .\mdatk.exe render -p C:\Temp\mdatk-demo\demo.mdatk.toml

Nothing else is needed for analysis and videos: Python, GTK 3, NumPy / SciPy / Matplotlib,
MDAnalysis, Barnaba and FFmpeg (_internal\tools\ffmpeg.exe) are included.

Not included / not available on Windows
  * cpptraj (AmberTools) for stripping and cleaning trajectories: AmberTools does not run on
    Windows. Clean the trajectory on Linux, a cluster or in WSL, or analyse trajectories that
    are already cleaned (leave "Strip and clean the trajectory with cpptraj" off).
  * Generated workflow scripts (Amber, GROMACS, Rosetta, Phenix, Coot ...) are bash scripts:
    MDATK writes them on Windows but does not run them. Copy the project folder to a Linux
    machine or cluster, or open it in WSL (Windows Subsystem for Linux), and run them there
    with "bash <script>".
  * The optional VTK 3D renderer (the built-in renderer is used).

This is a beta: please report problems at https://github.com/htchamblee/MDATK/issues

Licence: GNU General Public License v3 (LICENSE.txt). The bundled third-party components and
their licences are listed in licenses\THIRD_PARTY_NOTICES.txt; FFmpeg's licence and source:
licenses\ffmpeg\FFMPEG-NOTICE.txt.
