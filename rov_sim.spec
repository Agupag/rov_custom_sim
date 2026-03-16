# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for ROV Simulator.
Builds a one-folder distributable that includes all data files,
pybullet native libraries, and the ROV mesh assets.

Usage (on Windows):
    pip install pyinstaller pybullet numpy opencv-python Pillow
    pyinstaller rov_sim.spec
"""

import os
import sys
import importlib

block_cipher = None

# Locate pybullet_data package directory (contains plane.urdf etc.)
pybullet_data_spec = importlib.util.find_spec("pybullet_data")
if pybullet_data_spec and pybullet_data_spec.submodule_search_locations:
    pybullet_data_dir = pybullet_data_spec.submodule_search_locations[0]
else:
    import pybullet_data as _pbd
    pybullet_data_dir = os.path.dirname(_pbd.__file__)

# Data files to bundle alongside the exe
datas = [
    # ROV mesh assets (must be next to the exe)
    ('Assembly 1.obj', '.'),
    ('Assembly 1.gltf', '.'),
    # versioned CAD configs (v1/v2/v3 used by THRUSTER_CONFIGS auto-discovery)
    ('v1.obj', '.'), ('v1.gltf', '.'), ('v1.mtl', '.'),
    ('v2.obj', '.'), ('v2.gltf', '.'), ('v2.mtl', '.'),
    ('v3.obj', '.'), ('v3.gltf', '.'), ('v3.mtl', '.'),
    # joystick panel and shared-memory contract modules
    ('joystick_panel.py', '.'),
    ('sim_shared.py', '.'),
    # pybullet_data (plane.urdf, textures, etc.)
    (pybullet_data_dir, 'pybullet_data'),
]

# Hidden imports that PyInstaller may not auto-detect
hiddenimports = [
    'pybullet',
    'pybullet_data',
    'numpy',
    'joystick_panel',
    'sim_shared',
    'tkinter',
    'ctypes',
    'multiprocessing',
    'json',
    'struct',
    'PIL',
    'PIL.ImageGrab',
    'cv2',
]

a = Analysis(
    ['rov_sim.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],                    # Not one-file (use COLLECT for one-folder)
    exclude_binaries=True,
    name='ROV_Simulator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,             # Don't compress — avoids AV false positives
    console=True,           # Show console for log output
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='ROV_Simulator',
)
