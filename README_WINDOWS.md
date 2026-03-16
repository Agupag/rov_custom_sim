# ROV Simulator — Windows Standalone Build

## Quick Start (Running a Pre-Built .exe)

If someone already built the `.exe` for you:

1. Unzip the `ROV_Simulator` folder
2. Double-click **`ROV_Simulator.exe`**
3. No Python or other software needed!

---

## Building the .exe Yourself

### Prerequisites

- **Windows 10 or 11**
- **Python 3.10–3.12** from [python.org](https://www.python.org/downloads/)
  - ⚠️ During install, check **"Add Python to PATH"**
- Internet connection (to download packages)

> **Note:** Python 3.13+ may not work with PyBullet. Use **Python 3.12** for best compatibility.

### Option A: Automatic Build (Recommended)

1. Copy this entire project folder to a Windows PC
2. Double-click **`build_windows_exe.bat`**
3. Wait for it to finish (~5 minutes)
4. Find your executable at: `dist\ROV_Simulator\ROV_Simulator.exe`

### Option B: Manual Build

Open **Command Prompt** in the project folder and run:

```cmd
python -m venv build_venv
build_venv\Scripts\activate
pip install pyinstaller pybullet numpy opencv-python Pillow
pyinstaller rov_sim.spec --noconfirm --clean
```

The output will be in `dist\ROV_Simulator\`.

---

## Distributing to Others

The `dist\ROV_Simulator\` folder contains **everything** needed to run:

1. Right-click the `dist\ROV_Simulator` folder → **Send to → Compressed (zipped) folder**
2. Share the `.zip` file
3. Recipients just unzip and double-click `ROV_Simulator.exe`

**No Python installation required on the target machine.**

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `'python' is not recognized` | Install Python and check "Add to PATH" |
| `pip install pybullet` fails | Use Python 3.12 (not 3.13+) |
| `.exe` immediately closes | Right-click → Run as Administrator, or run from Command Prompt to see errors |
| Windows Defender blocks it | Click "More info" → "Run anyway" (false positive from PyInstaller) |
| Missing DLL errors | Install [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| Antivirus quarantines the .exe | Add the `ROV_Simulator` folder to your antivirus exclusions |

---

## Controls

| Key | Action |
|-----|--------|
| **ESC** | Quit |
| **R** | Reset ROV position |
| **F** | Toggle follow camera |
| **G** | Toggle chase camera |
| **T** | Toggle top-down view |
| **0** | Emergency surface |
| **I/J/K/L** | Camera pitch/yaw |
| **U/O** | Camera zoom in/out |
| **⏺ REC button** | Start/stop MP4 recording |

The controller panel (joystick window) opens automatically alongside the 3D view.

---

## Project Files

| File | Purpose |
|------|---------|
| `rov_sim.py` | Main simulator (also runs directly with Python) |
| `joystick_panel.py` | Controller UI panel |
| `rov_sim.spec` | PyInstaller build configuration |
| `build_windows_exe.bat` | Automated build script |
| `Assembly 1.obj/gltf` | ROV 3D model |
