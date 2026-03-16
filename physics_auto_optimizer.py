#!/usr/bin/env python3
"""
Advanced Physics Auto-Optimizer

This script:
1. Reads the latest log file
2. Analyzes physics behavior in detail
3. Calculates optimal parameter changes
4. Generates code patches to apply automatically
5. Displays expected improvements

Run this AFTER running rov_sim.py once.
"""

import os
import re
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class PhysicsParam:
    """Represents a physics parameter to potentially change."""
    name: str
    current_value: str
    current_parsed: float
    recommended_value: float
    ratio: float  # multiplier from current to recommended
    reason: str
    priority: str  # HIGH, MEDIUM, LOW
    code_location: str  # Description of where to find in code


class AdvancedPhysicsAnalyzer:
    TARGET_SURGE_SPEED_MIN = 0.3
    TARGET_SURGE_SPEED_MAX = 0.5

    def __init__(self, log_file=None):
        """Initialize with log file."""
        log_file = self._resolve_log_file(log_file)
        
        self.log_file = log_file
        self.rov_sim_path = "rov_sim.py"
        self.data = {}
        self.parse_log()
        self.current_params = self._read_current_params()

    def _resolve_log_file(self, log_file):
        """Pick an explicit log file or the newest log containing detailed CSV data."""
        if log_file is not None:
            return str(log_file)

        log_files = sorted(Path(".").glob("rov_sim_log_*.txt"))
        if not log_files:
            raise FileNotFoundError("No log file found. Run rov_sim.py first!")

        for candidate in reversed(log_files):
            try:
                if "# DETAILED_PHYSICS_CSV" in candidate.read_text():
                    return str(candidate)
            except OSError:
                continue

        raise ValueError(
            "No detailed physics CSV logs found. Enable LOG_PHYSICS_DETAILED in rov_sim.py and run the simulator again."
        )
        
    def parse_log(self):
        """Parse the log file and extract physics data."""
        print(f"\n📖 Reading log file: {self.log_file}")
        self.data = {
            'times': [],
            'positions': [],
            'velocities': [],
            'accelerations': [],
            'angular_vel': [],
            'drag_forces': [],
            'thruster_levels': [],
        }
        
        with open(self.log_file, 'r') as f:
            content = f.read()
        
        # Find CSV section
        csv_start = content.find("# DETAILED_PHYSICS_CSV")
        if csv_start == -1:
            raise ValueError(
                f"Selected log file has no detailed physics CSV data: {self.log_file}"
            )
        
        csv_content = content[csv_start:]
        lines = csv_content.split('\n')[1:]
        
        for line in lines:
            if not line.strip() or line.startswith('#'):
                continue
            try:
                parts = line.split(',')
                if len(parts) < 20:
                    continue
                
                t = float(parts[0])
                px, py, pz = float(parts[2]), float(parts[3]), float(parts[4])
                vx, vy, vz = float(parts[5]), float(parts[6]), float(parts[7])
                wx, wy, wz = float(parts[11]), float(parts[12]), float(parts[13])
                
                v_mag = math.sqrt(vx**2 + vy**2 + vz**2)
                w_mag = math.sqrt(wx**2 + wy**2 + wz**2)
                
                self.data['times'].append(t)
                self.data['positions'].append((px, py, pz))
                self.data['velocities'].append(v_mag)
                self.data['angular_vel'].append(w_mag)
                
                # Parse thrust levels if present (last column)
                if len(parts) > 30:
                    thr_str = parts[-1].strip('"')
                    thr_vals = [float(x) for x in thr_str.split(';')]
                    self.data['thruster_levels'].append(max(abs(t) for t in thr_vals))
                
            except (ValueError, IndexError):
                continue
        
        if self.data['times']:
            print(f"✅ Parsed {len(self.data['times'])} physics samples")
        else:
            print("⚠️  Could not parse any physics data")
    
    def _read_current_params(self) -> dict:
        """Read current physics parameters from rov_sim.py."""
        params = {}
        try:
            with open(self.rov_sim_path, 'r') as f:
                content = f.read()
            
            # Extract key parameters
            patterns = {
                'MAX_THRUST_H': r'MAX_THRUST_H\s*=\s*([\d.]+)',
                'MAX_THRUST_V': r'MAX_THRUST_V\s*=\s*([\d.]+)',
                'LIN_DRAG_BODY': r'LIN_DRAG_BODY\s*=\s*\(([\d., ]+)\)',
                'RIGHTING_K_RP': r'RIGHTING_K_RP\s*=\s*([\d.]+)',
                'RIGHTING_KD_RP': r'RIGHTING_KD_RP\s*=\s*([\d.]+)',
                'BUOYANCY_SCALE': r'BUOYANCY_SCALE\s*=\s*([\d.]+)',
                'MAX_SPEED': r'MAX_SPEED\s*=\s*([\d.]+)',
                'MAX_OMEGA': r'MAX_OMEGA\s*=\s*([\d.]+)',
            }
            
            for name, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    params[name] = match.group(1).strip()
        
        except Exception as e:
            print(f"⚠️  Could not read current parameters: {e}")
        
        return params

    def _get_current_lin_drag_x(self) -> float:
        """Return the surge-axis linear drag coefficient from rov_sim.py."""
        raw = self.current_params.get('LIN_DRAG_BODY')
        if not raw:
            return 4.0
        try:
            return float(raw.split(',')[0].strip())
        except (ValueError, IndexError):
            return 4.0
    
    def analyze(self) -> List[PhysicsParam]:
        """Analyze physics and recommend changes."""
        changes = []
        
        if not self.data['velocities']:
            print("❌ No velocity data to analyze")
            return changes
        
        # Analyze speed
        max_speed = max(self.data['velocities'])
        avg_speed = sum(self.data['velocities']) / len(self.data['velocities'])
        
        print(f"\n" + "="*70)
        print("📊 PHYSICS ANALYSIS")
        print("="*70)
        print(f"\n📈 Speed Analysis:")
        print(f"   Max velocity:        {max_speed:.3f} m/s")
        print(f"   Average velocity:    {avg_speed:.3f} m/s")
        print(f"   Simulation duration: {self.data['times'][-1]:.2f} seconds")
        
        # Check if thrusters were used
        max_thrust = max(self.data['thruster_levels']) if self.data['thruster_levels'] else 0
        print(f"   Thruster activity:   {max_thrust:.1%} of max")
        
        # ANALYSIS: Speed recommendations
        current_drag_x = self._get_current_lin_drag_x()

        if max_speed > self.TARGET_SURGE_SPEED_MAX:
            ratio = max_speed / self.TARGET_SURGE_SPEED_MAX
            change = ratio - 1.0
            changes.append(PhysicsParam(
                name="LIN_DRAG_BODY",
                current_value=self.current_params.get('LIN_DRAG_BODY', 'unknown'),
                current_parsed=current_drag_x,
                recommended_value=current_drag_x * ratio,
                ratio=ratio,
                reason=(
                    f"Speed too high: {max_speed:.2f} m/s "
                    f"(target: {self.TARGET_SURGE_SPEED_MIN:.1f}-{self.TARGET_SURGE_SPEED_MAX:.1f} m/s), "
                    f"increase drag by {change*100:.0f}%"
                ),
                priority="HIGH",
                code_location="Line ~101: LIN_DRAG_BODY = (...)"
            ))
        elif max_speed < self.TARGET_SURGE_SPEED_MIN:
            ratio = max(0.5, max_speed / self.TARGET_SURGE_SPEED_MIN)
            recommended_drag = current_drag_x * ratio
            change = 1.0 - ratio
            changes.append(PhysicsParam(
                name="LIN_DRAG_BODY",
                current_value=self.current_params.get('LIN_DRAG_BODY', 'unknown'),
                current_parsed=current_drag_x,
                recommended_value=recommended_drag,
                ratio=ratio,
                reason=(
                    f"Speed too low: {max_speed:.2f} m/s "
                    f"(target: {self.TARGET_SURGE_SPEED_MIN:.1f}-{self.TARGET_SURGE_SPEED_MAX:.1f} m/s), "
                    f"decrease drag by {change*100:.0f}%"
                ),
                priority="HIGH",
                code_location="Line ~101: LIN_DRAG_BODY = (...)"
            ))
        else:
            print(f"\n✅ Speed is in good range ({max_speed:.2f} m/s)")
        
        return changes
    
    def print_recommendations(self, changes: List[PhysicsParam]):
        """Print recommended changes."""
        if not changes:
            print(f"\n✅ No changes recommended!")
            return
        
        print(f"\n" + "="*70)
        print("💡 RECOMMENDED CHANGES")
        print("="*70)
        
        for i, change in enumerate(changes, 1):
            print(f"\n{i}. {change.name}")
            print(f"   Priority:     {change.priority}")
            print(f"   Current:      {change.current_value}")
            print(f"   Recommended:  {change.recommended_value:.3f}")
            print(f"   Change:       {change.ratio:.2%}")
            print(f"   Reason:       {change.reason}")
            print(f"   Location:     {change.code_location}")
    
    def print_instructions(self, changes: List[PhysicsParam]):
        """Print step-by-step instructions."""
        print(f"\n" + "="*70)
        print("📋 NEXT STEPS")
        print("="*70)
        
        if changes:
            print(f"\n1. Review the recommended changes above")
            print(f"2. Wait for AI to apply the changes to rov_sim.py")
            print(f"3. Run rov_sim.py again to test new parameters")
            print(f"4. AI will analyze the new log and iterate if needed")
            print(f"\n⏳ The AI will now apply the changes automatically...")
        else:
            print(f"\nNo changes needed! Current physics are well-tuned.")
            print(f"Recommended next steps:")
            print(f"  1. Try different thruster combinations")
            print(f"  2. Aim for smooth, responsive control")
            print(f"  3. Check for realistic underwater behavior")


def main():
    try:
        print("\n🔬 ADVANCED PHYSICS ANALYZER")
        print("="*70)
        
        analyzer = AdvancedPhysicsAnalyzer()
        changes = analyzer.analyze()
        analyzer.print_recommendations(changes)
        analyzer.print_instructions(changes)
        
        print("\n" + "="*70)
        print("✅ Analysis complete")
        print("="*70 + "\n")
        
        return changes
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPlease run 'python3 rov_sim.py' first to generate a log file.")
        return []
    except ValueError as e:
        print(f"❌ Error: {e}")
        return []
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    main()
