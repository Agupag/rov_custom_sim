#!/usr/bin/env python3
"""
Physics Auto-Analyzer & Optimizer

This script:
1. Reads the latest log file
2. Analyzes physics behavior
3. Recommends parameter changes
4. Provides exact code changes to apply

Run this AFTER running rov_sim.py once.
"""

import os
import re
import math
from datetime import datetime
from pathlib import Path

class PhysicsAnalyzer:
    TARGET_SURGE_SPEED_MIN = 0.3
    TARGET_SURGE_SPEED_MAX = 0.5

    def __init__(self, log_file=None):
        """Initialize with log file."""
        log_file = self._resolve_log_file(log_file)
        
        self.log_file = log_file
        self.data = self._parse_log()

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
        
    def _parse_log(self):
        """Parse the log file and extract physics data."""
        with open(self.log_file, 'r') as f:
            content = f.read()
        
        # Find CSV section
        csv_start = content.find("# DETAILED_PHYSICS_CSV")
        if csv_start == -1:
            raise ValueError("No physics CSV data found in log file")
        
        # Get header line
        csv_content = content[csv_start:]
        lines = csv_content.split('\n')[1:]  # Skip header comment
        
        # Parse CSV lines
        data = {
            'times': [],
            'positions': [],
            'velocities_world': [],
            'velocities_body': [],
            'accelerations': [],
            'thrusts': [],
            'thrust_levels': [],
            'drag_forces': [],
            'buoyancy_forces': [],
            'roll': [],
            'pitch': [],
            'yaw': [],
        }
        
        for line in lines:
            if not line.strip() or line.startswith('#'):
                continue
            try:
                parts = line.split(',')
                if len(parts) < 20:
                    continue
                
                # Parse: time, step, px, py, pz, vx_w, vy_w, vz_w, vx_b, vy_b, vz_b, ...
                t = float(parts[0])
                data['times'].append(t)
                
                # Position
                px, py, pz = float(parts[2]), float(parts[3]), float(parts[4])
                data['positions'].append((px, py, pz))
                
                # Velocity (world)
                vx_w = float(parts[5])
                vy_w = float(parts[6])
                vz_w = float(parts[7])
                v_mag = math.sqrt(vx_w**2 + vy_w**2 + vz_w**2)
                data['velocities_world'].append((vx_w, vy_w, vz_w, v_mag))
                
                # Velocity (body) - typically parts 8-10
                vx_b = float(parts[8])
                vy_b = float(parts[9])
                vz_b = float(parts[10])
                data['velocities_body'].append((vx_b, vy_b, vz_b))
                
            except (ValueError, IndexError):
                continue
        
        if not data['times']:
            raise ValueError("Failed to parse any physics data from log")
        
        return data
    
    def analyze(self):
        """Analyze physics behavior."""
        print("\n" + "="*70)
        print("🔬 PHYSICS ANALYSIS REPORT")
        print("="*70)
        
        results = {
            'max_speed': self._analyze_speed(),
            'acceleration': self._analyze_acceleration(),
            'stability': self._analyze_stability(),
            'drag_effectiveness': self._analyze_drag(),
            'thrust_utilization': self._analyze_thrust(),
        }
        
        return results
    
    def _analyze_speed(self):
        """Analyze maximum speed achieved."""
        max_speeds = [v[3] for v in self.data['velocities_world']]
        max_v = max(max_speeds)
        avg_v = sum(max_speeds) / len(max_speeds)
        
        print(f"\n📊 SPEED ANALYSIS")
        print(f"  Max speed achieved:        {max_v:.3f} m/s")
        print(f"  Average speed:             {avg_v:.3f} m/s")
        print(
            f"  Speed stability:           "
            f"{'✅ In expected ROV range' if self.TARGET_SURGE_SPEED_MIN <= max_v <= self.TARGET_SURGE_SPEED_MAX else '⚠️ Outside expected ROV range'}"
        )
        
        return {'max': max_v, 'avg': avg_v, 'data': max_speeds}
    
    def _analyze_acceleration(self):
        """Analyze acceleration characteristics."""
        if len(self.data['velocities_world']) < 2:
            return {}
        
        accelerations = []
        dt = (self.data['times'][-1] - self.data['times'][0]) / len(self.data['times'])
        
        for i in range(1, len(self.data['velocities_world'])):
            v_prev = self.data['velocities_world'][i-1][3]
            v_curr = self.data['velocities_world'][i][3]
            a = (v_curr - v_prev) / dt if dt > 0 else 0
            accelerations.append(a)
        
        max_a = max(accelerations) if accelerations else 0
        avg_a = sum(accelerations) / len(accelerations) if accelerations else 0
        
        print(f"\n⚡ ACCELERATION ANALYSIS")
        print(f"  Max acceleration:          {max_a:.3f} m/s²")
        print(f"  Average acceleration:      {avg_a:.3f} m/s²")
        
        return {'max': max_a, 'avg': avg_a}
    
    def _analyze_stability(self):
        """Analyze pitch/roll stability."""
        print(f"\n🏗️  STABILITY ANALYSIS")
        print(f"  Position range (X):        {self.data['positions'][0][0]:.2f} to {max([p[0] for p in self.data['positions']]):.2f} m")
        print(f"  Position range (Y):        {min([p[1] for p in self.data['positions']]):.2f} to {max([p[1] for p in self.data['positions']]):.2f} m")
        print(f"  Position range (Z):        {min([p[2] for p in self.data['positions']]):.2f} to {max([p[2] for p in self.data['positions']]):.2f} m")
        
        return {}
    
    def _analyze_drag(self):
        """Analyze drag effectiveness."""
        max_speed = max(v[3] for v in self.data['velocities_world'])
        print(f"\n🌊 DRAG EFFECTIVENESS")
        print(f"  Velocity profile shows drag effect")
        print(
            f"  Linear damping appears "
            f"{'plausible' if max_speed <= self.TARGET_SURGE_SPEED_MAX else 'likely low for this ROV setup'}"
        )
        
        return {}
    
    def _analyze_thrust(self):
        """Analyze thrust utilization."""
        print(f"\n🚀 THRUST UTILIZATION")
        print(f"  Thrusters should be toggled during run")
        
        return {}
    
    def generate_recommendations(self):
        """Generate physics change recommendations."""
        analysis = self.analyze()
        max_v = analysis['max_speed']['max']
        
        print("\n" + "="*70)
        print("💡 RECOMMENDATIONS")
        print("="*70)
        
        recommendations = []
        
        # Speed analysis
        if max_v > self.TARGET_SURGE_SPEED_MAX:
            recommendations.append({
                'param': 'LIN_DRAG_BODY',
                'reason': (
                    f'Max speed {max_v:.2f} m/s is too high '
                    f'(target: {self.TARGET_SURGE_SPEED_MIN:.1f}-{self.TARGET_SURGE_SPEED_MAX:.1f} m/s)'
                ),
                'action': 'Increase drag coefficients by 20-40%',
                'priority': 'HIGH'
            })
        elif max_v < self.TARGET_SURGE_SPEED_MIN:
            recommendations.append({
                'param': 'LIN_DRAG_BODY',
                'reason': (
                    f'Max speed {max_v:.2f} m/s is too low '
                    f'(target: {self.TARGET_SURGE_SPEED_MIN:.1f}-{self.TARGET_SURGE_SPEED_MAX:.1f} m/s)'
                ),
                'action': 'Decrease drag coefficients by 20-40%',
                'priority': 'HIGH'
            })
        else:
            recommendations.append({
                'param': 'LIN_DRAG_BODY',
                'reason': (
                    f'Max speed {max_v:.2f} m/s is in the expected range '
                    f'({self.TARGET_SURGE_SPEED_MIN:.1f}-{self.TARGET_SURGE_SPEED_MAX:.1f} m/s)'
                ),
                'action': 'No change needed',
                'priority': 'LOW'
            })
        
        # Print recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['param']}")
            print(f"   Priority:  {rec['priority']}")
            print(f"   Reason:    {rec['reason']}")
            print(f"   Action:    {rec['action']}")
        
        return recommendations
    
    def print_current_values(self):
        """Print current physics parameters from rov_sim.py."""
        print("\n" + "="*70)
        print("⚙️  CURRENT PHYSICS PARAMETERS (from rov_sim.py)")
        print("="*70)
        
        try:
            with open('rov_sim.py', 'r') as f:
                content = f.read()
            
            params_to_find = [
                'MAX_THRUST_H',
                'MAX_THRUST_V',
                'LIN_DRAG_BODY',
                'RIGHTING_K_RP',
                'RIGHTING_KD_RP',
                'BUOYANCY_SCALE',
                'BALLAST_SCALE',
                'MAX_SPEED',
            ]
            
            for param in params_to_find:
                match = re.search(f'{param}\\s*=\\s*([^\n]+)', content)
                if match:
                    value = match.group(1).strip()
                    print(f"  {param:25s} = {value}")
        
        except Exception as e:
            print(f"  Error reading parameters: {e}")


def main():
    try:
        analyzer = PhysicsAnalyzer()
        analyzer.print_current_values()
        analyzer.generate_recommendations()
        
        print("\n" + "="*70)
        print("✅ Analysis complete. Ready for physics optimization.")
        print("="*70 + "\n")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPlease run 'python3 rov_sim.py' first to generate a log file.")
    except ValueError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
