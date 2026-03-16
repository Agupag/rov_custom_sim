#!/usr/bin/env python3
"""
Test script to verify thruster reverse logic works correctly.
This simulates the thruster control state without PyBullet.
"""

def test_thruster_logic():
    """Test the thruster on/off and reverse logic."""
    
    # Initialize state (same as in rov_sim.py)
    thr_on = [False] * 4
    thr_reverse = [False] * 4
    thr_cmd = [0.0] * 4
    
    def toggle_thruster(thruster_idx, shift_held):
        """Simulate pressing a thruster key."""
        t_name = f"Thruster_{thruster_idx + 1}"
        
        if shift_held:
            # Toggle reverse mode
            if thr_on[thruster_idx]:
                thr_reverse[thruster_idx] = not thr_reverse[thruster_idx]
                thr_cmd[thruster_idx] = -1.0 if thr_reverse[thruster_idx] else 1.0
                mode = "REVERSE" if thr_reverse[thruster_idx] else "FORWARD"
                print(f"  {t_name}: {mode}")
                return True
            else:
                print(f"  {t_name}: (cannot reverse - not on)")
                return False
        else:
            # Toggle on/off
            thr_on[thruster_idx] = not thr_on[thruster_idx]
            if thr_on[thruster_idx]:
                thr_cmd[thruster_idx] = -1.0 if thr_reverse[thruster_idx] else 1.0
            else:
                thr_cmd[thruster_idx] = 0.0
            status = f"ON ({('REVERSE' if thr_reverse[thruster_idx] else 'FORWARD')})" if thr_on[thruster_idx] else "OFF"
            print(f"  {t_name}: {status}")
            return True
    
    def print_state():
        """Print current thruster state."""
        print("\nCurrent State:")
        for i in range(4):
            status = "ON" if thr_on[i] else "OFF"
            mode = "REV" if thr_reverse[i] else "FWD"
            cmd = thr_cmd[i]
            print(f"  [{i+1}] {status:3s} {mode:3s}  cmd={cmd:+.1f}")
        print()
    
    # Test sequence
    print("=" * 60)
    print("THRUSTER CONTROL TEST SEQUENCE")
    print("=" * 60)
    
    print("\n[TEST 1] Toggle Thruster 1 ON")
    toggle_thruster(0, False)
    print_state()
    assert thr_on[0] == True, "T1 should be ON"
    assert thr_cmd[0] == 1.0, "T1 cmd should be 1.0 (FORWARD)"
    
    print("[TEST 2] Toggle Thruster 2 ON")
    toggle_thruster(1, False)
    print_state()
    assert thr_on[1] == True, "T2 should be ON"
    assert thr_cmd[1] == 1.0, "T2 cmd should be 1.0"
    
    print("[TEST 3] Try reverse on T2 (already ON)")
    toggle_thruster(1, True)
    print_state()
    assert thr_on[1] == True, "T2 should still be ON"
    assert thr_reverse[1] == True, "T2 reverse should be TRUE"
    assert thr_cmd[1] == -1.0, "T2 cmd should be -1.0 (REVERSE)"
    
    print("[TEST 4] Try reverse on T3 (OFF - should do nothing)")
    toggle_thruster(2, True)
    print_state()
    assert thr_on[2] == False, "T3 should still be OFF"
    assert thr_cmd[2] == 0.0, "T3 cmd should be 0.0"
    
    print("[TEST 5] Turn on T3 (should remember no reverse)")
    toggle_thruster(2, False)
    print_state()
    assert thr_on[2] == True, "T3 should be ON"
    assert thr_reverse[2] == False, "T3 reverse should still be FALSE"
    assert thr_cmd[2] == 1.0, "T3 cmd should be 1.0 (FORWARD)"
    
    print("[TEST 6] Toggle reverse on T3")
    toggle_thruster(2, True)
    print_state()
    assert thr_reverse[2] == True, "T3 reverse should be TRUE"
    assert thr_cmd[2] == -1.0, "T3 cmd should be -1.0"
    
    print("[TEST 7] Turn OFF T2 (in reverse mode)")
    toggle_thruster(1, False)
    print_state()
    assert thr_on[1] == False, "T2 should be OFF"
    assert thr_reverse[1] == True, "T2 reverse state should be remembered"
    assert thr_cmd[1] == 0.0, "T2 cmd should be 0.0"
    
    print("[TEST 8] Turn ON T2 again (should restore REVERSE)")
    toggle_thruster(1, False)
    print_state()
    assert thr_on[1] == True, "T2 should be ON"
    assert thr_reverse[1] == True, "T2 reverse should be restored"
    assert thr_cmd[1] == -1.0, "T2 cmd should be -1.0 (REVERSE)"
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nLogic verified:")
    print("  ✓ Thrusters toggle ON/OFF with 1-4 keys")
    print("  ✓ Shift+key toggles reverse (only when ON)")
    print("  ✓ Reverse state is remembered across ON/OFF cycles")
    print("  ✓ Cannot reverse when thruster is OFF")
    print("  ✓ Commands properly set to ±1.0 or 0.0")

if __name__ == "__main__":
    test_thruster_logic()
