#!/usr/bin/env python3
"""
Simple test script to verify JugVid2cpp integration works.
This bypasses the full juggling_tracker dependencies.
"""

import sys
import os
import time
import cv2
import numpy as np

# Add the juggling_tracker modules to path
sys.path.append('/home/twain/Projects/JugVid2/juggling_tracker')

try:
    from modules.jugvid2cpp_interface import JugVid2cppInterface
    print("✓ Successfully imported JugVid2cppInterface")
except ImportError as e:
    print(f"✗ Failed to import JugVid2cppInterface: {e}")
    sys.exit(1)

def test_jugvid2cpp_integration():
    """Test the JugVid2cpp integration."""
    print("Testing JugVid2cpp Integration...")
    
    # Initialize the interface
    interface = JugVid2cppInterface()
    print(f"✓ Interface initialized with mode: {interface.mode}")
    
    # Test starting the interface
    print("Starting JugVid2cpp interface...")
    if interface.start():
        print("✓ JugVid2cpp interface started successfully")
        
        # Test getting frames for a few seconds
        print("Testing frame acquisition...")
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 5.0:  # Test for 5 seconds
            # Get frames
            depth_frame, color_frame, depth_image, color_image = interface.get_frames()
            
            if color_image is not None:
                frame_count += 1
                print(f"✓ Frame {frame_count}: {color_image.shape} - {color_image.dtype}")
                
                # Test getting identified balls
                identified_balls = interface.get_identified_balls()
                if identified_balls:
                    print(f"  → Found {len(identified_balls)} balls:")
                    for ball in identified_balls:
                        if 'original_3d' in ball:
                            x, y, z = ball['original_3d']
                            print(f"    - {ball['name']}: ({x:.3f}, {y:.3f}, {z:.3f})")
                
                # Show the frame (if display is available)
                try:
                    cv2.imshow('JugVid2cpp Integration Test', color_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except:
                    pass  # No display available
            
            time.sleep(0.1)  # 10 FPS for testing
        
        print(f"✓ Processed {frame_count} frames in 5 seconds")
        
        # Test status
        status = interface.get_status()
        print(f"✓ Interface status: {status}")
        
        # Stop the interface
        interface.stop()
        print("✓ Interface stopped successfully")
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        return True
    else:
        print("✗ Failed to start JugVid2cpp interface")
        return False

if __name__ == "__main__":
    print("JugVid2cpp Integration Test")
    print("=" * 40)
    
    success = test_jugvid2cpp_integration()
    
    if success:
        print("\n✓ Integration test PASSED!")
        print("\nThe JugVid2cpp integration is working correctly.")
        print("You can now use: python main.py --jugvid2cpp")
        print("(Make sure to install missing dependencies like filterpy first)")
    else:
        print("\n✗ Integration test FAILED!")
        print("Check the error messages above.")