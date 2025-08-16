#!/usr/bin/env python3
"""
Example Python integration script for the Ball Tracker C++ application.
This demonstrates how to run the C++ tracker and process its output in real-time.
"""

import subprocess
import time
import json
import signal
import sys
from typing import Dict, List, Optional, Tuple

class BallTracker:
    """
    Python wrapper for the C++ Ball Tracker application.
    Handles subprocess management and data parsing.
    """
    
    def __init__(self, executable_path: str = "./build/bin/ball_tracker"):
        """
        Initialize the ball tracker.
        
        Args:
            executable_path: Path to the compiled C++ executable
        """
        self.executable_path = executable_path
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        
    def start(self) -> bool:
        """
        Start the C++ ball tracker process.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            self.process = subprocess.Popen(
                [self.executable_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line-buffered
            )
            self.is_running = True
            print("Ball tracker started successfully")
            return True
            
        except FileNotFoundError:
            print(f"Error: Executable not found at {self.executable_path}")
            print("Make sure to build the C++ application first using:")
            print("  ./build.sh (Linux/Mac) or build.bat (Windows)")
            return False
        except Exception as e:
            print(f"Error starting ball tracker: {e}")
            return False
    
    def stop(self):
        """Stop the ball tracker process."""
        if self.process and self.is_running:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.is_running = False
            print("Ball tracker stopped")
    
    def read_frame(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Read one frame of ball tracking data.
        
        Returns:
            Dictionary with ball positions, or None if no data available
            Format: {"red": {"x": 0.1, "y": -0.2, "z": 0.8}, ...}
        """
        if not self.process or not self.is_running:
            return None
        
        try:
            # Read one line with timeout
            line = self.process.stdout.readline()
            if not line:
                return None
            
            line = line.strip()
            if not line:
                return None
            
            # Parse the ball data
            ball_positions = {}
            ball_data_list = line.split(';')
            
            for ball_data in ball_data_list:
                parts = ball_data.split(',')
                if len(parts) == 4:
                    name = parts[0]
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        ball_positions[name] = {"x": x, "y": y, "z": z}
                    except ValueError:
                        continue  # Skip malformed data
            
            return ball_positions if ball_positions else None
            
        except Exception as e:
            print(f"Error reading frame: {e}")
            return None
    
    def get_error_output(self) -> str:
        """Get any error output from the C++ process."""
        if self.process and self.process.stderr:
            try:
                return self.process.stderr.read()
            except:
                return ""
        return ""

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\nShutting down...")
    sys.exit(0)

def main():
    """Main demonstration function."""
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize the tracker
    tracker = BallTracker()
    
    if not tracker.start():
        return
    
    print("Ball tracker is running. Press Ctrl+C to stop.")
    print("Move colored balls in front of the camera to see tracking data.")
    print("-" * 60)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Read tracking data
            ball_positions = tracker.read_frame()
            
            if ball_positions:
                frame_count += 1
                
                # Display the data
                print(f"Frame {frame_count}:")
                for ball_name, pos in ball_positions.items():
                    print(f"  {ball_name.capitalize()}: "
                          f"X={pos['x']:.3f}m, Y={pos['y']:.3f}m, Z={pos['z']:.3f}m")
                
                # Calculate distance from camera
                for ball_name, pos in ball_positions.items():
                    distance = (pos['x']**2 + pos['y']**2 + pos['z']**2)**0.5
                    print(f"  {ball_name.capitalize()} distance: {distance:.3f}m")
                
                print("-" * 40)
                
                # Show FPS every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"Average FPS: {fps:.1f}")
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.001)
            
            # Check if process is still running
            if tracker.process and tracker.process.poll() is not None:
                print("C++ tracker process has exited")
                error_output = tracker.get_error_output()
                if error_output:
                    print(f"Error output: {error_output}")
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()

if __name__ == "__main__":
    main()