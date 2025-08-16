# Ball Tracker - High-Performance 3D Juggling Ball Tracking

A high-performance C++ application for real-time 3D tracking of juggling balls using Intel RealSense depth cameras and OpenCV computer vision.

**Last Updated:** 2025-08-16 16:04:00 UTC

## Features

- **Real-time 3D tracking** of pink, orange, green, and yellow juggling balls
- **Interactive calibration mode** with click-to-calibrate functionality
- **Smart occlusion handling** - merges nearby detections when fingers partially cover balls
- **Optimized color combinations** - best results with yellow, pink, green (3-ball) or all 4 colors
- **Persistent settings** - automatically saves and loads color calibration
- **High-performance streaming** at 848x480@90fps (fallback to 1280x720@30fps)
- **Depth-aligned color detection** for accurate 3D coordinate calculation
- **HSV color space detection** for robust color tracking under varying lighting
- **Optimized memory management** to prevent leaks during continuous operation
- **Standard output protocol** for easy integration with other applications

## Requirements

### Hardware
- Intel RealSense depth camera (D400 series recommended)
- USB 3.0 port for camera connection
- Modern CPU with AVX2 support (recommended for optimal performance)

### Software Dependencies
- **C++ Compiler**: Visual Studio 2019+ (MSVC) or GCC 7+ or Clang 6+
- **CMake**: Version 3.10 or higher
- **Intel RealSense SDK 2.0**: Latest version from [GitHub releases](https://github.com/IntelRealSense/librealsense/releases)
- **OpenCV**: Version 4.x from [opencv.org](https://opencv.org/releases/)
- **nlohmann/json**: Included as `json.hpp` (automatically downloaded)

## Installation

### 1. Install Dependencies

#### Windows (Visual Studio)
1. Install Visual Studio Community with "Desktop development with C++" workload
2. Download and install Intel RealSense SDK 2.0 from the official releases
3. Download OpenCV for Windows and extract to `C:\opencv`
4. Download and install CMake, ensuring it's added to system PATH

#### Linux (Ubuntu/Debian)
```bash
# Install build tools
sudo apt update
sudo apt install build-essential cmake pkg-config

# Install Intel RealSense SDK
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main"
sudo apt update
sudo apt install librealsense2-devel

# Install OpenCV
sudo apt install libopencv-dev
```

### 2. Build the Application

```bash
# Clone or navigate to the project directory
cd /path/to/BallTracker

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the application
cmake --build . --config Release

# The executable will be in build/bin/ball_tracker (or ball_tracker.exe on Windows)
```

## Usage

### Calibration Mode (Recommended First Step)
Before using the tracker, calibrate it for your specific lighting conditions and ball colors:

```bash
# Start interactive calibration mode
./build/bin/ball_tracker calibrate

# On Windows
./build/bin/ball_tracker.exe calibrate
```

**Calibration Interface:**
- **Live camera feed** shows detected balls with colored circles and labels
- **Click-to-calibrate**: Simply click on a ball in the video to automatically extract its HSV values
- **Color selection**: Use number keys to choose which color you're calibrating
- **Real-time feedback**: See HSV ranges and detection results immediately
- **Intelligent HSV extraction**: Samples a 5x5 pixel area and calculates optimal ranges with tolerance

**Calibration Workflow:**
1. **Select color**: Press '1' for Pink, '2' for Orange, '3' for Green, '4' for Yellow
2. **Click on ball**: Click directly on a ball of the selected color in the camera feed
3. **Automatic calibration**: HSV values are automatically extracted and set
4. **Verify detection**: See if the ball is now properly detected (colored circle appears)
5. **Repeat**: Switch colors and calibrate other balls
6. **Save**: Press 's' to save your calibration

**Calibration Controls:**
- **'1', '2', '3', '4'**: Select Pink, Orange, Green, or Yellow color for calibration
- **Mouse click**: Click on a ball to calibrate the currently selected color
- **'s'**: Save current settings to `ball_settings.json`
- **'r'**: Reset to default values
- **'q' or ESC**: Quit calibration mode

### Tracking Mode
```bash
# Run the ball tracker (uses saved calibration settings)
./build/bin/ball_tracker

# On Windows
./build/bin/ball_tracker.exe
```

### Output Format
The application continuously outputs detected ball positions to stdout in the following format:
```
ball_name,X,Y,Z;ball_name,X,Y,Z
```

**Example output:**
```
pink,0.15,-0.20,0.88;orange,-0.10,0.35,0.92
green,0.05,0.12,1.15;yellow,0.25,0.18,0.95
pink,0.18,-0.18,0.85;orange,-0.08,0.38,0.94;green,0.02,0.15,1.12;yellow,0.28,0.20,0.98
```

Where:
- `ball_name`: "pink", "orange", "green", or "yellow"
- `X,Y,Z`: 3D coordinates in meters relative to the camera coordinate system
- Multiple balls are separated by semicolons (`;`)
- Each frame outputs one line (empty line if no balls detected)

### Integration with Python Applications

```python
import subprocess

# Start the C++ tracker
process = subprocess.Popen(
    ["./build/bin/ball_tracker"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

# Read real-time data
while True:
    line = process.stdout.readline().strip()
    if line:
        # Parse ball data
        ball_locations = {}
        for ball_data in line.split(';'):
            parts = ball_data.split(',')
            if len(parts) == 4:
                name = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                ball_locations[name] = {'x': x, 'y': y, 'z': z}
        
        # Use the data in your application
        print(f"Ball positions: {ball_locations}")
```

## Configuration

### Color Detection Tuning

**Recommended Method: Click-to-Calibrate**
The easiest and most accurate way to tune color detection:

```bash
./build/bin/ball_tracker calibrate
```

**Step-by-step calibration:**
1. Hold up a pink ball in front of the camera
2. Press '1' to select pink color calibration
3. Click directly on the pink ball in the video feed
4. Repeat for orange ('2'), green ('3'), and yellow ('4') balls
5. Press 's' to save your calibration
6. Press 'q' to exit and start tracking

**Pro Tips:**
- Use good, consistent lighting for best results
- Click on the center of the ball for optimal color sampling
- The system automatically handles HSV wrap-around for pink/orange colors
- Calibrate with the same lighting conditions you'll use for tracking
- The system intelligently merges nearby detections to handle finger occlusion

**Recommended Color Combinations:**
- **3-ball juggling**: Yellow, Pink, Green (best separation)
- **4-ball juggling**: All colors work, but pink/orange may have minor cross-detection
- **Best separation**: Avoid using pink and orange together if possible

**Manual Method: Edit Settings File**
You can also manually edit the `ball_settings.json` file:
```json
{
    "pink": {
        "min_hsv": [150, 150, 90],
        "max_hsv": [170, 255, 255]
    },
    "orange": {
        "min_hsv": [5, 150, 120],
        "max_hsv": [15, 255, 255]
    },
    "green": {
        "min_hsv": [45, 120, 70],
        "max_hsv": [75, 255, 255]
    },
    "yellow": {
        "min_hsv": [25, 120, 100],
        "max_hsv": [35, 255, 255]
    }
}
```

**HSV Parameter Guide:**
- **Hue (H)**: 0-180, represents the color (0=red, 60=green, 120=blue)
- **Saturation (S)**: 0-255, color intensity (0=gray, 255=vivid)
- **Value (V)**: 0-255, brightness (0=black, 255=bright)

### Performance Tuning
- **Minimum contour area**: Adjust `MIN_CONTOUR_AREA` to filter noise
- **Maximum depth**: Modify `MAX_DEPTH` to limit detection range
- **Frame rate**: The application automatically tries 90 FPS, then falls back to 60 FPS

## Troubleshooting

### Common Issues

1. **"No RealSense device found"**
   - Ensure camera is connected to USB 3.0 port
   - Check that RealSense SDK is properly installed
   - Try different USB ports

2. **"OpenCV not found" during build**
   - Verify OpenCV installation path
   - On Windows, ensure OpenCV is in `C:\opencv` or update CMakeLists.txt
   - On Linux, install `libopencv-dev` package

3. **Low frame rate or performance issues**
   - Close other applications using the camera
   - Ensure adequate lighting for color detection
   - Check CPU usage and consider lowering resolution if needed

4. **Inaccurate color detection**
   - Run calibration mode: `./build/bin/ball_tracker calibrate`
   - Ensure good, consistent lighting conditions
   - Consider the color and material of your juggling balls
   - Check that `ball_settings.json` exists and contains your calibrated values

### Debug Mode
To see additional debug information, the application outputs status messages to stderr:
```bash
./ball_tracker 2>debug.log  # Redirect debug info to file
```

## Technical Details

### Architecture
- **Camera Interface**: Intel RealSense SDK 2.0 for depth and color streaming
- **Computer Vision**: OpenCV 4.x for image processing and color detection
- **Color Space**: HSV for robust color detection under varying lighting
- **Settings Management**: nlohmann/json for persistent calibration storage
- **3D Calculation**: RealSense intrinsic calibration for pixel-to-world coordinate conversion
- **Memory Management**: Explicit OpenCV matrix release to prevent memory leaks
- **Dual Mode Design**: Separate calibration and tracking modes for optimal workflow

### Performance Optimizations
- Compiler optimizations enabled (`-O3` for GCC/Clang, `/O2` for MSVC)
- AVX2 instructions when available
- Efficient contour processing with area filtering
- Aligned depth frames for accurate 3D mapping
- Minimal memory allocations in the main loop

### Coordinate System
The output coordinates follow the RealSense camera coordinate system:
- **X**: Right (positive) / Left (negative)
- **Y**: Down (positive) / Up (negative)  
- **Z**: Forward/Away from camera (positive)
- **Units**: Meters

## License

This project is provided as-is for educational and development purposes.

## Contributing

When making changes:
1. Test both calibration and tracking modes with actual RealSense hardware
2. Verify performance under various lighting conditions
3. Test settings save/load functionality
4. Update this README with timestamp: 2025-08-16 16:04:00 UTC
5. Ensure backward compatibility with the output format