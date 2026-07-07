"""
Test script for AirSim drone in ForestSimulation environment
Make sure the Unreal Engine simulation is running before executing this script
"""

import airsim
import time

def main():
    print("Connecting to AirSim...")
    
    # Connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✓ Connected to AirSim")
    
    # Enable API control
    client.enableApiControl(True)
    print("✓ API control enabled")
    
    # Arm the drone
    client.armDisarm(True)
    print("✓ Drone armed")
    
    # Takeoff
    print("\nTaking off...")
    client.takeoffAsync().join()
    print("✓ Takeoff complete")
    
    # Hover for 2 seconds
    time.sleep(2)
    
    # Fly in a square pattern
    print("\nFlying in a square pattern...")
    speed = 5
    
    # Forward
    print("  → Moving forward")
    client.moveToPositionAsync(10, 0, -10, speed).join()
    time.sleep(1)
    
    # Right
    print("  → Moving right")
    client.moveToPositionAsync(10, 10, -10, speed).join()
    time.sleep(1)
    
    # Backward
    print("  → Moving backward")
    client.moveToPositionAsync(0, 10, -10, speed).join()
    time.sleep(1)
    
    # Left (back to start)
    print("  → Moving left")
    client.moveToPositionAsync(0, 0, -10, speed).join()
    time.sleep(1)
    
    print("✓ Square pattern complete")
    
    # Take a photo
    print("\nCapturing image...")
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])
    
    if responses:
        response = responses[0]
        filename = "forest_capture.png"
        airsim.write_file(filename, response.image_data_uint8)
        print(f"✓ Image saved as {filename}")
    
    # Land
    print("\nLanding...")
    client.landAsync().join()
    print("✓ Landed")
    
    # Disarm
    client.armDisarm(False)
    print("✓ Drone disarmed")
    
    # Disable API control
    client.enableApiControl(False)
    print("✓ API control disabled")
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. Unreal Engine with ForestSim is running")
        print("2. AirSim plugin is loaded")
        print("3. settings.json is configured for Multirotor mode")
