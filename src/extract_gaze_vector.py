import time
import os
import matplotlib.pyplot as plt
# conda activate gazeml
# cd '/media/hn_97/VOLUME G/GazeML-org/src'
# python tello_control_updated.py
# python elg_run.py
# python extract_gaze_vector.py
# Define the file where the gaze vector is stored
gaze_vector_file = "gaze_vector.txt"

# Track the last modification time of the file
last_modified_time = None

def read_last_gaze_vector():
    """Reads the last gaze vector from the file"""
    try:
        # Check the file modification time
        current_modified_time = os.path.getmtime(gaze_vector_file)
        
        # If the file has been modified since the last read, read the latest gaze data
        if current_modified_time != last_modified_time:
            with open(gaze_vector_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    return last_line
        return None
    except FileNotFoundError:
        print("Gaze vector file not found!")
        return None

def classify_gaze(x, y):
    """Classify gaze direction with swapped and inverted axes."""
    threshold = 0.15

    # Swap and invert axes
    gaze_x = -y  # Horizontal direction (invert Y)
    gaze_y = -x  # Vertical direction (invert X)

    # Center box check
    if -threshold <= gaze_x <= threshold and -threshold <= gaze_y <= threshold:
        return "Center"

    dx = abs(gaze_x) - threshold
    dy = abs(gaze_y) - threshold

    if dx > dy:
        return "Left" if gaze_x < -threshold else "Right"
    else:
        return "Up" if gaze_y < -threshold else "Down"


def get_color_for_direction(direction):
    """Return a color based on the simplified gaze direction."""
    color_map = {
        "Center": "black",
        "Left": "orange",
        "Right": "cyan",
        "Up": "blue",
        "Down": "green"
    }
    return color_map.get(direction, "gray")  # Fallback to gray
def get_color_for_direction(direction):
    """Return a color based on the simplified gaze direction."""
    color_map = {
        "Center": "black",
        "Left": "orange",
        "Right": "cyan",
        "Up": "blue",
        "Down": "green"
    }
    # Return the color for the direction, defaulting to gray if not found
    return color_map.get(direction, "gray")  # Default color



def plot_gaze_vector(x_values, y_values, directions):
    """Update the real-time plot with color-coded gaze vectors and boundary lines."""
    plt.clf()

    # Plot each gaze point with its corresponding color
    for x, y, direction in zip(x_values, y_values, directions):
        color = get_color_for_direction(direction)
        
        # Ensure 'up' appears at the top of the plot and 'down' appears at the bottom
        label = direction if direction not in plt.gca().get_legend_handles_labels()[1] else ""
        plt.scatter(x, -y, color=color, label=label)  # Invert y for correct up/down positioning

    # Plot decision boundaries
    threshold = 0.15
    plt.axvline(x=threshold, color='gray', linestyle='--')
    plt.axvline(x=-threshold, color='gray', linestyle='--')
    plt.axhline(y=threshold, color='gray', linestyle='--')
    plt.axhline(y=-threshold, color='gray', linestyle='--')
    
    # Diagonal lines
    plt.plot([-1, 1], [-1, 1], color='gray', linestyle=':', linewidth=1)   # y = x
    plt.plot([-1, 1], [1, -1], color='gray', linestyle=':', linewidth=1)   # y = -x

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('Horizontal Gaze (x)')
    plt.ylabel('Vertical Gaze (y)')
    plt.title('Real-time Gaze Vector Tracking')
    plt.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.15, 1.0))
    plt.pause(0.1)


def main():
    print("Gaze Vector Extraction is running. Press Ctrl+C to stop.")
    
    x_values = []
    y_values = []
    directions = []

    plt.ion()
    plt.figure(figsize=(6, 6))

    try:
        while True:
            gaze_data = read_last_gaze_vector()

            if gaze_data:
                # Parse the gaze data
                x, y = map(float, gaze_data.split(","))
                print(f"Raw Gaze Vector (x, y) = ({x:.3f}, {y:.3f})")

                # Swap and invert axes for consistent interpretation and plotting
                gaze_x = -y  # Horizontal
                gaze_y = -x  # Vertical

                # Classify gaze using the original values
                gaze_direction = classify_gaze(x, y)
                print(f"Gaze Direction: {gaze_direction}")

                # Store transformed values for plotting
                x_values.append(gaze_x)
                y_values.append(gaze_y)
                directions.append(gaze_direction)

                # Update the real-time plot
                plot_gaze_vector(x_values, y_values, directions)

                # Update the last modified time to avoid re-reading the file
                global last_modified_time
                last_modified_time = os.path.getmtime(gaze_vector_file)

            else:
                print("Waiting for gaze data...")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting program gracefully.")
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main(


    )
