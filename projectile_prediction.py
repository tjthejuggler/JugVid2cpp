import numpy as np
import matplotlib.pyplot as plt

# --- 1. Your Raw Data ---
raw_data = """
green,-0.174779,0.310473,0.875000
green,-0.197755,0.276975,1.149000
green,-0.183988,0.131164,1.130000
green,-0.099680,-0.057856,0.757000
green,-0.085409,-0.122794,0.746000
green,-0.068675,-0.180552,0.749000
green,-0.058880,-0.234053,0.754000
green,-0.044283,-0.275377,0.752000
green,-0.025968,-0.292817,0.759000
green,0.070614,-0.282965,0.753000
green,0.093468,-0.250757,0.753000
green,0.112700,-0.193412,0.754000
green,0.133344,-0.123582,0.760000
green,0.150202,-0.035685,0.740000
green,0.312382,0.308566,1.136000
green,0.246860,0.325830,0.865000
green,0.219232,0.297152,0.770000
"""

# --- 2. Configuration ---
VERTICAL_AXIS = 'Y'
NUM_POINTS_IN_SEGMENT = 5
PREDICTION_STEPS = 50 
G = 9.81

# --- Core Physics Functions ---

def parse_data(data_string):
    lines = data_string.strip().split('\n')
    points = [list(map(float, line.split(',')[1:])) for line in lines]
    return np.array(points)

def get_axis_indices(vertical_axis_char):
    if vertical_axis_char.upper() == 'Y': return 1, [0, 2]
    elif vertical_axis_char.upper() == 'Z': return 2, [0, 1]
    raise ValueError("VERTICAL_AXIS must be 'Y' or 'Z'")

def estimate_initial_conditions(points_segment, v_idx, h_indices):
    t_indices = np.arange(len(points_segment))
    y_coords = points_segment[:, v_idx]
    a, b, c = np.polyfit(t_indices, y_coords, 2)
    
    # Check if the fit is valid, but DON'T raise an error.
    # We will handle the invalid case visually.
    is_valid_fit = (a < 0)
    
    # To prevent a math error (sqrt of a negative), we use abs().
    # The 'is_valid_fit' flag will tell us if the result is physically meaningful.
    dt = np.sqrt(abs(a) / (0.5 * G))
    
    p0, v0 = np.zeros(3), np.zeros(3)
    p0[v_idx], v0[v_idx] = c, b / dt
    for i in h_indices:
        slope, intercept = np.polyfit(t_indices, points_segment[:, i], 1)
        p0[i], v0[i] = intercept, slope / dt
        
    return p0, v0, dt, is_valid_fit

def predict_trajectory(p0, v0, dt, num_steps, v_idx, is_valid):
    time_array = np.arange(num_steps) * dt
    trajectory = np.zeros((num_steps, 3))
    
    # Decide the sign of gravity based on the fit.
    # For invalid (upward) fits, we flip gravity to show the incorrect parabola.
    gravity_sign = -1.0 if is_valid else 1.0

    for i in [0, 2]:
        if i != v_idx: trajectory[:, i] = p0[i] + v0[i] * time_array
    trajectory[:, v_idx] = p0[v_idx] + v0[v_idx] * time_array + 0.5 * (gravity_sign * G) * (time_array**2)
    return trajectory, time_array

# --- Plotting Function for a Single Prediction ---

def plot_single_prediction(original_data, fit_points, start_index, predicted_path, dt, pred_time, v_idx, h_indices, is_valid):
    original_time = np.arange(len(original_data)) * dt
    fit_points_time = (np.arange(len(fit_points)) + start_index) * dt
    predicted_time_offset = pred_time + start_index * dt

    h_idx1, h_idx2 = h_indices[0], h_indices[1]
    axis_labels = ['X', 'Y', 'Z']
    vert_label, horiz1_label, horiz2_label = axis_labels[v_idx], axis_labels[h_idx1], axis_labels[h_idx2]

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    title = f"Prediction Based on Points {start_index} to {start_index + NUM_POINTS_IN_SEGMENT - 1}"
    if not is_valid:
        title += "\n(INVALID FIT: Upward-Opening Parabola Detected)"
        fig.patch.set_facecolor('#ffdddd') # Add a red tint to invalid plots
    
    fig.suptitle(title, fontsize=16, weight='bold')

    def plot_ax(ax, x_pred, y_pred, x_orig, y_orig, x_fit, y_fit, title, xlabel, ylabel, equal_axis=False):
        ax.plot(x_orig, y_orig, 'bo', label='Original Data Points', alpha=0.4, markersize=5)
        # Use a different color for the prediction line if the fit is invalid
        line_color = 'g-' if is_valid else 'm--'
        ax.plot(x_pred, y_pred, line_color, label='Predicted Path', linewidth=2)
        ax.plot(x_fit, y_fit, 'rx', markersize=10, mew=2, label='Fit Segment')
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(True); ax.legend()
        if equal_axis: ax.axis('equal')

    plot_ax(axs[0, 0], predicted_time_offset, predicted_path[:, v_idx], original_time, original_data[:, v_idx], fit_points_time, fit_points[:, v_idx], 'Height vs. Time', 'Time (s)', f'{vert_label} (Height)')
    plot_ax(axs[0, 1], predicted_path[:, h_idx1], predicted_path[:, h_idx2], original_data[:, h_idx1], original_data[:, h_idx2], fit_points[:, h_idx1], fit_points[:, h_idx2], f'Top-Down View ({horiz2_label} vs. {horiz1_label})', f'{horiz1_label}', f'{horiz2_label}', equal_axis=True)
    plot_ax(axs[1, 0], predicted_time_offset, predicted_path[:, h_idx1], original_time, original_data[:, h_idx1], fit_points_time, fit_points[:, h_idx1], f'{horiz1_label} Position vs. Time', 'Time (s)', f'{horiz1_label}')
    plot_ax(axs[1, 1], predicted_time_offset, predicted_path[:, h_idx2], original_time, original_data[:, h_idx2], fit_points_time, fit_points[:, h_idx2], f'{horiz2_label} Position vs. Time', 'Time (s)', f'{horiz2_label}')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- Main Execution ---
if __name__ == "__main__":
    full_data = parse_data(raw_data)
    vertical_idx, horizontal_indices = get_axis_indices(VERTICAL_AXIS)
    
    num_segments = len(full_data) - NUM_POINTS_IN_SEGMENT + 1
    for start_index in range(num_segments):
        points_for_fitting = full_data[start_index : start_index + NUM_POINTS_IN_SEGMENT]
        print(f"--- Generating prediction for segment starting at index {start_index} ---")
        
        try:
            p0, v0, dt, is_valid = estimate_initial_conditions(points_for_fitting, vertical_idx, horizontal_indices)
            predicted_path, predicted_time = predict_trajectory(p0, v0, dt, PREDICTION_STEPS, vertical_idx, is_valid)
            plot_single_prediction(full_data, points_for_fitting, start_index, predicted_path, dt, predicted_time, vertical_idx, horizontal_indices, is_valid)
        except np.linalg.LinAlgError as e:
            print(f"Could not generate prediction due to a math error: {e}")
            continue

    print("\nDisplaying all generated plot windows...")
    plt.show()