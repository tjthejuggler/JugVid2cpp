import numpy as np
import matplotlib.pyplot as plt

# --- 1. Your Raw Data ---
raw_data = """
green,-0.079242,0.491529,1.383000
green,-0.066179,0.410496,1.155000
green,-0.053516,0.283122,1.037000
green,-0.040393,0.182026,1.029000
green,-0.055552,0.164670,1.459000
green,-0.042050,0.031826,1.032000
green,-0.043517,0.032936,1.068000
green,-0.035585,-0.035078,1.037000
green,-0.035859,-0.035349,1.045000
green,-0.030758,-0.099990,1.032000
green,-0.032159,-0.104544,1.079000
green,-0.025178,-0.154221,1.043000
green,-0.018013,-0.197846,1.022000
green,-0.018189,-0.199782,1.032000
green,-0.010802,-0.235465,1.017000
green,-0.010781,-0.235002,1.015000
green,-0.000242,-0.270115,1.029000
green,-0.000238,-0.266440,1.015000
green,0.007257,-0.290982,1.022000
green,0.007321,-0.293544,1.031000
green,0.012538,-0.302976,1.017000
green,0.021716,-0.307834,1.020000
green,0.021950,-0.311154,1.031000
green,0.029135,-0.303579,1.027000
green,0.028908,-0.301214,1.019000
green,0.038295,-0.286850,1.024000
green,0.037921,-0.284049,1.014000
green,0.050088,-0.262833,1.025000
green,0.050186,-0.263346,1.027000
green,0.057688,-0.225091,1.012000
green,0.067053,-0.182534,1.022000
green,0.065938,-0.179498,1.005000
green,0.077687,-0.128809,1.017000
green,0.077076,-0.127795,1.009000
green,0.084269,-0.065242,1.015000
green,0.083771,-0.064856,1.009000
green,0.095734,0.005689,1.025000
green,0.097696,0.005806,1.046000
green,0.146202,0.314083,1.113000
green,0.152224,0.380737,1.054000
green,0.195407,0.488745,1.353000
"""

# --- 2. Configuration ---
VERTICAL_AXIS = 'Y'
NUM_POINTS_IN_SEGMENT = 10
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