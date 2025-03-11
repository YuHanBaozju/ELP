import os
from davis_vis import elp_real_davis
# ANSI color codes
COLOR_GREEN = "\033[32m"
COLOR_RESET = "\033[0m"

FPS = 1


def process_fourth_level_subdirectories(root_dir):
    """
    Traverse the root directory and find all fourth-level subdirectories,
    calling the sim_davis function on each of them.
    """
    for subdir, dirs, files in os.walk(root_dir):
        relative_path = os.path.relpath(subdir, root_dir)
        level = relative_path.count(os.sep) + 1  # 计算相对于 root_dir 的层级数

        if level == 2:
            print(f'{COLOR_GREEN}{subdir}{COLOR_RESET}')
            ## ablation_on = 'none' or 'filter' or 'laplacian'
            elp_real_davis(subdir, vis=False,ablation_on='none')


# Example usage
root_directory = r'F:\Event_camera\DVS_AF_est\dataset\DAVIS346\dataset'  # Replace with the actual root directory path
process_fourth_level_subdirectories(root_directory)
