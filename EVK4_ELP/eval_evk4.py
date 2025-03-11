import os
from evk4_vis import elp_real_evk4
# from ablation_e2vid_evk4_vis import elp_real_evk4
# from ablation_evk4_vis import elp_ablation_evk4

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
        # 计算当前目录的层级
        relative_path = os.path.relpath(subdir, root_dir)
        level = relative_path.count(os.sep) + 1  # 计算相对于 root_dir 的层级数

        # 如果是第四层目录，调用 sim_davis
        if level == 2:
            print(f'{COLOR_GREEN}{subdir}{COLOR_RESET}')
            elp_real_evk4(subdir, vis=False)
            # elp_ablation_evk4(subdir, vis=False, abalation_on='filter')


# Example usage
root_directory = r'F:\Event_camera\DVS_AF_est\dataset\EVK4\dataset'  # Replace with the actual root directory path
process_fourth_level_subdirectories(root_directory)
