import os
import zipfile
import shutil


def zip_and_delete_conv_frames(root_dir):
    """
    Traverse all subdirectories in the given root directory.
    If a "conv_frames" folder is found, zip it and save the zip file in the same subdirectory.
    After zipping, delete the "conv_frames" folder.
    """
    for subdir, dirs, files in os.walk(root_dir):
        # Check if 'conv_frames' folder exists in the current subdirectory
        conv_frames_path = os.path.join(subdir, 'conv_frames')
        if os.path.isdir(conv_frames_path):
            zip_filename = os.path.join(subdir, 'conv_frames.zip')

            # Create a zip file for the conv_frames folder
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(conv_frames_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, conv_frames_path)
                        zipf.write(file_path, arcname)

            print(f"Zipped {conv_frames_path} to {zip_filename}")

            # Delete the conv_frames folder after zipping
            shutil.rmtree(conv_frames_path)
            print(f"Deleted {conv_frames_path}")


# Example usage
root_directory = r'E:\Event_camera\DVS_AF_est\dataset\SYN'  # Replace with your root directory path
zip_and_delete_conv_frames(root_directory)
