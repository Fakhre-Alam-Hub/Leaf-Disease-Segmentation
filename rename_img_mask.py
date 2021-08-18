"""
This module renames the filename of given folder
"""

def imagefile_rename(path):
    """
    Function to rename multiple files in image folder

    Parameters
    ----------
    path: path of iamge folder
    """

    os.chdir(path) 

    for i, file in enumerate(sorted(os.listdir())):
        file_name, file_ext = os.path.splitext(file)
        i = str(i).zfill(5)

        new_name = '{}{}'.format(i, file_ext)

        os.rename(file, new_name)

    return


def maskfile_rename(path):
    """
    Function to rename multiple files in mask folder

    Parameters
    ----------
    path: path of mask folder

    """
    os.chdir(path) 

    for i, file in enumerate(sorted(os.listdir())):
        file_name, file_ext = os.path.splitext(file)
        i = str(i).zfill(5)

        new_name = '{}{}'.format(i, file_ext)

        os.rename(file, new_name)

    return

# Driver Code
if __name__ == '__main__':
    import os

    img_path = r"C:\project\Leaf Disease Segmentation\Modular Code\data\images"
    mask_path = r"C:\project\Leaf Disease Segmentation\Modular Code\data\masks"
      
    # Calling imagefile_rename() function
    imagefile_rename(img_path)
    print("Image file rename done!")

    # Calling maskfile_rename() function
    maskfile_rename(mask_path)
    print("Masks file rename done!")
