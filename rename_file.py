import os

class rename:
	'''
	A class to rename the files of the given base folder.

	[
	Example: base_dir is 'Experiment' for folder structure given below:
	Experiment:
		->data:
			->images: img1, img2, img3, ...
			->masks: mask1, mask2, mask3, ...
	]
	...

    Attributes
    ----------
    BASE_DIR : str
        path of the base folder

    Methods
    -------
    imageFile_rename():
        Renames the files of folder named images.
    maskFile_rename():
        Renames the files of folder named masks.
	'''

	def __init__(self, BASE_DIR):
		'''
		Constructs all the necessary attributes for the rename object.

        Parameters
        ----------
            BASE_DIR : str
                path of the base folder
		'''
		self.BASE_DIR = BASE_DIR


	def imageFile_rename(self):
		'''
		Rename the file name in folder named images.

        Parameters
        ----------
        None

        Returns
        -------
        None
		'''

		# Change working directory to folder which contains images
		os.chdir(os.path.join(self.BASE_DIR, 'data', 'images')) 

		for i, file in enumerate(sorted(os.listdir())):
			file_name, file_ext = os.path.splitext(file)
			i = str(i).zfill(5)
			new_name = '{}{}'.format(i, file_ext)
			os.rename(file, new_name)

		print("Images file rename done!")

	def maskFile_rename(self):
		'''
		Rename the file name in folder named masks.

        Parameters
        ----------
        None

        Returns
        -------
        None
		'''

		# Change working directory to folder which contains masks
		os.chdir(os.path.join(self.BASE_DIR, 'data', 'masks')) 

		for i, file in enumerate(sorted(os.listdir())):
			file_name, file_ext = os.path.splitext(file)
			i = str(i).zfill(5)
			new_name = '{}{}'.format(i, file_ext)
			os.rename(file, new_name)

		print("Masks file rename done!")


# Driver Code
if __name__ == '__main__':
    base_dir = r"C:\project\Leaf Disease Segmentation\Experiment"

	# creating object of rename class
    obj = rename(base_dir)

    # rename files of images folder
    obj.imageFile_rename()

    # rename files of masks folder
    obj.maskFile_rename()