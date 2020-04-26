from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            custom_point (list): which points to train on
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.key_pts_frame.dropna(inplace=True)
        self.key_pts_frame.reset_index(drop=True, inplace=True)
        self.transform = transform

    def __len__(self):
        #######################################################################
        # TODO:                                                               #
        # Return the length of the dataset                                    #
        #######################################################################

        return self.key_pts_frame.shape[0]

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

    def __getitem__(self, idx):
        sample = {'image': None, 'keypoints': None}
        #######################################################################
        # TODO:                                                               #
        # Return the idx sample in the Dataset. A sample should be a          #
        # dictionary where the key, value should be like                      #
        #        {'image': image of shape [C, H, W],                          #
        #         'keypoints': keypoints of shape [num_keypoints, 2]}         #
        #######################################################################
        sample_orig = self.key_pts_frame.iloc[idx]
        #print(sample_orig)
        img_str = sample_orig['Image']

        img_int = np.array([int(s) for s in img_str.split(' ')])
        img_int = np.expand_dims(img_int.reshape(96,96),axis=0)
        sample['image'] = img_int
        #print(img_int.shape)

        landmarks = self.key_pts_frame.iloc[idx, :-1].values
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample['keypoints'] = landmarks
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        if self.transform:
            sample = self.transform(sample)

        return sample
