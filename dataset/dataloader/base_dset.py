import glob
import os
import pickle
import random
import numpy as np
import cv2
import torch


class BaseLoader(torch.utils.data.Dataset):
    def __init__(self, triplets, transform=None, resz=None):
        self.triplets = triplets
        self.transform = transform
        self.resz = resz

    def __getitem__(self, index):
        traj1_pth, traj2_pth, traj3_pth = self.triplets[index]
        traj1 = self.read_traj(traj1_pth)
        traj2 = self.read_traj(traj2_pth)
        traj3 = self.read_traj(traj3_pth)

        traj1_depth = torch.FloatTensor([self.resize_img(tran[0][0]) for tran in traj1] + [self.resize_img(traj1[-1][3][0])])
        traj2_depth = torch.FloatTensor([self.resize_img(tran[0][0]) for tran in traj2] + [self.resize_img(traj2[-1][3][0])])
        traj3_depth = torch.FloatTensor([self.resize_img(tran[0][0]) for tran in traj3] + [self.resize_img(traj3[-1][3][0])])


        traj1_states = torch.FloatTensor([self.extract_state_from_img(tran[0], 14, relative=True) for tran in traj1] + [self.extract_state_from_img(traj1[-1][3], 14, relative=True)])
        traj2_states = torch.FloatTensor([self.extract_state_from_img(tran[0], 14, relative=True) for tran in traj2] + [self.extract_state_from_img(traj2[-1][3], 14, relative=True)])
        traj3_states = torch.FloatTensor([self.extract_state_from_img(tran[0], 14, relative=True) for tran in traj3] + [self.extract_state_from_img(traj3[-1][3], 14, relative=True)])

        action_dim = len(traj1[0][1])
        traj1_actions = torch.FloatTensor([[0.0] * action_dim] + [tran[1] for tran in traj1])
        traj2_actions = torch.FloatTensor([[0.0] * action_dim] + [tran[1] for tran in traj2])
        traj3_actions = torch.FloatTensor([[0.0] * action_dim] + [tran[1] for tran in traj3])

        # Add dumy action
        assert len(traj1_actions) == len(traj1_states) == len(traj1_depth)
        assert len(traj2_actions) == len(traj2_states) == len(traj2_depth)
        assert len(traj3_actions) == len(traj3_states) == len(traj3_depth)

        # if self.transform is not None:
        #     traj1 = self.transform(traj1)
        #     traj2 = self.transform(traj2)
        #     traj3 = self.transform(traj3)

        return traj1_depth, traj2_depth, traj3_depth, traj1_states, traj2_states ,traj3_states, traj1_actions, traj2_actions, traj3_actions
        # return [traj1_depth, traj1_states, traj1_actions], [traj2_depth, traj2_states, traj2_actions], [traj3_depth, traj3_states, traj3_actions]

    def __len__(self):
        return len(self.triplets)

    def resize_img(self, img, resz=None):
        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX) # Normalize the pixels to values in 0-1

        img = cv2.resize(img, dsize=self.resz) if self.resz is not None else img
        return img

    def read_traj(self, file):
        with open(file, "rb") as f:
            d = pickle.load(f, encoding="latin1")
            # demos.extend(d[0])
            return d[0]

    def refine_traj_data(self, traj_data):
        new_traj_data = []
        old_action = [0.0] * 7
        for tran in traj_data:
            new_action = tran[1]
            new_traj_data.append([tran[0], old_action])
            old_action = new_action

    def resize_traj_obs(self, traj_data, H, W):
        traj_data = [[cv2.resize(trans[0], (H, W)), trans[1], cv2.resize(trans[2], (H, W))] for trans in traj_data]
        return traj_data

    def extract_state_from_img(self, img, num_states, c=-1, relative=False):
        num_dims = len(img.shape)
        assert num_dims == 3 or num_dims == 4, "input is not an image"
        state = img[:, c, :, :].flatten()[:num_states] if num_dims == 4 else img[c, :, :].flatten()[
                                                                                :num_states]

        if relative:
            state = np.concatenate((state[:6]-state[6:12], state[12:]))

        return state

class BaseDset(object):

    def __init__(self):
        self.__base_path = ""

        self.__train_set = {}
        self.__test_set = {}
        self.__train_keys = []
        self.__test_keys = []

    def load(self, base_path):
        self.__base_path = base_path
        train_dir = os.path.join(self.__base_path, 'train')
        test_dir = os.path.join(self.__base_path, 'test')

        self.__train_set = {}
        self.__test_set = {}
        self.__train_keys = []
        self.__test_keys = []

        for class_id in os.listdir(train_dir):
            class_dir = os.path.join(train_dir, class_id)
            self.__train_set[class_id] = []
            self.__train_keys.append(class_id)
            for traj_name in os.listdir(class_dir):
                traj_path = glob.glob(os.path.join(class_dir, traj_name+ '/*.pickle'))
                for p in traj_path:
                    self.__train_set[class_id].append(p)

        for class_id in os.listdir(test_dir):
            class_dir = os.path.join(test_dir, class_id)
            self.__test_set[class_id] = []
            self.__test_keys.append(class_id)
            for traj_name in os.listdir(class_dir):
                traj_path = glob.glob(os.path.join(class_dir, traj_name + '/*.pickle'))
                for p in traj_path:
                    self.__test_set[class_id].append(p)

        return len(self.__train_keys), len(self.__test_keys)

    def getTriplet(self, split='train'):
        if split == 'train':
            dataset = self.__train_set
            keys = self.__train_keys
        else:
            dataset = self.__test_set
            keys = self.__test_keys

        pos_idx = 0
        neg_idx = 0
        pos_anchor_traj_idx = 0
        pos_traj_idx = 0
        neg_traj_idx = 0

        pos_idx = random.randint(0, len(keys) - 1)
        while True:
            neg_idx = random.randint(0, len(keys) - 1)
            if pos_idx != neg_idx:
                break

        pos_anchor_traj_idx = random.randint(0, len(dataset[keys[pos_idx]]) - 1)
        while True:
            pos_traj_idx = random.randint(0, len(dataset[keys[pos_idx]]) - 1)
            if pos_anchor_traj_idx != pos_traj_idx:
                break

        neg_traj_idx = random.randint(0, len(dataset[keys[neg_idx]]) - 1)

        pos_anchor_traj = dataset[keys[pos_idx]][pos_anchor_traj_idx]
        pos_traj = dataset[keys[pos_idx]][pos_traj_idx]
        neg_traj = dataset[keys[neg_idx]][neg_traj_idx]

        return pos_anchor_traj, pos_traj, neg_traj
