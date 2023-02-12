import sys
import os
import numpy as np
from tqdm import tqdm

indices_to_arrange = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
indices_to_select = [0, 1, 4, 7, 2, 5, 8, 6, 12, 15, 16, 18, 20, 17, 19, 21]

if __name__ == "__main__":

    assert len(sys.argv) == 2

    fname = "surreal_{f}.npz".format(f=sys.argv[-1])

    assert os.path.exists(fname)

    data = np.load(fname, allow_pickle=True)

    video_2d = data['video_2d']
    video_3d = data['video_3d']
    frame_nums = data['frame_num']

    data_3d = []
    data_2d = []

    for i in tqdm(range(video_2d.shape[0])):
        seq_length = int(frame_nums[i])

        data_2d.append(video_2d[i, :seq_length, :].reshape(-1, 24, 2))
        data_3d.append(video_3d[i, :seq_length, :].reshape(-1, 24, 3))

    data_3d = np.vstack(data_3d)
    data_2d = np.vstack(data_2d)

    print(data_3d.shape)
    print(data_2d.shape)

    np.savez_compressed("surreal_{f}_compiled".format(f=sys.argv[-1]), data_3d=data_3d, data_2d=data_2d)