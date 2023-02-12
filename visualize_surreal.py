import numpy as np
from matplotlib import pyplot as plt

indices_to_arrange = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
indices_to_select = [0, 2, 5, 8, 1, 4, 7, 12, 16, 18, 20, 17, 19, 21]

current_video = 0
videos_marked = []

EDGES = [[0, 1], [0, 4], [0, 7], [1, 2], [2, 3], [4, 5], [5, 6], [7, 8], [8, 9], [9, 10], [7, 11], [11, 12], [12, 13]]
lefts = [4, 5, 6, 8, 9, 10]
rights = [1, 2, 3, 11, 12, 13]

def draw_skeleton(pose, ax, is_3d=False, jnts_14=True):

    col_right = 'b'
    col_left = 'r'

    if is_3d:
        ax.scatter(pose[:, 0], pose[:, 1], zs=pose[:, 2], color='k')
    else:
        ax.scatter(pose[:, 0], pose[:, 1], color='k')

    for u, v in EDGES:
        col_to_use = 'k'

        if u in lefts and v in lefts:
            col_to_use = col_left
        elif u in rights and v in rights:
            col_to_use = col_right

        if is_3d:
            ax.plot([pose[u, 0], pose[v, 0]], [pose[u, 1], pose[v, 1]], zs=[pose[u, 2], pose[v, 2]], color=col_to_use)
        else:
            ax.plot([pose[u, 0], pose[v, 0]], [pose[u, 1], pose[v, 1]], color=col_to_use)

def on_press(event):

    if event.key == 'x':
        videos_marked.append(current_video)

        np.savez_compressed('marked_vids', vids=videos_marked)

        print(f"Video {current_video} is marked")


if __name__ == "__main__":

    data = np.load("surreal_train.npz", allow_pickle=True)

    video_2d = data['video_2d'][:, :, indices_to_arrange, :][:, :, indices_to_select, :]
    frame_nums = data['frame_num']

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    fig.canvas.mpl_connect('key_press_event', on_press)

    for i in range(video_2d.shape[0]):

        current_video = i

        seq_length = int(frame_nums[i])

        print(f"Processing video {i}")

        for j in range(seq_length):

            jnts_2d = video_2d[i, j]

            draw_skeleton(jnts_2d, ax)

            ax.set_xlim((0, 320))
            ax.set_ylim((240, 0))

            plt.draw()
            plt.pause(1e-2)
            ax.clear()

    print(videos_marked)
