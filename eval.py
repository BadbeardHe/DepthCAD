import os
import cv2
import argparse
import numpy as np


def bin_loader(path):
    if not os.path.exists(path):
        raise FileNotFoundError

    shape = (424, 512)
    data = np.fromfile(path, dtype=np.float32).reshape(shape)
    data = np.nan_to_num(data, 0)
    
    return data


def loss(pred, ideal):
    t_valid = 0
    t_max = 9
    
    pred[pred >= t_max] = 0
    pred[pred < t_valid] = 0

    # For numerical stability
    mask = (ideal > t_valid) & (ideal < t_max) 
    num_valid = mask.sum()

    pred = pred[mask]
    ideal = ideal[mask]

    # MAE
    diff = pred - ideal
    diff_abs = np.abs(diff)
    mae = diff_abs.sum() / (num_valid + 1e-8)

    # Rel
    rel = diff_abs / (ideal + 1e-8)
    rel = rel.sum() / (num_valid + 1e-8)

    # delta
    r1 = ideal / (pred + 1e-8)
    r2 = pred / (ideal + 1e-8)
    ratio = np.maximum(r1, r2)

    del_1 = (ratio < 1.25).astype('float32')
    del_2 = (ratio < 1.25 ** 2).astype('float32')
    del_3 = (ratio < 1.25 ** 3).astype('float32')

    del_1 = del_1.sum() / (num_valid + 1e-8)
    del_2 = del_2.sum() / (num_valid + 1e-8)
    del_3 = del_3.sum() / (num_valid + 1e-8)

    result = [mae, rel, del_1, del_2, del_3]
    return result


def eval(val_list_path, pred_dir, out_dir):
    loss_mae = []
    loss_rel = []
    loss_del_1 = []
    loss_del_2 = []
    loss_del_3 = []

    with open(val_list_path, 'r') as f:
        idxs = f.readlines()
    idxs = [idx.strip() for idx in idxs]

    ideal_root = "/Path/to/Ideal/depth/path"

    for idx in idxs:
        ideal = bin_loader(os.path.join(ideal_root, idx)) / 1e3
        pred = np.load(f"{pred_dir}/{idx}.npy")
        pred = np.nan_to_num(pred, 0)

        loss_list = loss(pred, ideal)

        loss_mae.append(loss_list[0])
        loss_rel.append(loss_list[1])
        loss_del_1.append(loss_list[2])
        loss_del_2.append(loss_list[3])
        loss_del_3.append(loss_list[4])

    # save results
    mae_mean = sum(loss_mae) / len(loss_mae)
    rel_mean = sum(loss_rel) / len(loss_rel)
    del_1_mean = sum(loss_del_1) / len(loss_del_1)
    del_2_mean = sum(loss_del_2) / len(loss_del_2)
    del_3_mean = sum(loss_del_3) / len(loss_del_3)

    print("mae_mean, rel_mean, del_1_mean, del_2_mean, del_3_mean: {0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f}".format(mae_mean, rel_mean, del_1_mean, del_2_mean, del_3_mean))

    with open(f"{out_dir}/result_metrics.txt", "w") as text_file:
        text_file.write("mae_mean, rel_mean, del_1_mean, del_2_mean, del_3_mean: {0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f}".format(mae_mean, rel_mean, del_1_mean, del_2_mean, del_3_mean))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_list_path",
        type=str,
    )
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--pred_dir", type=str, default=None)
    
    args = parser.parse_args()
    test_list_path = args.test_list_path
    out_dir = args.out_dir
    pred_dir = args.pred_dir

    eval(test_list_path, out_dir=out_dir)