from matplotlib import pyplot as plt

from VGGClass import VGGClass
from config import device
from loss_functions import uncertainty_MSE_loss
from dataset import StaticHedgedDataset, StaticHedgedQuintupletDataset
from utils import load_checkpoint_model, plot_graph, get_logger
from torch.utils.data import DataLoader

from inference_single_frame_u_net import test_MSE, generate_MSE_evaluation_report
from inference_recurrent_uncertainty_u_net import test_recurrent_uncertainty, generate_uncertainty_evaluation_report, \
    generate_recurrent_results

import numpy as np
import torch.nn.functional
import os


def inference_MSE_unet(checkpoints, path):
    result_by_density = dict()

    logger = get_logger()

    # prediction model
    model = load_checkpoint_model(checkpoints)

    # classification model
    vgg_model = VGGClass().to(device)
    vgg_model.eval()

    criterion = torch.nn.MSELoss()

    # density range
    full_density_range = np.arange(0.0, 0.9, 0.1)

    for i in range(len(full_density_range)):
        density_range = full_density_range[i:i + 1]
        print("working on density range", density_range)
        test_dataset = StaticHedgedDataset(os.path.join('..', 'data', 'synthetic_data', 'test'), density_range)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4)
        result_by_density[density_range[0]] = test_MSE(test_loader, model, vgg_model, criterion, density_range[0],
                                                       logger, path)

    generate_MSE_evaluation_report(result_by_density, path)


def inference_real_data(checkpoints, path):
    # prediction model
    model = load_checkpoint_model(checkpoints[0])

    # classification model
    vgg_model = VGGClass().to(device)
    vgg_model.eval()

    density_range = [0.]
    test_dataset = StaticHedgedQuintupletDataset(os.path.join('..', 'data', 'synthetic_data', 'test'), density_range)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=0)
    preprocess_model = load_checkpoint_model(checkpoints[1])
    generate_recurrent_results(test_loader, model, preprocess_model, density_range[0], path)


def inference_recurrent_uncertainty_unet(checkpoints, path):
    result_by_density_1_frame = dict()
    result_by_density_2_frames = dict()
    result_by_density_3_frames = dict()
    result_by_density_5_frames = dict()
    logger = get_logger()

    # prediction model
    model = load_checkpoint_model(checkpoints[0])

    # classification model
    vgg_model = VGGClass().to(device)
    vgg_model.eval()

    criterion = uncertainty_MSE_loss

    # density range
    full_density_range = np.arange(0.0, 0.9, 0.1)

    for i in range(len(full_density_range)):
        density_range = full_density_range[i:i + 1]

        test_dataset = StaticHedgedQuintupletDataset(os.path.join('..', 'data', 'synthetic_data', 'test'),
                                                     density_range)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=0)
        preprocess_model = load_checkpoint_model(checkpoints[1])
        result_1_frame, result_2_frames, result_3_frames, result_5_frames = test_recurrent_uncertainty(
            test_loader, model,
            preprocess_model, vgg_model,
            criterion, density_range[0],
            logger, path)
        result_by_density_1_frame[density_range[0]] = result_1_frame
        result_by_density_2_frames[density_range[0]] = result_2_frames
        result_by_density_3_frames[density_range[0]] = result_3_frames
        result_by_density_5_frames[density_range[0]] = result_5_frames

    path1 = os.path.join(path, "Uncertainty U-Net")
    os.makedirs(path1, exist_ok=True)
    generate_uncertainty_evaluation_report(result_by_density_1_frame, path1)
    path2 = os.path.join(path, "Recurrent Uncertainty U-Net (2frames)")
    os.makedirs(path2, exist_ok=True)
    generate_uncertainty_evaluation_report(result_by_density_2_frames, path2)
    path3 = os.path.join(path, "Recurrent Uncertainty U-Net (3frames)")
    os.makedirs(path3, exist_ok=True)
    generate_uncertainty_evaluation_report(result_by_density_3_frames, path3)
    path4 = os.path.join(path, "Recurrent Uncertainty U-Net (5frames)")
    os.makedirs(path4, exist_ok=True)
    generate_uncertainty_evaluation_report(result_by_density_5_frames, path4)


def comparison(title, data_paths, data_labels, y_axis, path):
    densities = np.arange(0.0, 0.9, 0.1)

    x_axis = "density"
    y_data = []
    x_data = []

    for d in data_paths:
        y_data.append(np.load(d, allow_pickle=True))
        x_data.append(densities)

    plot_graph(title, x_axis, y_axis, x_data, y_data, data_labels, path)


def perform_comparisons_recurrent_models():
    # comparison between vgg recurrent models - top 1 with 2 frames
    root = os.path.join("screenshots", "report", "5_recurrent_results",
                        "recurrent_comparison")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "vgg_top_1_acc_recurrent_comparison(2-frame).png")
    comparison("top 1 vgg accuracy (2-frame)",
               ["screenshots/report/1_single_frame_uncertainty_u_net/top_1_model_acc.npy",
                "screenshots/report/2_recurrent_uncertainty_u_net/top_1_model_acc.npy",
                "screenshots/report/5_recurrent_results/ep300/2_frame_report/top_1_model_acc.npy",
                "screenshots/report/5_recurrent_results/ep400/2_frame_report/top_1_model_acc.npy",
                "screenshots/report/5_recurrent_results/ep400(1e-5)/Recurrent Uncertainty U-Net (2frames)/top_1_model_acc.npy",
                "screenshots/report/3_ground_truth/top_1_masked_acc.npy",
                "screenshots/report/3_ground_truth/top_1_ground_truth_acc.npy"],
               ["Single Frame Uncertainty U-Net", "Recurrent Uncertainty U-Net ep200(1e-3)",
                "Recurrent Uncertainty U-Net ep300(1e-4)", "Recurrent Uncertainty U-Net ep400(same density)",
                "Recurrent Uncertainty U-Net ep400(1e-5)", "hedged image", "Ground Truth"], "accuracy", path)

    # comparison between vgg recurrent models - top 1 with 3 frames
    path = os.path.join(root, "vgg_top_1_acc_recurrent_comparison(3-frame).png")
    comparison("top 1 vgg accuracy (3-frame)",
               ["screenshots/report/1_single_frame_uncertainty_u_net/top_1_model_acc.npy",
                "screenshots/report/2_recurrent_uncertainty_u_net_triplet/top_1_model_acc.npy",
                "screenshots/report/5_recurrent_results/ep300/3_frame_report/top_1_model_acc.npy",
                "screenshots/report/5_recurrent_results/ep400/3_frame_report/top_1_model_acc.npy",
                "screenshots/report/5_recurrent_results/ep400(1e-5)/Recurrent Uncertainty U-Net (3frames)/top_1_model_acc.npy",
                "screenshots/report/3_ground_truth/top_1_masked_acc.npy",
                "screenshots/report/3_ground_truth/top_1_ground_truth_acc.npy"],
               ["Single Frame Uncertainty U-Net", "Recurrent Uncertainty U-Net ep200(1e-3)",
                "Recurrent Uncertainty U-Net ep300(1e-4)", "Recurrent Uncertainty U-Net ep400(same density)",
                "Recurrent Uncertainty U-Net ep400(1e-5)", "hedged image", "Ground Truth"], "accuracy", path)

    # comparison between vgg recurrent models - top 1 with 5 frames
    path = os.path.join(root, "vgg_top_1_acc_recurrent_comparison(5-frame).png")
    comparison("top 1 vgg accuracy, (5-frame)",
               ["screenshots/report/1_single_frame_uncertainty_u_net/top_1_model_acc.npy",
                "screenshots/report/2_recurrent_uncertainty_u_net_quintuplet/top_1_model_acc.npy",
                "screenshots/report/5_recurrent_results/ep300/5_frame_report/top_1_model_acc.npy",
                "screenshots/report/5_recurrent_results/ep400/5_frame_report/top_1_model_acc.npy",
                "screenshots/report/5_recurrent_results/ep400(1e-5)/Recurrent Uncertainty U-Net (5frames)/top_1_model_acc.npy",
                "screenshots/report/3_ground_truth/top_1_masked_acc.npy",
                "screenshots/report/3_ground_truth/top_1_ground_truth_acc.npy"],
               ["Single Frame Uncertainty U-Net", "Recurrent Uncertainty U-Net ep200(1e-3)",
                "Recurrent Uncertainty U-Net ep300(1e-4)", "Recurrent Uncertainty U-Net ep400(same density)",
                "Recurrent Uncertainty U-Net ep400(1e-5)", "hedged image", "Ground Truth"], "accuracy", path)


def generate_inference_data():
    # MSE U-net - simple vs complex
    root = "screenshots/report"
    print("processing simple MSE u-net model")
    path = os.path.join(root, "0_single_frame_MSE_u_net", "simple")
    os.makedirs(path, exist_ok=True)
    inference_MSE_unet(
        "records/actual_results/0_MSE_UNet/old_model/0_simple_MSE_net/checkpoint/ep500/BEST_checkpoint.tar", path)

    print("processing complex MSE u-net model")
    path = os.path.join(root, "0_single_frame_MSE_u_net", "complex")
    os.makedirs(path, exist_ok=True)
    inference_MSE_unet(
        "records/actual_results/0_MSE_UNet/old_model/1_complex_MSE_net/checkpoint/ep500/BEST_checkpoint.tar", path)

    # Recurrent Uncertainty U-Net + Single frame Uncertainty U-Net
    print("procesesing recurrent uncertainty u-net model")
    path = os.path.join("screenshots", "report", "1_uncertainty_results")
    os.makedirs(path, exist_ok=True)
    inference_recurrent_uncertainty_unet([
        "records/actual_results/2_recurrent_frame_uncertainty_u_net/new_model/checkpoint/ep300(lr=1e-4)/BEST_checkpoint.tar",
        "records/actual_results/1_Uncertainty_UNet/new_model/single_frame_clamp_uncertainty_network/checkpoints/ep400(lr=3e-3)/BEST_checkpoint.tar"],
        path)


def generate_report_images():
    # 0: ground truth/hedged images only
    root = "screenshots/result_graphs"
    path = os.path.join(root, "0_top_1_vgg_ground_truth_and_masked_images")
    comparison("ground truth / hedged images top-1 VGG accuracy",
               ["screenshots/report/3_ground_truth/top_1_masked_acc.npy",
                "screenshots/report/3_ground_truth/top_1_ground_truth_acc.npy"],
               ["Hedged image", "Ground Truth"], "accuracy", path)

    path = os.path.join(root, "0_top_5_vgg_ground_truth_and_masked_images")
    comparison("ground truth / hedged images top-5 VGG accuracy",
               ["screenshots/report/3_ground_truth/top_5_masked_acc.npy",
                "screenshots/report/3_ground_truth/top_5_ground_truth_acc.npy"],
               ["Hedged image", "Ground Truth"], "accuracy", path)

    # 1: simple vs complex single frame mse u-net

    path = os.path.join(root, "1_top_1_vgg_simple_vs_complex.png")

    comparison("single frame MSE U-Net top-1 VGG accuracy",
               ["screenshots/report/0_single_frame_MSE_u_net/complex/top_1_model_acc.npy",
                "screenshots/report/0_single_frame_MSE_u_net/simple/top_1_model_acc.npy",
                "screenshots/report/3_ground_truth/top_1_masked_acc.npy",
                "screenshots/report/3_ground_truth/top_1_ground_truth_acc.npy"],
               ["Complex model", "Simple model", "Hedged image", "Ground Truth"], "accuracy", path)

    path = os.path.join(root, "1_top_5_vgg_simple_vs_complex.png")
    comparison("single frame MSE U-Net top-5 VGG accuracy",
               ["screenshots/report/0_single_frame_MSE_u_net/complex/top_5_model_acc.npy",
                "screenshots/report/0_single_frame_MSE_u_net/simple/top_5_model_acc.npy",
                "screenshots/report/3_ground_truth/top_5_masked_acc.npy",
                "screenshots/report/3_ground_truth/top_5_ground_truth_acc.npy"],
               ["Complex model", "Simple model", "Hedged image", "Ground Truth"], "accuracy", path)

    path = os.path.join(root, "1_mse_simple_vs_complex.png")
    comparison("single frame MSE U-Net MSE",
               ["screenshots/report/0_single_frame_MSE_u_net/complex/mse_loss.npy",
                "screenshots/report/0_single_frame_MSE_u_net/simple/mse_loss.npy"],
               ["Complex model", "Simple model"], "mse", path)

    # 2: uncertainty U-net vs MSE U-net
    path = os.path.join(root, "2_top_1_vgg_single_frame_Uncertainty.png")
    comparison("single frame Uncertainty U-Net top-1 VGG accuracy",
               ["screenshots/report/1_single_frame_uncertainty_u_net/top_1_model_acc.npy",
                "screenshots/report/0_single_frame_MSE_u_net/complex/top_1_model_acc.npy",
                "screenshots/report/3_ground_truth/top_1_masked_acc.npy",
                "screenshots/report/3_ground_truth/top_1_ground_truth_acc.npy"],
               ["Single Frame Uncertainty U-Net", "Single Frame MSE U-Net", "Hedged image", "Ground Truth"], "accuracy",
               path=path)

    path = os.path.join(root, "2_top_5_vgg_single_frame_Uncertainty.png")
    comparison("single frame Uncertainty U-Net top-5 VGG accuracy",
               ["screenshots/report/1_single_frame_uncertainty_u_net/top_5_model_acc.npy",
                "screenshots/report/0_single_frame_MSE_u_net/complex/top_5_model_acc.npy",
                "screenshots/report/3_ground_truth/top_5_masked_acc.npy",
                "screenshots/report/3_ground_truth/top_5_ground_truth_acc.npy"],
               ["Single Frame Uncertainty U-Net", "Single Frame MSE U-Net", "Hedged image", "Ground Truth"], "accuracy",
               path=path)

    path = os.path.join(root, "2_mse_single_frame_Uncertainty.png")
    comparison("single frame Uncertainty U-Net MSE",
               ["screenshots/report/1_single_frame_uncertainty_u_net/mse_loss.npy",
                "screenshots/report/0_single_frame_MSE_u_net/complex/mse_loss.npy"],
               ["Single Frame Uncertainty U-Net", "Single Frame MSE U-Net"], "mse", path)

    # 3: Recurrent Uncertainty U-net vs uncertainty U-net
    path = os.path.join(root, "3_top_1_vgg_recurrent_Uncertainty.png")
    comparison("Recurrent Uncertainty U-Net top-1 VGG accuracy",
               ["screenshots/report/5_recurrent_results/ep300/2_frame_report/top_1_model_acc.npy",
                "screenshots/report/1_single_frame_uncertainty_u_net/top_1_model_acc.npy",
                "screenshots/report/3_ground_truth/top_1_masked_acc.npy",
                "screenshots/report/3_ground_truth/top_1_ground_truth_acc.npy"],
               ["Recurrent Uncertainty U-Net (2-frames)", "Single Frame Uncertainty U-Net", "Hedged image",
                "Ground Truth"],
               "accuracy", path=path)

    path = os.path.join(root, "3_top_5_vgg_recurrent_Uncertainty.png")
    comparison("Recurrent Uncertainty U-Net top-5 VGG accuracy",
               ["screenshots/report/5_recurrent_results/ep300/2_frame_report/top_5_model_acc.npy",
                "screenshots/report/1_single_frame_uncertainty_u_net/top_5_model_acc.npy",
                "screenshots/report/3_ground_truth/top_5_masked_acc.npy",
                "screenshots/report/3_ground_truth/top_5_ground_truth_acc.npy"],
               ["Recurrent Uncertainty U-Net (2-frames)", "Single Frame Uncertainty U-Net", "Hedged image",
                "Ground Truth"],
               "accuracy", path=path)

    path = os.path.join(root, "3_mse_recurrent_Uncertainty.png")
    comparison("Recurrent Uncertainty U-Net MSE",
               ["screenshots/report/5_recurrent_results/ep300/2_frame_report/mse_loss.npy",
                "screenshots/report/1_single_frame_uncertainty_u_net/mse_loss.npy"],
               ["Recurrent Uncertainty U-Net (2-frames)", "Single Frame Uncertainty U-Net"], "mse", path)

    # 4: Recurrent Uncertainty U-net multi-frames comparisons
    path = os.path.join(root, "4_top_1_vgg_recurrent_Uncertainty_frame_no_comparison.png")
    comparison("Recurrent Uncertainty U-Net top-1 VGG accuracy",
               ["screenshots/report/5_recurrent_results/ep300/2_frame_report/top_1_model_acc.npy",
                "screenshots/report/5_recurrent_results/ep300/3_frame_report/top_1_model_acc.npy",
                "screenshots/report/5_recurrent_results/ep300/5_frame_report/top_1_model_acc.npy",
                "screenshots/report/3_ground_truth/top_1_masked_acc.npy",
                "screenshots/report/3_ground_truth/top_1_ground_truth_acc.npy"],
               ["Recurrent Uncertainty U-Net (2-frames)", "Recurrent Uncertainty U-Net (3-frames)",
                "Recurrent Uncertainty U-Net (5-frames)", "Hedged image", "Ground Truth"],
               "accuracy", path=path)

    path = os.path.join(root, "4_top_5_vgg_recurrent_Uncertainty_frame_no_comparison.png")
    comparison("Recurrent Uncertainty U-Net top-5 VGG accuracy",
               ["screenshots/report/5_recurrent_results/ep300/2_frame_report/top_5_model_acc.npy",
                "screenshots/report/5_recurrent_results/ep300/3_frame_report/top_5_model_acc.npy",
                "screenshots/report/5_recurrent_results/ep300/5_frame_report/top_5_model_acc.npy",
                "screenshots/report/3_ground_truth/top_5_masked_acc.npy",
                "screenshots/report/3_ground_truth/top_5_ground_truth_acc.npy"],
               ["Recurrent Uncertainty U-Net (2-frames)", "Recurrent Uncertainty U-Net (3-frames)",
                "Recurrent Uncertainty U-Net (5-frames)", "Hedged image", "Ground Truth"],
               "accuracy", path=path)

    path = os.path.join(root, "4_mse_recurrent_Uncertainty_frame_no_comparison.png")
    comparison("Recurrent Uncertainty U-Net MSE",
               ["screenshots/report/5_recurrent_results/ep300/2_frame_report/mse_loss.npy",
                "screenshots/report/5_recurrent_results/ep300/3_frame_report/mse_loss.npy",
                "screenshots/report/5_recurrent_results/ep300/5_frame_report/mse_loss.npy"],
               ["Recurrent Uncertainty U-Net (2-frames)", "Recurrent Uncertainty U-Net (3-frames)",
                "Recurrent Uncertainty U-Net (5-frames)"], "mse", path)

    # 5: All comparisons:
    path = os.path.join(root, "5_top_1_vgg.png")
    comparison("top-1 VGG accuracy",
               ["screenshots/report/0_single_frame_MSE_u_net/complex/top_1_model_acc.npy",
                "screenshots/report/1_single_frame_uncertainty_u_net/top_1_model_acc.npy",
                "screenshots/report/5_recurrent_results/ep300/5_frame_report/top_1_model_acc.npy",
                "screenshots/report/3_ground_truth/top_1_masked_acc.npy",
                "screenshots/report/3_ground_truth/top_1_ground_truth_acc.npy"],
               ["Single frame MSE U-Net", "Single frame Uncertainty U-Net", "Recurrent Uncertainty U-Net (5-frames)",
                "Hedged image",
                "Ground Truth"],
               "accuracy", path=path)

    path = os.path.join(root, "5_top_5_vgg.png")
    comparison("top-5 VGG accuracy",
               ["screenshots/report/0_single_frame_MSE_u_net/complex/top_5_model_acc.npy",
                "screenshots/report/1_single_frame_uncertainty_u_net/top_5_model_acc.npy",
                "screenshots/report/5_recurrent_results/ep300/5_frame_report/top_5_model_acc.npy",
                "screenshots/report/3_ground_truth/top_5_masked_acc.npy",
                "screenshots/report/3_ground_truth/top_5_ground_truth_acc.npy"],
               ["Single frame MSE U-Net", "Single frame Uncertainty U-Net", "Recurrent Uncertainty U-Net (5-frames)",
                "Hedged image",
                "Ground Truth"],
               "accuracy", path=path)

    path = os.path.join(root, "5_mse.png")
    comparison("MSE",
               ["screenshots/report/0_single_frame_MSE_u_net/complex/mse_loss.npy",
                "screenshots/report/1_single_frame_uncertainty_u_net/mse_loss.npy",
                "screenshots/report/5_recurrent_results/ep300/5_frame_report/mse_loss.npy"],
               ["Single frame MSE U-Net", "Single frame Uncertainty U-Net", "Recurrent Uncertainty U-Net (5-frames)"],
               "mse", path)

    # effective density plot
    generate_effective_density_plot()


def get_x_intersect(m, y, c):
    return (y - c) / m


def get_gradient(pt1, pt2):
    return (pt1[1] - pt2[1]) / (pt1[0] - pt2[0])


def get_constant(m, pt):
    return pt[1] - m * pt[0]


def find_segment(segments, target):
    for i in range(len(segments) - 1):
        if segments[i] > target > segments[i + 1]:
            pt1 = [0.1 * i, segments[i]]
            pt2 = [0.1 * (i + 1), segments[i + 1]]
            return pt1, pt2
    return -1, -1


def get_effective_density(hedged_path, model_path):
    hedged_ys = np.load(hedged_path)
    target_path_ys = np.load(model_path)
    effective_hedge_density = []
    for target_y in target_path_ys:
        pt1, pt2 = find_segment(hedged_ys, target_y)
        m = get_gradient(pt1, pt2)
        c = get_constant(m, pt1)
        effective_hedge_density.append(get_x_intersect(m, target_y, c))
    return effective_hedge_density


def generate_effective_density_plot():
    root = "screenshots/report/6_effective_density"
    densities = np.arange(0.0, 0.9, 0.1)
    x_data = [densities] * 3
    x_data[1] = np.arange(0.0, 0.8, 0.1)

    effective_density = get_effective_density("screenshots/report/3_ground_truth/top_1_masked_acc.npy",
                                                     "screenshots/report/5_recurrent_results/ep300/5_frame_report/top_1_model_acc.npy")

    alex_effective_density = np.array([0.025, 0.06, 0.09, 0.11, 0.17, 0.22, 0.31, 0.4])

    y_data = [effective_density, alex_effective_density, densities]


    legend = ["VGG top-1 result - Recurrent Uncertainty U-Net (5-frames)",
              "VGG top-1 result - W-Net (2-frames)",
              "Identity"]
    path = os.path.join(root, "Effective Hedge Density Comparison.png")
    plot_graph("Effective Hedge Density Comparison", "hedge density",
               "effective hedge density", x_data, y_data, legend, path, x_lim=[0, 0.85], y_lim=[0, 0.85])

def main():
    # generate_inference_data()
    # generate_report_images()
    # perform_comparisons_recurrent_models()
    generate_effective_density_plot()

    # path = os.path.join("screenshots", "real_data_result")
    # os.makedirs(path, exist_ok=True)
    # inference_real_data([
    #     "records/actual_results/2_recurrent_frame_uncertainty_u_net/new_model/checkpoint/ep300(lr=1e-4)/BEST_checkpoint.tar",
    #     "records/actual_results/1_Uncertainty_UNet/new_model/single_frame_clamp_uncertainty_network/checkpoints/ep100(lr=3e-3)/BEST_checkpoint.tar"],
    #     path)


if __name__ == '__main__':
    main()
