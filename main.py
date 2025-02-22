from copy import deepcopy
import random
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import os
import argparse
import xgboost as xgb
from torch import nn
from dataloader import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score, confusion_matrix, \
    classification_report, roc_curve
from tqdm import trange
from model import EmbeddingNet
import json

# set cuda environment variable
# note: Please comment out this line of code, if GPU is unavailable,
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# required functions
class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list, highlight_indices: np.ndarray = None):
        self.num_classes = num_classes
        self.labels = labels
        self.highlight_indices = highlight_indices
        self.matrix = np.zeros((num_classes, num_classes))
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def update(self, preds, real):
        for p, t in zip(preds, real):
            self.matrix[p, t] += 1

    def plot(self):
        self.confusion_matrix = deepcopy(self.matrix)
        for i in range(len(self.matrix[0])):
            total_num = np.sum(self.matrix[:, i])
            for j in range(len(self.matrix[0])):
                # self.confusion_matrix[j][i] = round(self.confusion_matrix[j][i] / total_num, 3)
                self.confusion_matrix[j][i] = round(float(self.confusion_matrix[j][i]) / total_num, 2) if total_num > 0 else 0.0

        matrix = np.array(self.confusion_matrix)
        if opt.dataset == 'cicids':
            plt.figure(figsize=(10, 8))
            fontsize = 8
        elif opt.dataset == 'botiot':
            plt.figure(figsize=(7, 6))
            fontsize = 7
        else:
            print("Please define the figure size and fontsize in advance.")
            return

        plt.imshow(matrix, cmap='Blues', aspect='auto')

        plt.xticks(range(self.num_classes), self.labels, rotation=30, fontsize=fontsize)
        plt.yticks(range(self.num_classes), self.labels, fontsize=fontsize)
        plt.colorbar()
        plt.xlabel('True Labels', fontsize=fontsize)
        plt.ylabel('Predicted Labels', fontsize=fontsize)

        # mark probability
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = matrix[y, x]
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         fontsize=fontsize,
                         color="white" if info > thresh else "black")

        # highlight specified indices
        if self.highlight_indices is not None:
            for index in self.highlight_indices:
                plt.gca().get_xticklabels()[index].set_color('red')
                plt.gca().get_yticklabels()[index].set_color('red')

        result_dir = f'./result/{opt.dataset}/hybrid/{opt.split}/'
        os.makedirs(result_dir, exist_ok=True)
        plt.savefig(result_dir + 'confusion_matrix.svg', bbox_inches='tight')
        plt.show()


def save_args(opt, path="./args/args.json"):
    # save argparse parameters to json
    with open(path, "w") as f:
        json.dump(vars(opt), f, indent=4)
    print(f"Arguments saved to {path}")


def load_args(path="./args/args.json"):
    # load argparse parameters from json
    with open(path, "r") as f:
        args_dict = json.load(f)
    return argparse.Namespace(**args_dict)


def map_label(label, classes):
    # transform label and make them continuous (like label 2, 5 ,7 transform to 2, 3, 4)
    mapped_label = torch.zeros_like(label, dtype=torch.long)
    for i, class_label in enumerate(classes):
        mapped_label[label == class_label] = i
    return mapped_label.to(device)


def inverse_map(label, classes):
    # inverse label transformation
    mapped_label = np.zeros_like(label)
    classes = classes.cpu().numpy()
    for i, class_label in enumerate(classes):
        mapped_label[label == i] = class_label
    return mapped_label


def histogram_plot(pred_score, stage):
    # 计算 x 轴范围
    x_min = pred_score.min()
    x_max = pred_score.max()

    # 计算自适应刻度间隔
    range_x = x_max - x_min
    x_interval = range_x / 5  # 让 x 轴有 5 个间隔

    # 柱状图的区间（15 份）
    bins_hist = np.linspace(x_min, x_max, 15)  # 15 份 bin

    # 修正柱状图偏移问题（align='edge' 让柱子从 bins_hist[i] 开始）
    bar_width = bins_hist[1] - bins_hist[0]

    # 创建图形
    fig, ax = plt.subplots(figsize=(6, 4))

    # 设置 x 轴刻度
    ax.set_xticks(np.arange(x_min, x_max + x_interval, x_interval))
    ax.set_xlim(x_min - 0.02, x_max + 0.02)

    # 统计每个区间中的样本比例
    y_seen_unseen = dataset.test_seen_unseen_label.cpu().numpy()
    if stage == 1:
        y_true = dataset.binary_test.cpu().numpy()
        benign_counts, _ = np.histogram(pred_score[y_true == 0], bins=bins_hist)
        malicious = pred_score[y_true == 1]
        seen_malicious_counts, _ = np.histogram(malicious[y_seen_unseen == 0], bins=bins_hist)
        unseen_malicious_counts, _ = np.histogram(malicious[y_seen_unseen == 1], bins=bins_hist)
        # 计算比例（归一化为总数的比例）
        benign_counts = benign_counts / len(pred_score[y_true == 0])
        seen_malicious_counts = seen_malicious_counts / len(malicious[y_seen_unseen == 0])
        unseen_malicious_counts = unseen_malicious_counts / len(malicious[y_seen_unseen == 1])
        # 绘制柱状图（Benign）
        ax.bar(bins_hist[:-1], benign_counts, width=bar_width, align='edge',
               color="#C7DDEC", edgecolor="#3E89BE", alpha=0.3, label="Benign")
        # 设置标签和图例
        ax.set_xlabel("Detector Score")
        ax.set_ylabel("Density")
        ax.set_title("Density Histogram of the Detector")
    elif stage == 2:
        seen_malicious_counts, _ = np.histogram(pred_score[y_seen_unseen == 0], bins=bins_hist)
        unseen_malicious_counts, _ = np.histogram(pred_score[y_seen_unseen == 1], bins=bins_hist)
        # 计算比例（归一化为总数的比例）
        seen_malicious_counts = seen_malicious_counts / len(pred_score[y_seen_unseen == 0])
        unseen_malicious_counts = unseen_malicious_counts / len(pred_score[y_seen_unseen == 1])
        ax.set_xlabel("Discriminator Score")
        ax.set_ylabel("Density")
        ax.set_title("Density Histogram of the Discriminator")
    else:
        print("wrong stage")
        return

    # 绘制柱状图（seen_Malicious）
    ax.bar(bins_hist[:-1], seen_malicious_counts, width=bar_width, align='edge',
           color="#C0DFC0", edgecolor="#198D19", alpha=0.3, label="Seen Malicious")

    # 绘制柱状图（unseen_Malicious）
    ax.bar(bins_hist[:-1], unseen_malicious_counts, width=bar_width, align='edge',
           color="#FFD966", edgecolor="#FFC000", alpha=0.3, label="Unseen Malicious")

    ax.legend(loc='upper right', bbox_to_anchor=(0.9, 1))  # 让图例稍微向左移动
    plt.show()

def indicator(k_matrix, sentry):
    # obtain adaptive K for each sample
    k_values = adaptive_K.view(-1).long()

    # representing the first k neighbors to be calculated for each sample
    indices = torch.arange(k_matrix.size(1)).expand(k_matrix.size(0), -1).to(device)

    # broadcast K for each sample
    mask = indices < k_values.unsqueeze(1)

    # calculate seen count for each sample
    seen_count = (k_matrix < sentry).float() * mask.float()

    seen_count = seen_count.sum(dim=1)

    # calculate ood score
    score = k_values.float() / (seen_count + 1)

    return score


def evaluation(y_true, score, option):
    # classification report
    score = np.array(score)
    if option == 3:
        report = classification_report(y_true, score,
                                       target_names=dataset.traffic_names[dataset.knownclasses.cpu().numpy()], digits=4)
        print(report)
        return

    # customized stage report
    if option == 4:
        report = classification_report(y_true, score,
                                       target_names=dataset.traffic_names[dataset.novelclasses.cpu().numpy()], digits=4)
        print(report)
        return

    # ultimate hybrid report
    if option == 5:
        report = classification_report(y_true, score, target_names=dataset.traffic_names, digits=4)
        print(report)
        return

    # calculate AUC-ROC
    auc_roc = roc_auc_score(y_true, score)
    print(f"AUC-ROC: {auc_roc}")

    # calculate Precision-Recall and AUC-PR
    precision, recall, _ = precision_recall_curve(y_true, score)
    auc_pr = auc(recall, precision)
    print(f"AUC-PR: {auc_pr}")

    # choose the threshold with optimal f1_score
    if opt.f1score:
        f1_scores = 2 * (precision * recall) / (precision + recall)
        best_f1_index = np.argmax(f1_scores)
        best_threshold = _[best_f1_index]
        print(best_threshold)
        y_pred = (score >= best_threshold).astype(int)

    # choose the threshold through TPR[threshold] (Model Performance when True Positive Rate achieves threshold like 1.95 or 1.99)
    else:
        _, tpr, thresholds = roc_curve(y_true, score)
        best_index = np.argwhere(tpr >= opt.threshold)[0]
        best_threshold = thresholds[best_index]
        print(best_threshold)
        y_pred = (score >= best_threshold).astype(int)

    # calculate F1-Score
    f1 = f1_score(y_true, y_pred)
    print(f"F1-Score: {f1}")

    # calculate Micro Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")

    # calculate Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{conf_matrix}")

    # visualization of Precision-Recall Curve
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    # calculate class-wise Recall
    recall_per_class = {}
    y_pred = torch.from_numpy(y_pred).to(device)
    ave_recall = 0

    if option == 1:  # detector
        for cls in dataset.allclasses:
            if cls.item() == 0:
                tp = torch.sum(torch.logical_and(y_pred.eq(0), dataset.test_label.eq(0))).item()
                fn = torch.sum(torch.logical_and(y_pred.eq(1), dataset.test_label.eq(0))).item()
            else:
                tp = torch.sum(torch.logical_and(y_pred.eq(1), dataset.test_label.eq(cls))).item()
                fn = torch.sum(torch.logical_and(y_pred.eq(0), dataset.test_label.eq(cls))).item()

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall_per_class[cls.item()] = recall
            ave_recall += recall

    elif option == 2:  # discrimination
        for cls in dataset.maliciousclasses:
            if cls in dataset.knownclasses:
                tp = torch.sum(
                    torch.logical_and(y_pred.eq(0), dataset.test_label[dataset.benign_size_test:].eq(cls))).item()
                fn = torch.sum(
                    torch.logical_and(y_pred.eq(1), dataset.test_label[dataset.benign_size_test:].eq(cls))).item()
            else:
                tp = torch.sum(
                    torch.logical_and(y_pred.eq(1), dataset.test_label[dataset.benign_size_test:].eq(cls))).item()
                fn = torch.sum(
                    torch.logical_and(y_pred.eq(0), dataset.test_label[dataset.benign_size_test:].eq(cls))).item()

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall_per_class[cls.item()] = recall
            ave_recall += recall

    ave_recall /= dataset.allclasses.shape[0]
    print("Recall per class:")
    for cls, recall in recall_per_class.items():
        print(f"Class {dataset.traffic_names[cls]}: Recall = {recall}")

    return y_pred, best_threshold


parser = argparse.ArgumentParser()

# set hyperparameters
# note: Please set the customized parameter to false, since Bot-Iot dataset has only 5 classes.
# note: For all CIC-IDS2017 dataset splits, use cicids_args.json in args folder to get the reported result.
# note: For all Bot-Iot dataset splits, use botiot_args.json in args folder to get the reported result.
parser.add_argument('--dataset', default='cicids', help='Dataset')
parser.add_argument('--split', default='4', help='Dataset split for training and evaluation')
parser.add_argument('--manualSeed', type=int, default=42, help='Random seed')
parser.add_argument('--resSize', type=int, default=70, help='Size of visual features')
parser.add_argument('--embedSize', type=int, default=128, help='Size of embedding h')
parser.add_argument('--outzSize', type=int, default=32, help='Size of non-liner projection z')
parser.add_argument('--kmin', type=int, default=3, help='Local neighbors kmin')
parser.add_argument('--kmax', type=int, default=20, help='Local neighbors kmax')
parser.add_argument('--khat', type=int, default=200, help='To calculate local density')
parser.add_argument('--customized', type=bool, default=False, help='Whether to use customized model')
parser.add_argument('--f1score', type=bool, default=False, help='Evaluate model performance with optimal F1-score')
parser.add_argument('--threshold', type=float, default=0.99, help='Evaluate model performance with TPR[threshold]')

opt = parser.parse_args()

# load pre-defined hyperparameters
# note: If you want to customize the hyperparameters, please comment out this line of code.
opt = load_args("args/cicids_args.json")

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set seed
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# loading data
dataset = DataLoader(opt)

# 1st step: Detect malicious traffic
from adbench.baseline.Supervised import supervised

model = supervised(seed=42, model_name='XGB')  # initialization

# training procedure of detector
print("start fitting detector")
start_time = time.time()
model.fit(dataset.train_feature.cpu().numpy(), dataset.binary_train.cpu().numpy())  # fit
end_time = time.time()
print("end fitting detector")
print("Training time of the detector：%.4f seconds" % (end_time - start_time))

# inference procedure of detector
print("start evaluating detector")
start_time = time.time()
detector_score = model.predict_score(dataset.test_feature.cpu().numpy())  # predict
end_time = time.time()
# evaluation of detector
y_true = dataset.binary_test.cpu().numpy()
detector_prediction, _ = evaluation(y_true, detector_score, 1)
histogram_plot(detector_score, 1)

print("end evaluating detector")
print("Inference time of the detector：%.4f seconds" % (end_time - start_time))

# 2nd step: Discriminate unknown categories traffic
test_feature = dataset.all_malicious_feature[dataset.train_seen_feature.shape[0]:]

# sentry denotes whether it belongs to the test set or training set
sentry = dataset.train_seen_feature.shape[0]

# initialize hyperparameters and matrices
pairwise = nn.PairwiseDistance(p=2)
khat = opt.khat

# training procedure of discriminator
print("start fitting and evaluating discriminator")
indice_matrix = torch.load(f"./matrix/{opt.dataset}/{opt.split}/indice_matrix.pt").to(device)
dis_matrix = torch.load(f"./matrix/{opt.dataset}/{opt.split}/dis_matrix.pt").to(device)

# calculate khat nearest neighborhood average distance
class_mean_distances = {}

for cls in dataset.maliciousclasses:
    class_distances = dis_matrix[dataset.test_label[dataset.benign_size_test:] == cls]
    class_mean = class_distances.mean().item()
    class_mean_distances[cls.item()] = class_mean

for cls, mean_distance in class_mean_distances.items():
    print(f"Class {dataset.traffic_names[cls]}: Mean nearest distance = {mean_distance}")

# inference procedure of discriminator
min_distance = torch.min(dis_matrix)
max_distance = torch.max(dis_matrix)
normalized_distances = (dis_matrix - min_distance) / (max_distance - min_distance)
inverted_distances = 1 - normalized_distances
K_min = opt.kmin
K_max = opt.kmax
adaptive_K = (K_min + inverted_distances * (K_max - K_min)).long()
discriminator_score = indicator(indice_matrix, sentry)

discriminator_prediction, threshold = evaluation(dataset.test_seen_unseen_label.cpu().numpy(),
                                                  discriminator_score.cpu().numpy(), 2)
histogram_plot(discriminator_score.cpu().numpy(), 2)
print("end fitting and evaluating discriminator")

# 3rd step: Classify known categories traffic
print("start fitting and evaluating classifier")
# training procedure of known_class_classifier
known_class_classifier = xgb.XGBClassifier()
known_class_classifier.fit(dataset.train_seen_feature.cpu().numpy(),
                           map_label(dataset.train_seen_label, dataset.knownclasses).cpu().numpy())
# inference procedure of known_class_classifier
known_preds = known_class_classifier.predict(dataset.test_seen_feature.cpu().numpy())
# evaluation of known_class_classifier
evaluation(map_label(dataset.test_seen_label, dataset.knownclasses).cpu().numpy(), known_preds, 3)

print("end fitting and evaluating classifier")

# 4th step: Classify unknown categories traffic (Optional)
if opt.customized:
    print("start evaluating customized model")
    # load unknown_class_classifier
    unknown_class_classifier = xgb.XGBClassifier()
    unknown_class_classifier.load_model('./models/' + opt.dataset + '/' + opt.split + '/cls.model')
    mapper = Embedding_Net(opt).to(device)
    mapper.load_state_dict(torch.load('./models/' + opt.dataset + '/' + opt.split + '/map.pt'))
    mapper.eval()
    # inference procedure of unknown_class_classifier
    with torch.no_grad():
        embed, _ = mapper(dataset.test_unseen_feature)

    unknown_preds = unknown_class_classifier.predict(embed.cpu().numpy())
    # evaluation of unknown_class_classifier
    evaluation(map_label(dataset.test_unseen_label, dataset.novelclasses).cpu().numpy(), unknown_preds, 4)

    print("end evaluating customized model")
else:
    # w/o test unknown classifier
    unknown_preds = map_label(dataset.test_unseen_label, dataset.novelclasses).cpu().numpy()

# 5th step: hybrid ultimate output (Unknown predictions integration is optional)
print("start calculating hybrid performance")
# inverse predictions for malicious traffic
known_preds_inverse = inverse_map(known_preds, dataset.knownclasses)
unknown_preds_inverse = inverse_map(unknown_preds, dataset.novelclasses)

# collect all predictions (benign and malicious)
preds_all = np.concatenate(
    (detector_prediction[:dataset.benign_size_test].cpu().numpy(), known_preds_inverse, unknown_preds_inverse), axis=0)

# get score for benign and malicious traffic
score_for_benign = detector_prediction[:dataset.benign_size_test]
score_for_malicious = detector_prediction[dataset.benign_size_test:]

# get score for known and unknown malicious traffic
score_for_known = discriminator_prediction[:dataset.test_seen_feature.shape[0]]
score_for_unknown = discriminator_prediction[dataset.test_seen_feature.shape[0]:]

# get index for misdetected benign traffic
det_wrong_benign = torch.where(score_for_benign.eq(1))[0].to(device)
# get index for undetected malicious traffic
det_wrong_malicious = torch.where(score_for_malicious.eq(0))[0].to(device)

# get index for misdiscriminated known malicious traffic
dis_wrong_known = torch.where(score_for_known.eq(1))[0].to(device)
# get index for misdiscriminated unknown malicious traffic
dis_wrong_unknown = torch.where(score_for_unknown.eq(0))[0].to(device)

with torch.no_grad():
    # 1.give misdiscriminated known malicious traffic new labels from unknown classes
    if len(dis_wrong_known) > 0:
        if opt.customized:
            known_features = dataset.test_seen_feature[dis_wrong_known]
            corrected_known_preds = unknown_class_classifier.predict(mapper(known_features)[0].cpu().numpy())
            corrected_known_preds_inverse = inverse_map(corrected_known_preds, dataset.novelclasses)
            preds_all[dataset.benign_size_test + dis_wrong_known.cpu().numpy()] = corrected_known_preds_inverse
        else:
            # random assignment for unseen classes
            corrected_known_preds = torch.randint(low=0, high=dataset.novelclasses.shape[0],
                                                  size=(dis_wrong_known.shape[0],))
            corrected_known_preds_inverse = inverse_map(corrected_known_preds, dataset.novelclasses)
            preds_all[dataset.benign_size_test + dis_wrong_known.cpu().numpy()] = corrected_known_preds_inverse

    # 2. give misdiscriminated unknown malicious traffic new labels from known classes
    if len(dis_wrong_unknown) > 0:
        unknown_features = dataset.test_unseen_feature[dis_wrong_unknown]
        corrected_unknown_preds = known_class_classifier.predict(unknown_features.cpu().numpy())
        corrected_unknown_preds_inverse = inverse_map(corrected_unknown_preds, dataset.knownclasses)
        preds_all[dataset.benign_size_test + dataset.test_seen_feature.shape[
            0] + dis_wrong_unknown.cpu().numpy()] = corrected_unknown_preds_inverse

    # 3. give misdetected benign traffic new labels from malicious classes
    if len(det_wrong_benign) > 0:
        test_benign_feature = dataset.test_feature[:dataset.benign_size_test]
        benign_features = test_benign_feature[det_wrong_benign]

        indice_matrix = torch.IntTensor(size=(benign_features.shape[0], opt.kmax)).to(device)
        dis_matrix = torch.FloatTensor(size=(benign_features.shape[0], 1)).to(device)

        for i in trange(benign_features.shape[0]):
            expand_feature = benign_features[i].unsqueeze(0).expand_as(dataset.all_malicious_feature)
            dis = pairwise(expand_feature, dataset.all_malicious_feature)
            # sort and selection
            distances, indices = torch.topk(dis, k=khat, largest=False)
            indice_matrix[i] = indices[:opt.kmax]
            dis_matrix[i] = distances.mean()

        min_distance = torch.min(dis_matrix)
        max_distance = torch.max(dis_matrix)
        normalized_distances = (dis_matrix - min_distance) / (max_distance - min_distance)
        inverted_distances = 1 - normalized_distances
        K_min = opt.kmin
        K_max = opt.kmax
        adaptive_K = (K_min + inverted_distances * (K_max - K_min)).long()

        benign_discriminator_scores = indicator(indice_matrix, sentry).cpu().numpy()
        if opt.customized:
            known_unknown_preds = np.where(benign_discriminator_scores < threshold,
                                           inverse_map(known_class_classifier.predict(benign_features.cpu().numpy()),
                                                       dataset.knownclasses),
                                           inverse_map(unknown_class_classifier.predict(
                                               mapper(benign_features)[0].cpu().numpy()), dataset.novelclasses))
        else:
            # random assignment for unseen classes
            corrected_benign_unknown_preds = torch.randint(low=0, high=dataset.novelclasses.shape[0],
                                                           size=(benign_features.shape[0],))
            known_unknown_preds = np.where(benign_discriminator_scores < threshold,
                                           inverse_map(known_class_classifier.predict(benign_features.cpu().numpy()),
                                                       dataset.knownclasses),
                                           inverse_map(corrected_benign_unknown_preds, dataset.novelclasses))

        preds_all[det_wrong_benign.cpu().numpy()] = known_unknown_preds

    # 4. give undetected malicious traffic benign labels
    if len(det_wrong_malicious) > 0:
        preds_all[dataset.benign_size_test + det_wrong_malicious.cpu().numpy()] = 0

# 5. final evaluation
evaluation(dataset.test_label.cpu().numpy(), preds_all, 5)
traffic_names = dataset.traffic_names
if opt.dataset == 'cicids':
    traffic_names[3] = "GoldenEye"
    traffic_names[4] = "Hulk"
    traffic_names[5] = "Slowhttptest"
    traffic_names[6] = "Slowloris"
    traffic_names[-1] = "XSS"
    traffic_names[-2] = "Sql Injection"
    traffic_names[-3] = "Brute Force"
confusion = ConfusionMatrix(num_classes=len(dataset.allclasses), labels=traffic_names,
                            highlight_indices=dataset.novelclasses.cpu().numpy())
confusion.update(preds_all, dataset.test_label.cpu().numpy())
confusion.plot()
