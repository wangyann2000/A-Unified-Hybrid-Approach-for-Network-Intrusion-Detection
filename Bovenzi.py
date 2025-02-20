from copy import deepcopy
import random
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import os
import argparse
import xgboost as xgb
from dataloader import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score, confusion_matrix, \
    classification_report, roc_curve
from model import Embedding_Net
import json

# set cuda environment variable
# note: If GPU is unavailable, please comment out this line of code.
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

        result_dir = f'./result/{opt.dataset}/hybrid//Bovenzi/{opt.split}/'
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


def evaluation(y_true, score, option):
    # classification report
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
# note: Since Bot-Iot dataset has only 5 classes, no customized model is trained, please set the customized parameter to false.
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
# note: Please comment out this line of code, if you want to customize the hyperparameters,
opt = load_args("args/botiot_args.json")

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
print("end evaluating detector")
print("Inference time of the detector：%.4f seconds" % (end_time - start_time))

# 2nd step: Discriminate unknown categories traffic
test_feature = dataset.all_malicious_feature[dataset.train_seen_feature.shape[0]:]
# training procedure of discriminator
print("start fitting and evaluating discriminator")
known_class_classifier = xgb.XGBClassifier()
known_class_classifier.fit(dataset.train_seen_feature.cpu().numpy(), map_label(dataset.train_seen_label, dataset.knownclasses).cpu().numpy())
proba = known_class_classifier.predict_proba(test_feature.cpu().numpy())
discriminator_score = np.min(proba, axis=1)

discriminator_prediction, threshold = evaluation(dataset.test_seen_unseen_label.cpu().numpy(), discriminator_score, 2)

print("end fitting and evaluating discriminator")

# 3rd step: Classify known categories traffic
print("start fitting and evaluating classifier")
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

        proba = known_class_classifier.predict_proba(benign_features.cpu().numpy())
        benign_discriminator_scores = np.min(proba, axis=1)

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
