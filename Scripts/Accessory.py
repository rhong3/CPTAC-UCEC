import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import sklearn as skl
from scipy import interp
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from itertools import cycle


def realout(pdx, path, name):
    lbdict = {0: 'MSI', 1: 'Endometroid', 2: 'Serous-like', 3: 'POLE'}
    pdx = np.asmatrix(pdx)
    prl = pdx.argmax(axis=1).astype('uint8')
    prl = pd.DataFrame(prl, columns = ['Prediction'])
    prl = prl.replace(lbdict)
    out = pd.DataFrame(pdx, columns = ['MSI_score', 'Endometroid_score', 'Serious-like_score', 'POLE_score'])
    out = pd.concat([out, prl], axis=1)
    out.insert(loc=0, column='Num', value=out.index)
    out.to_csv("../Results/{}/out/{}.csv".format(path, name), index=False)


def metrics(pdx, tl, path, name):
    lbdict = {0: 'MSI', 1: 'Endometroid', 2: 'Serous-like', 3: 'POLE'}
    pdx = np.asmatrix(pdx)
    prl = pdx.argmax(axis=1).astype('uint8')
    prl = pd.DataFrame(prl, columns = ['Prediction'])
    outt = pd.DataFrame(pdx, columns = ['MSI_score', 'Endometroid_score', 'Serious-like_score', 'POLE_score'])
    outtl = pd.DataFrame(tl, columns = ['True_label'])
    out = pd.concat([outt,prl,outtl], axis=1)

    stprl = prl.replace(lbdict)
    stouttl = outtl.replace(lbdict)
    stout = pd.concat([outt, stprl, stouttl], axis=1)
    stout.to_csv("../Results/{}/out/{}.csv".format(path, name), index=False)

    accu = 0
    tott = out.shape[0]
    accua = 0
    accub = 0
    accuc = 0
    accud = 0
    for idx, row in out.iterrows():
        if row['Prediction'] == row['True_label']:
            accu += 1
            if row['True_label'] == 0:
                accua += 1
            elif row['True_label'] == 1:
                accub += 1
            elif row['True_label'] == 2:
                accuc += 1
            elif row['True_label'] == 3:
                accud += 1

    accur = round(accu/tott,2)
    print('Total Accuracy:')
    print(accur)

    tota = out[out.True_label == 0].shape[0]
    totb = out[out.True_label == 1].shape[0]
    totc = out[out.True_label == 2].shape[0]
    totd = out[out.True_label == 3].shape[0]
    try:
        accuar = round(accua/tota,2)
    except ZeroDivisionError:
        accuar = "No data for MSI."
    print('MSI Accuracy:')
    print(accuar)
    try:
        accubr = round(accub/totb,2)
    except ZeroDivisionError:
        accubr = "No data for Endometroid."
    print('Endometroid Accuracy:')
    print(accubr)
    try:
        accucr = round(accuc/totc,2)
    except ZeroDivisionError:
        accucr = "No data for Serious-like."
    print('Serious-like Accuracy:')
    print(accucr)
    try:
        accudr = round(accud/totd,2)
    except ZeroDivisionError:
        accudr = "No data for POLE."
    print('POLE Accuracy:')
    print(accudr)
    try:
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # PRC
        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        microy = []
        microscore = []
        for i in range(4):
            fpr[i], tpr[i], _ = skl.metrics.roc_curve(np.asarray((outtl.iloc[:, 0].values == int(i)).astype('uint8')),
                                                      np.asarray(pdx[:, i]).ravel())
            try:
                roc_auc[i] = skl.metrics.roc_auc_score(np.asarray((outtl.iloc[:, 0].values == int(i)).astype('uint8')),
                                                       np.asarray(pdx[:, i]).ravel())
            except ValueError:
                roc_auc[i] = np.nan

            microy.extend(np.asarray((outtl.iloc[:, 0].values == int(i)).astype('uint8')))
            microscore.extend(np.asarray(pdx[:, i]).ravel())

            precision[i], recall[i], _ = \
                skl.metrics.precision_recall_curve(np.asarray((outtl.iloc[:, 0].values == int(i)).astype('uint8')),
                                                   np.asarray(pdx[:, i]).ravel())
            try:
                average_precision[i] = \
                    skl.metrics.average_precision_score(np.asarray((outtl.iloc[:, 0].values == int(i)).astype('uint8')),
                                                        np.asarray(pdx[:, i]).ravel())
            except ValueError:
                average_precision[i] = np.nan

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = skl.metrics.roc_curve(np.asarray(microy).ravel(), np.asarray(microscore).ravel())
        roc_auc["micro"] = skl.metrics.auc(fpr["micro"], tpr["micro"])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = skl.metrics.precision_recall_curve(np.asarray(microy).ravel(),
                                                                                    np.asarray(microscore).ravel())
        average_precision["micro"] = skl.metrics.average_precision_score(np.asarray(microy).ravel(),
                                                                         np.asarray(microscore).ravel(), average="micro")

        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(4):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= 4

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = skl.metrics.auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
        for i, color in zip(range(4), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of {0} (area = {1:0.2f})'.format(lbdict[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of {}'.format(name))
        plt.legend(loc="lower right")
        plt.savefig("../Results/{}/out/{}_ROC.png".format(path, name))

        print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
        # setup plot details
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red'])
        plt.figure(figsize=(7, 9))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        lines.append(l)
        labels.append('iso-f1 curves')

        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('micro-average Precision-recall (area = {0:0.2f})'
                      ''.format(average_precision["micro"]))

        for i, color in zip(range(4), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append('Precision-recall for {0} (area = {1:0.2f})'.format(lbdict[i], average_precision[i]))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('{} Precision-Recall curve: Average Accu={}'.format(name, accur))
        plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=12))
        plt.savefig("../Results/{}/out/{}_PRC.png".format(path, name))
    except ValueError:
        print('Not able to generate plots based on this set!')


def py_returnCAMmap(activation, weights_LR):
    n_feat, w, h, n = activation.shape
    act_vec = np.reshape(activation, [n_feat, w*h])
    n_top = weights_LR.shape[0]
    out = np.zeros([w, h, n_top])

    for t in range(n_top):
        weights_vec = np.reshape(weights_LR[t], [1, weights_LR[t].shape[0]])
        heatmap_vec = np.dot(weights_vec,act_vec)
        heatmap = np.reshape(np.squeeze(heatmap_vec), [w, h])
        out[:,:,t] = heatmap

    return out


def im2double(im):
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


def py_map2jpg(imgmap, rang, colorMap):
    if rang is None:
        rang = [np.min(imgmap), np.max(imgmap)]

    heatmap_x = np.round(imgmap*255).astype(np.uint8)

    return cv2.applyColorMap(heatmap_x, cv2.COLORMAP_JET)


def CAM(net, w, pred, x, y, path, name, rd=0):
    lbdict = {0: 'MSI', 1: 'Endometroid', 2: 'Serous-like', 3: 'POLE'}
    DIRA = "../Results/{}/out/{}_MSI_img".format(path, name)
    DIRB = "../Results/{}/out/{}_Endometroid_img".format(path, name)
    DIRC = "../Results/{}/out/{}_Serious-like_img".format(path, name)
    DIRD = "../Results/{}/out/{}_POLE_img".format(path, name)
    rd = rd*1000

    try:
        os.mkdir(DIRA)
    except(FileExistsError):
        pass

    try:
        os.mkdir(DIRB)
    except(FileExistsError):
        pass

    try:
        os.mkdir(DIRC)
    except(FileExistsError):
        pass

    try:
        os.mkdir(DIRD)
    except(FileExistsError):
        pass

    pdx = np.asmatrix(pred)

    prl = pdx.argmax(axis=1).astype('uint8')

    for ij in range(len(y)):
        id = str(ij + rd)
        if prl[ij, 0] == 0:
            if y[ij] == 0:
                ddt = 'Correct'
            else:
                ddt = 'Wrong'

        elif prl[ij, 0] == 1:
            if y[ij] == 1:
                ddt = 'Correct'
            else:
                ddt = 'Wrong'

        elif prl[ij, 0] == 2:
            if y[ij] == 2:
                ddt = 'Correct'
            else:
                ddt = 'Wrong'

        elif prl[ij, 0] == 3:
            if y[ij] == 3:
                ddt = 'Correct'
            else:
                ddt = 'Wrong'
        else:
            ddt = 'Error'
            print("Prediction value error!")

        weights_LR = w
        activation_lastconv = np.array([net[ij]])
        weights_LR = weights_LR.T
        activation_lastconv = activation_lastconv.T

        topNum = 1  # generate heatmap for top X prediction results
        scores = pred[ij]
        scoresMean = np.mean(scores, axis=0)
        ascending_order = np.argsort(scoresMean)
        IDX_category = ascending_order[::-1]  # [::-1] to sort in descending order
        if prl[ij, 0] == 0:
            curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[[0], :])
            DIRR = DIRA
            catt = 'MSI'
        elif prl[ij, 0] == 1:
            curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[[1], :])
            DIRR = DIRB
            catt = 'Endometroid'
        elif prl[ij, 0] == 2:
            curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[[2], :])
            DIRR = DIRC
            catt = 'Serous-like'
        elif prl[ij, 0] == 3:
            curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[[3], :])
            DIRR = DIRD
            catt = 'POLE'
        else:
            curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[[0], :])
            DIRR = DIRA
            catt = 'MSI'
        for kk in range(topNum):
            curCAMmap_crops = curCAMmapAll[:, :, kk]
            curCAMmapLarge_crops = cv2.resize(curCAMmap_crops, (299, 299))
            curHeatMap = cv2.resize(im2double(curCAMmapLarge_crops), (299, 299))  # this line is not doing much
            curHeatMap = im2double(curHeatMap)
            curHeatMap = py_map2jpg(curHeatMap, None, 'jet')
            xim = x[ij].reshape(-1, 3)
            xim1 = xim[:, 0].reshape(-1, 299)
            xim2 = xim[:, 1].reshape(-1, 299)
            xim3 = xim[:, 2].reshape(-1, 299)
            image = np.empty([299,299,3])
            image[:, :, 0] = xim1
            image[:, :, 1] = xim2
            image[:, :, 2] = xim3
            a = im2double(image) * 255
            b = im2double(curHeatMap) * 255
            curHeatMap = a * 0.6 + b * 0.4
            ab = np.hstack((a,b))
            full = np.hstack((curHeatMap, ab))
            # imname = DIRR + '/' + id + ddt + '_' + catt + '.png'
            # imname1 = DIRR + '/' + id + ddt + '_' + catt + '_img.png'
            # imname2 = DIRR + '/' + id + ddt + '_' + catt + '_hm.png'
            imname3 = DIRR + '/' + id + ddt + '_' + catt + '_full.png'
            # cv2.imwrite(imname, curHeatMap)
            # cv2.imwrite(imname1, a)
            # cv2.imwrite(imname2, b)
            cv2.imwrite(imname3, full)


def CAM_R(net, w, pred, x, path, name, rd=0):
    DIRR = "../Results/{}/out/{}_img".format(path, name)
    rd = rd * 1000

    try:
        os.mkdir(DIRR)
    except(FileExistsError):
        pass

    pdx = np.asmatrix(pred)

    prl = (pdx[:,1] > 0.5).astype('uint8')

    for ij in range(len(prl)):
        id = str(ij + rd)
        weights_LR = w
        activation_lastconv = np.array([net[ij]])
        weights_LR = weights_LR.T
        activation_lastconv = activation_lastconv.T

        topNum = 1  # generate heatmap for top X prediction results
        scores = pred[ij]
        scoresMean = np.mean(scores, axis=0)
        ascending_order = np.argsort(scoresMean)
        IDX_category = ascending_order[::-1]  # [::-1] to sort in descending order
        curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[[1], :])
        for kk in range(topNum):
            curCAMmap_crops = curCAMmapAll[:, :, kk]
            curCAMmapLarge_crops = cv2.resize(curCAMmap_crops, (299, 299))
            curHeatMap = cv2.resize(im2double(curCAMmapLarge_crops), (299, 299))  # this line is not doing much
            curHeatMap = im2double(curHeatMap)
            curHeatMap = py_map2jpg(curHeatMap, None, 'jet')
            xim = x[ij].reshape(-1, 3)
            xim1 = xim[:, 0].reshape(-1, 299)
            xim2 = xim[:, 1].reshape(-1, 299)
            xim3 = xim[:, 2].reshape(-1, 299)
            image = np.empty([299,299,3])
            image[:, :, 0] = xim1
            image[:, :, 1] = xim2
            image[:, :, 2] = xim3
            a = im2double(image) * 255
            b = im2double(curHeatMap) * 255
            curHeatMap = a * 0.6 + b * 0.4
            ab = np.hstack((a,b))
            full = np.hstack((curHeatMap, ab))
            # imname = DIRR + '/' + id + '.png'
            # imname1 = DIRR + '/' + id + '_img.png'
            # imname2 = DIRR + '/' + id +'_hm.png'
            imname3 = DIRR + '/' + id + '_full.png'
            # cv2.imwrite(imname, curHeatMap)
            # cv2.imwrite(imname1, a)
            # cv2.imwrite(imname2, b)
            cv2.imwrite(imname3, full)
