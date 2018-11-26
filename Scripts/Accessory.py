import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
import pandas as pd
import cv2


def realout(pdx, path, name):
    pdx = np.asmatrix(pdx)
    prl = (pdx[:, 1] > 0.5).astype('uint8')
    prl = pd.DataFrame(prl, columns=['Prediction'])
    out = pd.DataFrame(pdx, columns=['neg_score', 'pos_score'])
    out = pd.concat([out, prl], axis=1)
    out.insert(loc=0, column='Num', value=out.index)
    out.to_csv("../Neutrophil/{}/out/{}.csv".format(path, name), index=False)


def metrics(pdx, tl, path, name):
    pdx = np.asmatrix(pdx)
    prl = (pdx[:,1] > 0.5).astype('uint8')
    prl = pd.DataFrame(prl, columns = ['Prediction'])
    out = pd.DataFrame(pdx, columns = ['neg_score', 'pos_score'])
    outtl = pd.DataFrame(tl, columns = ['True_label'])
    out = pd.concat([out,prl,outtl], axis=1)
    out.to_csv("../Neutrophil/{}/out/{}.csv".format(path, name), index=False)
    accu = 0
    tott = out.shape[0]
    for idx, row in out.iterrows():
        if row['Prediction'] == row['True_label']:
            accu += 1
    accur = accu/tott
    accur = round(accur,2)
    print('Accuracy:')
    print(accur)
    y_score = pdx[:,1]
    try:
        auc = skl.metrics.roc_auc_score(tl, y_score)
        auc = round(auc,2)
        print('ROC-AUC:')
        print(auc)
        fpr, tpr, _ = skl.metrics.roc_curve(tl, y_score)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of {}'.format(name))
        plt.legend(loc="lower right")
        plt.savefig("../Neutrophil/{}/out/{}_ROC.png".format(path, name))

        average_precision = skl.metrics.average_precision_score(tl, y_score)
        print('Average precision-recall score: {0:0.2f}'.format(average_precision))
        plt.figure()
        precision, recall, _ = skl.metrics.precision_recall_curve(tl, y_score)
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('{} Precision-Recall curve: AP={:0.2f}; Accu={}'.format(name, average_precision, accur))
        plt.savefig("../Neutrophil/{}/out/{}_PRC.png".format(path, name))
    except(ValueError):
        print('Not able to generate plots based on this test set!')


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
    DIR = "../Neutrophil/{}/out/{}_posimg".format(path, name)
    DIRR = "../Neutrophil/{}/out/{}_negimg".format(path, name)
    rd = rd*1000

    try:
        os.mkdir(DIR)
    except(FileExistsError):
        pass

    try:
        os.mkdir(DIRR)
    except(FileExistsError):
        pass

    pdx = np.asmatrix(pred)

    prl = (pdx[:,1] > 0.5).astype('uint8')

    for ij in range(len(y)):
        id = str(ij + rd)
        if prl[ij] == 0:
            if y[ij] == 0:
                ddt = 'Correct'
            else:
                ddt = 'Wrong'

            weights_LR = w
            activation_lastconv = np.array([net[ij]])
            weights_LR = weights_LR.T
            activation_lastconv = activation_lastconv.T

            topNum = 1  # generate heatmap for top X prediction results
            scores = pred[ij]
            scoresMean = np.mean(scores, axis=0)
            ascending_order = np.argsort(scoresMean)
            IDX_category = ascending_order[::-1]  # [::-1] to sort in descending order
            curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[[0], :])
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
                # imname = DIR + '/' + id + ddt + '.png'
                # imname1 = DIR + '/' + id + ddt + '_img.png'
                # imname2 = DIR + '/' + id + ddt + '_hm.png'
                imname3 = DIRR + '/' + id + ddt + '_full.png'
                # cv2.imwrite(imname, curHeatMap)
                # cv2.imwrite(imname1, a)
                # cv2.imwrite(imname2, b)
                cv2.imwrite(imname3, full)


        else:
            if y[ij] == 1:
                ddt = 'Correct'
            else:
                ddt = 'Wrong'

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
                # imname = DIR + '/' + id + ddt + '.png'
                # imname1 = DIR + '/' + id + ddt +'_img.png'
                # imname2 = DIR + '/' + id + ddt + '_hm.png'
                imname3 = DIR + '/' + id + ddt + '_full.png'
                # cv2.imwrite(imname, curHeatMap)
                # cv2.imwrite(imname1, a)
                # cv2.imwrite(imname2, b)
                cv2.imwrite(imname3, full)


def CAM_R(net, w, pred, x, path, name, rd=0):
    DIRR = "../Neutrophil/{}/out/{}_img".format(path, name)
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
