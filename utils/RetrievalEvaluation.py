#coding=utf-8
import numpy as np
np.set_printoptions(threshold=np.inf)

def RetrievalEvaluation(C_depth, distM, model_label, depth_label, testMode=1):
    '''
    C_depth: retrieval number for the testing example, Nx1
    distM: distance matrix, row for testing example, column for training example
    model_label: model_label for training example
    depth_label: label for testing example

    testMode:
        1) use test  as query, find relevant examples in training data
        2) use test as query, find relevant examples in the testing data
    '''
    if testMode == 1:
        C = C_depth
        recall = np.zeros((distM.shape[0], distM.shape[1]))
        precision = np.zeros((distM.shape[0], distM.shape[1]))

        rankArray = np.zeros((distM.shape[0], distM.shape[1]))

    elif testMode == 2:
        C = C_depth - 1
        recall = np.zeros((distM.shape[0], distM.shape[1] - 1))
        precision = np.zeros((distM.shape[0], distM.shape[1] - 1))

        rankArray = np.zeros((distM.shape[0], distM.shape[1] - 1))

    nb_of_query = C.shape[0]
    p_points = np.zeros((nb_of_query, np.amax(C)))
    ap = np.zeros(nb_of_query)
    nn = np.zeros(nb_of_query)
    ft = np.zeros(nb_of_query)
    st = np.zeros(nb_of_query)
    dcg = np.zeros(nb_of_query)
    e_measure = np.zeros(nb_of_query)

    for qqq in range(nb_of_query):
        temp_dist = distM[qqq]
        s = list(temp_dist)
        R = sorted(range(len(s)), key=lambda k: s[k])
        if testMode == 1:
            model_label_l = model_label[R]
            numRetrieval = distM.shape[1]
            G = np.zeros(numRetrieval)
            rankArray[qqq] = R
        elif testMode == 2:
            model_label_l = model_label[R[1:]]
            numRetrieval = distM.shape[1] - 1
            G = np.zeros(numRetrieval)
            rankArray[qqq] = R[1:]

        for i in range(numRetrieval):
            if model_label_l[i] == depth_label[qqq]:
                G[i] = 1
        G_sum = np.cumsum(G)
        r1 = G_sum / float(C[qqq])
        p1 = G_sum / np.arange(1, numRetrieval + 1)
        r_points = np.zeros(C[qqq])
        for i in range(C[qqq]):
            temp = np.where(G_sum == i + 1)
            r_points[i] = np.where(G_sum == (i + 1))[0][0] + 1
        r_points_int = np.array(r_points, dtype=int)

        p_points[qqq][:int(C[qqq])] = G_sum[r_points_int - 1] / r_points
        ap[qqq] = np.mean(p_points[qqq][:int(C[qqq])])
        nn[qqq] = G[0]
        ft[qqq] = G_sum[C[qqq] - 1] / C[qqq]
        

        st[qqq] = G_sum[min(2 * C[qqq] - 1, G_sum.size - 1)] / C[qqq]
        p_32 = G_sum[min(31, G_sum.size - 1)] / min(32, G_sum.size)
        r_32 = G_sum[min(31, G_sum.size - 1)] / C[qqq]
        if p_32 == 0 and r_32 == 0:
            e_measure[qqq] = 0
        else:
            e_measure[qqq] = 2 * p_32 * r_32 / (p_32 + r_32)

        if testMode == 1:
            NORM_VALUE = 1 + np.sum(1 / np.log2(np.arange(2, C[qqq] + 1)))
            dcg_i = 1 / np.log2(np.arange(2, len(R) + 1)) * G[1:]
            dcg_i = np.insert(dcg_i, 0, G[0])
            dcg[qqq] = np.sum(dcg_i, axis=0) / NORM_VALUE
            recall[qqq] = r1
            precision[qqq] = p1
        elif testMode == 2:
            NORM_VALUE = 1 + np.sum(1 / np.log2(np.arange(2, C[qqq] + 1)))
            dcg_i = 1 / np.log2(np.arange(2, len(R[1:]) + 1)) * G[1:]
            dcg_i = np.insert(dcg_i, 0, G[0])
            dcg[qqq] = np.sum(dcg_i, axis=0) / NORM_VALUE
            recall[qqq] = r1
            precision[qqq] = p1
    print(np.shape(distM))

    nn_av = np.mean(nn)
    ft_av = np.mean(ft)
    st_av = np.mean(st)
    dcg_av = np.mean(dcg)
    e_av = np.mean(e_measure)
    map_ = np.mean(ap)

    pre = np.mean(precision, axis=0)
    rec = np.mean(recall, axis=0)

    return nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec, rankArray
