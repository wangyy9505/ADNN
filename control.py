
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import scale


def main():
    with open('pre_data.pkl', 'rb') as f:
        data = pickle.load(f)

    data1, y1 = data['data2'], data['y2']
    data2, y2 = data['data1'], data['y1']
    data3, y3 = data['data3'], data['y3']

    data3 = scale(data3, axis=0)

    # 数据分割, train, valid和test
    trainx, testx, trainy, testy = train_test_split(
        data1, y1, test_size=0.2, shuffle=True, stratify=y1, random_state=1234
    )

    trainx, validx, trainy, validy = train_test_split(
        trainx, trainy, test_size=len(testy), shuffle=True, stratify=trainy,
        random_state=1234
    )

    # RF
    valid_auc, test_auc, external_auc = [], [], []
    for n in [100, 300, 500, 700, 1000]:
        rf = RandomForestClassifier(
            n_estimators=n, n_jobs=50, random_state=1234
        )
        rf.fit(trainx, trainy)
        valid_pred = rf.predict_proba(validx)
        valid_auc.append(roc_auc_score(validy, valid_pred[:, 1]))

        test_pred = rf.predict_proba(testx)
        test_auc.append(roc_auc_score(testy, test_pred[:, 1]))

        exter_pred = rf.predict_proba(data3)
        external_auc.append(roc_auc_score(y3, exter_pred[:, 1]))

        if n == 100:
            rf_test = pd.DataFrame({'ytrue': testy, 'ypred': test_pred[:, 1]})
            rf_exter = pd.DataFrame({'ytrue': y3, 'ypred': exter_pred[:, 1]})
        else:
            if np.argmax(valid_auc) == len(valid_auc) - 1:
                rf_test = pd.DataFrame(
                    {'ytrue': testy, 'ypred': test_pred[:, 1]}
                )
                rf_exter = pd.DataFrame(
                    {'ytrue': y3, 'ypred': exter_pred[:, 1]}
                )

    rf_res = pd.DataFrame({
        'ntree': [100, 300, 500, 700, 1000], 'valid': valid_auc,
        'test': test_auc, 'external': external_auc
    })

    # AdaBoost
    valid_auc, test_auc, external_auc = [], [], []
    for n in [100, 300, 500, 700, 1000]:
        adb = AdaBoostClassifier(n_estimators=n, random_state=1234)
        adb.fit(trainx, trainy)
        valid_pred = adb.predict_proba(validx)
        valid_auc.append(roc_auc_score(validy, valid_pred[:, 1]))

        test_pred = adb.predict_proba(testx)
        test_auc.append(roc_auc_score(testy, test_pred[:, 1]))

        exter_pred = adb.predict_proba(data3)
        external_auc.append(roc_auc_score(y3, exter_pred[:, 1]))

        if n == 100:
            adb_test = pd.DataFrame({'ytrue': testy, 'ypred': test_pred[:, 1]})
            adb_exter = pd.DataFrame({'ytrue': y3, 'ypred': exter_pred[:, 1]})
        else:
            if np.argmax(valid_auc) == len(valid_auc) - 1:
                adb_test = pd.DataFrame(
                    {'ytrue': testy, 'ypred': test_pred[:, 1]}
                )
                adb_exter = pd.DataFrame(
                    {'ytrue': y3, 'ypred': exter_pred[:, 1]}
                )

    adb_res = pd.DataFrame({
        'ntree': [100, 300, 500, 700, 1000], 'valid': valid_auc,
        'test': test_auc, 'external': external_auc
    })

    # SVM
    valid_auc, test_auc, external_auc = [], [], []
    for c in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
        svm = SVC(C=c, probability=True, random_state=1234)
        svm.fit(trainx, trainy)
        valid_pred = svm.predict_proba(validx)
        valid_auc.append(roc_auc_score(validy, valid_pred[:, 1]))

        test_pred = svm.predict_proba(testx)
        test_auc.append(roc_auc_score(testy, test_pred[:, 1]))

        exter_pred = svm.predict_proba(data3)
        external_auc.append(roc_auc_score(y3, exter_pred[:, 1]))

        if c == 1e-4:
            svm_test = pd.DataFrame({'ytrue': testy, 'ypred': test_pred[:, 1]})
            svm_exter = pd.DataFrame({'ytrue': y3, 'ypred': exter_pred[:, 1]})
        else:
            if np.argmax(valid_auc) == len(valid_auc) - 1:
                svm_test = pd.DataFrame(
                    {'ytrue': testy, 'ypred': test_pred[:, 1]}
                )
                svm_exter = pd.DataFrame(
                    {'ytrue': y3, 'ypred': exter_pred[:, 1]}
                )

    svm_res = pd.DataFrame({
        'c': [1e-4, 1e-3, 1e-2, 1e-1, 1], 'valid': valid_auc,
        'test': test_auc, 'external': external_auc
    })

    # LinearSVM
    valid_auc, test_auc, external_auc = [], [], []
    for c in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
        svm = LinearSVC(C=c, random_state=1234)
        svm.fit(trainx, trainy)
        valid_pred = svm.predict(validx)
        valid_auc.append(roc_auc_score(validy, valid_pred))

        test_pred = svm.predict(testx)
        test_auc.append(roc_auc_score(testy, test_pred))

        exter_pred = svm.predict(data3)
        external_auc.append(roc_auc_score(y3, exter_pred))

        if c == 1e-4:
            linearsvm_test = pd.DataFrame(
                {'ytrue': testy, 'ypred': test_pred}
            )
            linearsvm_exter = pd.DataFrame(
                {'ytrue': y3, 'ypred': exter_pred}
            )
        else:
            if np.argmax(valid_auc) == len(valid_auc) - 1:
                linearsvm_test = pd.DataFrame(
                    {'ytrue': testy, 'ypred': test_pred}
                )
                linearsvm_exter = pd.DataFrame(
                    {'ytrue': y3, 'ypred': exter_pred}
                )

    linearsvm_res = pd.DataFrame({
        'c': [1e-4, 1e-3, 1e-2, 1e-1, 1], 'valid': valid_auc,
        'test': test_auc, 'external': external_auc
    })

    # Logistic
    valid_auc, test_auc, external_auc = [], [], []
    for c in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
        logit = LogisticRegression(C=c, random_state=1234, n_jobs=50)
        logit.fit(trainx, trainy)
        valid_pred = logit.predict_proba(validx)
        valid_auc.append(roc_auc_score(validy, valid_pred[:, 1]))

        test_pred = logit.predict_proba(testx)
        test_auc.append(roc_auc_score(testy, test_pred[:, 1]))

        exter_pred = logit.predict_proba(data3)
        external_auc.append(roc_auc_score(y3, exter_pred[:, 1]))

        if c == 1e-4:
            logit_test = pd.DataFrame(
                {'ytrue': testy, 'ypred': test_pred[:, 1]}
            )
            logit_exter = pd.DataFrame(
                {'ytrue': y3, 'ypred': exter_pred[:, 1]}
            )
        else:
            if np.argmax(valid_auc) == len(valid_auc) - 1:
                logit_test = pd.DataFrame(
                    {'ytrue': testy, 'ypred': test_pred[:, 1]}
                )
                logit_exter = pd.DataFrame(
                    {'ytrue': y3, 'ypred': exter_pred[:, 1]}
                )

    logit_res = pd.DataFrame({
        'c': [1e-4, 1e-3, 1e-2, 1e-1, 1], 'valid': valid_auc,
        'test': test_auc, 'external': external_auc
    })

    # 修改AUC结果
    rf_res.loc[:, 'method'] = 'RF'
    rf_res.columns = ['param', 'valid', 'test', 'external', 'method']

    adb_res.loc[:, 'method'] = 'AdaBoost'
    adb_res.columns = ['param', 'valid', 'test', 'external', 'method']

    svm_res.loc[:, 'method'] = 'SVM'
    svm_res.columns = ['param', 'valid', 'test', 'external', 'method']

    linearsvm_res.loc[:, 'method'] = 'LinearSVM'
    linearsvm_res.columns = ['param', 'valid', 'test', 'external', 'method']

    logit_res.loc[:, 'method'] = 'LogisticRegression'
    logit_res.columns = ['param', 'valid', 'test', 'external', 'method']

    result = pd.concat(
        [rf_res, adb_res, svm_res, linearsvm_res, logit_res], axis=0
    )
    result = result.groupby('method').apply(
        lambda x: x.iloc[np.argmax(x.valid.values)]
    )
    result = result[['method', 'param', 'test', 'external']]
    result.to_csv('control/auc_result.csv', index=None)

    # 预测概率和标签
    rf_test.loc[:, 'method'] = 'RF'
    rf_exter.loc[:, 'method'] = 'RF'

    svm_test.loc[:, 'method'] = 'SVM'
    svm_exter.loc[:, 'method'] = 'SVM'

    linearsvm_test.loc[:, 'method'] = 'LinearSVM'
    linearsvm_exter.loc[:, 'method'] = 'LinearSVM'

    logit_test.loc[:, 'method'] = 'LogisticRegression'
    logit_exter.loc[:, 'method'] = 'LogisticRegression'

    adb_test.loc[:, 'method'] = 'AdaBoost'
    adb_exter.loc[:, 'method'] = 'AdaBoost'

    test_pred = pd.concat(
        [rf_test, adb_test, svm_test, linearsvm_test, logit_test], axis=0
    )
    exter_pred = pd.concat(
        [rf_exter, adb_exter, svm_exter, linearsvm_exter, logit_exter], axis=0
    )
    test_pred.to_csv('control/test_pred.csv', index=None)
    exter_pred.to_csv('control/exter_pred.csv', index=None)


if __name__ == '__main__':
    main()
