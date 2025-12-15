from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tools import get_training_data
from mne.decoding import Vectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from tools import get_freq
import numpy as np
all_events_id = {1: 'Relax', 2: 'Left', 3: 'Right', 4: 'Both', 5: 'Feet'}
no_feet = {1: 'Relax', 2: 'Left', 3: 'Right', 4: 'Both'}
left_right_events_id = {1: 'Relax', 2: 'Left', 3: 'Right'}
hands_events_id = {2: 'Left', 3: 'Right'}
merge_hands = {1: 'Relax', 2: 'hands', 3: 'hands', 4: 'Both', 5: 'Feet'}


"""
raw_fnames = [r"Konrad/KONRAD-1_sciskanie_run1_20251202_194514_raw.fif",
              r"Konrad/KONRAD-2_sciskanie_run1_20251202_203846_raw.fif",
              r"Konrad/KONRAD-4_ruszanie_run1_20251202_205706_raw.fif"]
"""
raw_fnames = [r"Kasia/kasia1_run1_20251206_185546_raw.fif"]

_, _, _, y_test_all = get_training_data(all_events_id, raw_fnames)


train_epochs, y_train, test_epochs, y_test = get_training_data(merge_hands, raw_fnames)
X_train, freqs = get_freq(train_epochs)
X_test, _ = get_freq(test_epochs)

clf = Pipeline([
    ('vectorizer', Vectorizer()),
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight='balanced'
    ))
])

clf.fit(X_train, y_train)

y_pred_test_1  = clf.predict(X_test)
print("TEST:")
print(classification_report(y_test, y_pred_test_1))
print(confusion_matrix(y_test, y_pred_test_1))

joblib.dump(clf, 'general_model.joblib')


train_epochs, y_train, test_epochs, y_test= get_training_data(hands_events_id, raw_fnames)
X_train, freqs = get_freq(train_epochs)
X_test, _ = get_freq(test_epochs)


svm = OneVsRestClassifier(SVC(kernel='rbf', probability=True, class_weight='balanced'))
clf = Pipeline([('vectorizer', Vectorizer()),
                ('scaler', StandardScaler()),
                ("SVM", svm)])


clf.fit(X_train, y_train)

y_pred_test_2  = clf.predict(X_test)
print("TEST:")
print(classification_report(y_test, y_pred_test_2 ))
print(confusion_matrix(y_test, y_pred_test_2 ))

joblib.dump(clf, 'hands_model.joblib')
