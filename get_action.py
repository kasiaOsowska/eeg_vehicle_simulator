import numpy as np
import joblib
from mne import concatenate_raws
from mne.io import read_raw_fif
from tools import split_epochs_into_segments, get_freq, get_epochs
import mne

general_classifier = joblib.load('general_model.joblib')
hands_classifier = joblib.load('hands_model.joblib')

EPOCHS_BY_EVENT_ID = {}
ALL_EPOCHS = None


def initialize_epochs(all_events_id):
    raw_fnames = [r"Kasia/kasia2_run1_20251206_191128_raw.fif"]


    epochs = get_epochs(all_events_id, raw_fnames)

    segment_length = 2.0
    step = 0.5

    splitted_epochs = split_epochs_into_segments(epochs, segment_length, step)

    global EPOCHS_BY_EVENT_ID
    for event_id in range(1, 6):
        indices = np.where(splitted_epochs.events[:, -1] == event_id)
        if len(indices) > 0:
            EPOCHS_BY_EVENT_ID[event_id] = indices

    return splitted_epochs



def get_eeg_action(event_id, all_events_id):

    global ALL_EPOCHS, EPOCHS_BY_EVENT_ID

    if ALL_EPOCHS is None:
        ALL_EPOCHS = initialize_epochs(all_events_id)

    if event_id not in EPOCHS_BY_EVENT_ID or len(EPOCHS_BY_EVENT_ID[event_id]) == 0:
        raise ValueError(f"No epochs found for event ID {event_id}.")
    try:
        random_index = np.random.choice(EPOCHS_BY_EVENT_ID[event_id][0])
    except ValueError:
        raise ValueError(f"No epochs found for event ID {event_id}.")

    epoch = ALL_EPOCHS[random_index]
    X_train, _ = get_freq(epoch)
    classification = general_classifier.predict(X_train)
    print("-----------classification-----------")
    print(classification)
    if classification == 3 or classification == 2:
        classification = hands_classifier.predict(X_train)
        print("-----------classification hands-----------")
        print(classification)

    print("true event: " + str(all_events_id[event_id]))
    print("predicted "+ str(all_events_id[classification[0]]))

    steer = 0
    gas = 0
    brake = 0
    if classification == 1:  # relaks
        pass
    elif classification == 2:  # Lefts
        steer = -0.1
    elif classification == 3:  # Right
        steer = 0.1
    elif classification == 4:  # Both hands
        gas = 0.05
    elif classification == 5:  # Both feet
        brake = 1

    action = np.array([steer, gas, brake], dtype=np.float32)
    return action

