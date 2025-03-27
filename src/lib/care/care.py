import src.lib.care.care_functions as functions
import src.lib.care.utility as utility


def calculate_care_score(events):
    fb_sum = 0
    ws_sum = 0
    acc_sum = 0
    normal_count = 0
    anomaly_count = 0
    event_gs = []
    event_ps = []
    for event in events:
        df = event['data']

        # leave only prediction interval
        prediction = df[df['train_test'] == 'prediction']

        if not event['normal']:
            anomaly_count += 1

            # from prediction interval filter out abnormal status ids, include only 0, 2
            normal = prediction[prediction['status_type_id'].isin([0, 2])]

            g = normal['label'].values
            p = normal['prediction'].values
            fb = functions.calculate_coverage(g, p)
            fb_sum += fb
            ws = functions.calculate_earliness(p)
            ws_sum += ws
        else:
            normal_count += 1

            g = prediction['label'].values
            p = prediction['prediction'].values
            acc = functions.calculate_accuracy(g, p)
            acc_sum += acc

        s = prediction['status_type_id'].values
        p = prediction['prediction'].values

        event_g = 0 if event['normal'] else 1
        event_p = functions.calculate_reliability(s, p)
        event_gs.append(event_g)
        event_ps.append(event_p)

    efb = utility.fb_score(event_gs, event_ps)
    fb_mean = fb_sum / anomaly_count if anomaly_count > 0 else 0
    ws_mean = ws_sum / anomaly_count if anomaly_count > 0 else 0
    acc_mean = acc_sum / normal_count if normal_count > 0 else 0
    
    if all(x == 0 for x in event_ps):
        return 0
    elif acc_mean < 0.5:
        return acc_mean
    else:
        return (fb_mean + ws_mean + efb + 2 * acc_mean) / 5
