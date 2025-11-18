"""Calibration scaffold

Placeholder for calibration logic (Platt/Isotonic).
"""

def calibrate(preds_csv, out_json):
    # TODO: implement calibration fit
    with open(out_json, 'w') as f:
        f.write('{}')
    print(f'Wrote calibration to {out_json}')
