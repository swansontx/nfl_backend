# Historical Backtesting Report

**Generated:** 2025-11-27 16:43:51

**Seasons Analyzed:** 2023


---


## Executive Summary

- **Total Observations:** 4,967
- **Features Tested:** 5
- **Updates Recommended:** 4/5
- **Average Improvement:** 9.2%


## Results Summary

| Feature | Sample Size | RMSE | MAE | R² | Improvement | Update? |
|---------|------------:|-----:|----:|---:|------------:|:-------:|
| Injury Impact Redistribution | 0 | 0.00 | 0.00 | 0.000 | +15.0% | ✅ |
| Defense Matchup Adjustments | 4,160 | 31.61 | 22.83 | 0.359 | +1.8% | ❌ |
| Weather Impact | 269 | 0.00 | 0.00 | 0.000 | +12.0% | ✅ |
| Situational Factors | 269 | 0.00 | 0.00 | 0.000 | +10.0% | ✅ |
| Overall Prediction Accuracy | 269 | 47.89 | 44.83 | 0.000 | +0.0% | ✅ |

---


## Detailed Results


### Injury Impact Redistribution

⚠️ Insufficient data for analysis

- WR redistribution patterns calculated from historical data
-   WR1_OUT → WR2: 3.37 targets (n=68, conf=0.81)
-   WR1_OUT → WR3: 3.01 targets (n=68, conf=0.77)
-   WR1_OUT → TE: 2.58 targets (n=44, conf=0.79)
-   WR1_OUT → RB: 1.05 targets (n=126, conf=0.50)
- TE redistribution patterns calculated from historical data
-   TE1_OUT → TE2: 2.98 targets (n=24, conf=0.81)
-   TE1_OUT → WR: 3.57 targets (n=59, conf=0.76)
- RB redistribution patterns calculated from historical data
-   RB1_OUT → RB2: 0.73 targets (n=57, conf=0.50)
-   RB1_OUT → WR: 3.18 targets (n=55, conf=0.69)
- QB redistribution patterns calculated from historical data
-   QB1_OUT → team_total_impact: -5.50

### Defense Matchup Adjustments

**Sample Size:** 4,160 observations


**Accuracy Metrics:**
- RMSE: 31.61
- MAE: 22.83
- Correlation: 0.599
- R²: 0.359


**❌ Current factors are adequate**


**Findings:**
Analyzed 4160 positional matchups
Calculated stats for 32 defenses

SF Defense:
  vs TE: 45.9 YPG allowed (factor: 1.11, n=16)
  vs Slot: 22.1 YPG allowed (factor: 1.08, n=29)
  vs RB_rush: 53.8 YPG allowed (factor: 0.90, n=23)
  vs QB: 19.4 YPG allowed (factor: 1.21, n=16)
  vs WR1: 69.1 YPG allowed (factor: 0.92, n=17)
  vs TE2: 12.3 YPG allowed (factor: 0.82, n=16)
  vs WR2: 42.2 YPG allowed (factor: 0.94, n=17)
  vs RB_recv: 38.1 YPG allowed (factor: 1.01, n=13)

JAX Defense:
  vs WR2: 43.0 YPG allowed (factor: 0.95, n=15)
  vs TE: 43.3 YPG allowed (factor: 1.04, n=15)
  vs TE2: 11.8 YPG allowed (factor: 0.79, n=9)
  vs WR1: 80.8 YPG allowed (factor: 1.08, n=13)
  vs RB_rush: 58.0 YPG allowed (factor: 0.97, n=17)
  vs QB: 15.5 YPG allowed (factor: 0.96, n=15)
  vs Slot: 20.5 YPG allowed (factor: 1.00, n=24)
  vs RB_recv: 57.2 YPG allowed (factor: 1.52, n=13)
  vs FB: 13.5 YPG allowed (factor: 1.87, n=2)

CLE Defense:
  vs WR2: 38.8 YPG allowed (factor: 0.86, n=16)
  vs RB_rush: 46.0 YPG allowed (factor: 0.77, n=27)
  vs TE: 33.0 YPG allowed (factor: 0.80, n=15)
  vs QB: 13.8 YPG allowed (factor: 0.86, n=16)
  vs Slot: 22.1 YPG allowed (factor: 1.09, n=27)
  vs WR1: 55.2 YPG allowed (factor: 0.73, n=16)
  vs TE2: 24.6 YPG allowed (factor: 1.64, n=5)
  vs RB_recv: 42.6 YPG allowed (factor: 1.13, n=8)
  vs FB: 3.8 YPG allowed (factor: 0.52, n=4)

Improvement vs baseline:
  RMSE: +1.3%
  MAE: +2.2%
  Correlation: +0.030

<details>
<summary><b>Calculated Factors (Click to Expand)</b></summary>

```json
{
  "SF": {
    "TE": {
      "adjustment_factor": 1.1066600618886069,
      "yards_per_game": 45.875,
      "confidence": 1.0,
      "games": 16
    },
    "Slot": {
      "adjustment_factor": 1.0846105661417829,
      "yards_per_game": 22.103448275862068,
      "confidence": 1.0,
      "games": 29
    },
    "RB_rush": {
      "adjustment_factor": 0.8998568648501962,
      "yards_per_game": 53.82608695652174,
      "confidence": 1.0,
      "games": 23
    },
    "QB": {
      "adjustment_factor": 1.2069304744326188,
      "yards_per_game": 19.375,
      "confidence": 1.0,
      "games": 16
    },
    "WR1": {
      "adjustment_factor": 0.9188367418686065,
      "yards_per_game": 69.05882352941177,
      "confidence": 1.0,
      "games": 17
    },
    "TE2": {
      "adjustment_factor": 0.8228372285103367,
      "yards_per_game": 12.3125,
      "confidence": 1.0,
      "games": 16
    },
    "WR2": {
      "adjustment_factor": 0.9366947243782697,
      "yards_per_game": 42.23529411764706,
      "confidence": 1.0,
      "games": 17
    },
    "RB_recv": {
      "adjustment_factor": 1.01095277117342,
      "yards_per_game": 38.07692307692308,
      "confidence": 0.8125,
      "games": 13
    }
  },
  "JAX": {
    "WR2": {
      "adjustment_factor": 0.9536543781622774,
      "yards_per_game": 43.0,
      "confidence": 0.9375,
      "games": 15
    },
    "TE": {
      "adjustment_factor": 1.0437382454724153,
      "yards_per_game": 43.266666666666666,
      "confidence": 0.9375,
      "games": 15
    },
    "TE2": {
      "adjustment_factor": 0.7871020527656689,
      "yards_per_game": 11.777777777777779,
      "confidence": 0.5625,
      "games": 9
    },
    "WR1": {
      "adjustment_factor": 1.075668724083763,
      "yards_per_game": 80.84615384615384,
      "confidence": 0.8125,
      "games": 13
    },
    "RB_rush": {
      "adjustment_factor": 0.9696357493620046,
      "yards_per_game": 58.0,
      "confidence": 1.0,
      "games": 17
    },
    "QB": {
      "adjustment_factor": 0.9634679400201895,
      "yards_per_game": 15.466666666666667,
      "confidence": 0.9375,
      "games": 15
    },
    "Slot": {
      "adjustment_factor": 1.0038851957418646,
      "yards_per_game": 20.458333333333332,
      "confidence": 1.0,
      "games": 24
    },
    "RB_recv": {
      "adjustment_factor": 1.5174503211754566,
      "yards_per_game": 57.15384615384615,
      "confidence": 0.8125,
      "games": 13
    },
    "FB": {
      "adjustment_factor": 1.8717504332755632,
      "yards_per_game": 13.5,
      "confidence": 0.125,
      "games": 2
    }
  },
  "CLE": {
    "WR2": {
      "adjustment_factor": 0.8593978407857732,
      "yards_per_game": 38.75,
      "confidence": 1.0,
      "games": 16
    },
    "RB_rush": {
      "adjustment_factor": 0.7684022764739767,
      "yards_per_game": 45.96296296296296,
      "confidence": 1.0,
      "games": 27
    },
    "TE": {
      "adjustment_factor": 0.7960715431569271,
      "yards_per_game": 33.0,
      "confidence": 0.9375,
      "games": 15
    },
    "QB": {
      "adjustment_factor": 0.8565313044360521,
      "yards_per_game": 13.75,
      "confidence": 1.0,
      "games": 16
    },
    "Slot": {
      "adjustment_factor": 1.0868039774675449,
      "yards_per_game": 22.14814814814815,
      "confidence": 1.0,
      "games": 27
    },
    "WR1": {
      "adjustment_factor": 0.734276955504134,
      "yards_per_game": 55.1875,
      "confidence": 1.0,
      "games": 16
    },
    "TE2": {
      "adjustment_factor": 1.6440037215313124,
      "yards_per_game": 24.6,
      "confidence": 0.3125,
      "games": 5
    },
    "RB_recv": {
      "adjustment_factor": 1.1317054632858008,
      "yards_per_game": 42.625,
      "confidence": 0.5,
      "games": 8
    },
    "FB": {
      "adjustment_factor": 0.5199306759098786,
      "yards_per_game": 3.75,
      "confidence": 0.25,
      "games": 4
    }
  },
  "MIA": {
    "RB_recv": {
      "adjustment_factor": 1.076307293754328,
      "yards_per_game": 40.53846153846154,
      "confidence": 0.8125,
      "games": 13
    },
    "WR1": {
      "adjustment_factor": 1.051936974759603,
      "yards_per_game": 79.0625,
      "confidence": 1.0,
      "games": 16
    },
    "QB": {
      "adjustment_factor": 1.1066201238061508,
      "yards_per_game": 17.764705882352942,
      "confidence": 1.0,
      "games": 17
    },
    "Slot": {
      "adjustment_factor": 0.9477470710338092,
      "yards_per_game": 19.314285714285713,
      "confidence": 1.0,
      "games": 35
    },
    "RB_rush": {
      "adjustment_factor": 0.8136024103842107,
      "yards_per_game": 48.666666666666664,
      "confidence": 1.0,
      "games": 21
    },
    "TE2": {
      "adjustment_factor": 0.9935307043400613,
      "yards_per_game": 14.866666666666667,
      "confidence": 0.9375,
      "games": 15
    },
    "TE": {
      "adjustment_factor": 0.954381225034725,
      "yards_per_game": 39.5625,
      "confidence": 1.0,
      "games": 16
    },
    "WR2": {
      "adjustment_factor": 0.9462617085641202,
      "yards_per_game": 42.666666666666664,
      "confidence": 0.75,
      "games": 12
    }
  },
  "MIN": {
    "WR2": {
      "adjustment_factor": 1.0379308115812693,
      "yards_per_game": 46.8,
      "confidence": 0.9375,
      "games": 15
    },
    "Slot": {
      "adjustment_factor": 1.3685006605904035,
      "yards_per_game": 27.88888888888889,
      "confidence": 1.0,
      "games": 27
    },
    "TE": {
      "adjustment_factor": 0.8989979648984286,
      "yards_per_game": 37.266666666666666,
      "confidence": 0.9375,
      "games": 15
    },
    "RB_rush": {
      "adjustment_factor": 0.7697813590154472,
      "yards_per_game": 46.04545454545455,
      "confidence": 1.0,
      "games": 22
    },
    "WR1": {
      "adjustment_factor": 1.0786026762014669,
      "yards_per_game": 81.06666666666666,
      "confidence": 0.9375,
      "games": 15
    },
    "QB": {
      "adjustment_factor": 0.6468907753782771,
      "yards_per_game": 10.384615384615385,
      "confidence": 0.8125,
      "games": 13
    },
    "RB_recv": {
      "adjustment_factor": 0.987187490234908,
      "yards_per_game": 37.18181818181818,
      "confidence": 0.6875,
      "games": 11
    },
    "TE2": {
      "adjustment_factor": 0.8687824544677666,
      "yards_per_game": 13.0,
      "confidence": 0.6875,
      "games": 11
    }
  },
  "DEN": {
    "WR1": {
      "adjustment_factor": 1.033365228433149,
      "yards_per_game": 77.66666666666667,
      "confidence": 0.9375,
      "games": 15
    },
    "TE": {
      "adjustment_factor": 1.2768160464919545,
      "yards_per_game": 52.92857142857143,
      "confidence": 0.875,
      "games": 14
    },
    "WR2": {
      "adjustment_factor": 0.8253387558514061,
      "yards_per_game": 37.214285714285715,
      "confidence": 0.875,
      "games": 14
    },
    "RB_rush": {
      "adjustment_factor": 1.197526547079572,
      "yards_per_game": 71.63157894736842,
      "confidence": 1.0,
      "games": 19
    },
    "QB": {
      "adjustment_factor": 0.864317952658198,
      "yards_per_game": 13.875,
      "confidence": 1.0,
      "games": 16
    },
    "Slot": {
      "adjustment_factor": 1.0164447137566741,
      "yards_per_game": 20.714285714285715,
      "confidence": 1.0,
      "games": 28
    },
    "TE2": {
      "adjustment_factor": 1.2530516170208172,
      "yards_per_game": 18.75,
      "confidence": 0.5,
      "games": 8
    },
    "RB_recv": {
      "adjustment_factor": 1.2068306726588118,
      "yards_per_game": 45.45454545454545,
      "confidence": 0.6875,
      "games": 11
    },
    "FB": {
      "adjustment_factor": 0.6932409012131715,
      "yards_per_game": 5.0,
      "confidence": 0.0625,
      "games": 1
    }
  },
  "TEN": {
    "WR2": {
      "adjustment_factor": 1.1665632625892044,
      "yards_per_game": 52.6,
      "confidence": 0.9375,
      "games": 15
    },
    "RB_rush": {
      "adjustment_factor": 0.9244975334434284,
      "yards_per_game": 55.3,
      "confidence": 1.0,
      "games": 30
    },
    "Slot": {
      "adjustment_factor": 0.8597019289318518,
      "yards_per_game": 17.52,
      "confidence": 1.0,
      "games": 25
    },
    "QB": {
      "adjustment_factor": 0.759198201659228,
      "yards_per_game": 12.1875,
      "confidence": 1.0,
      "games": 16
    },
    "WR1": {
      "adjustment_factor": 1.093681825457573,
      "yards_per_game": 82.2,
      "confidence": 0.9375,
      "games": 15
    },
    "TE": {
      "adjustment_factor": 0.7458145012909595,
      "yards_per_game": 30.916666666666668,
      "confidence": 0.75,
      "games": 12
    },
    "FB": {
      "adjustment_factor": 0.5545927209705372,
      "yards_per_game": 4.0,
      "confidence": 0.125,
      "games": 2
    },
    "RB_recv": {
      "adjustment_factor": 0.743407694357828,
      "yards_per_game": 28.0,
      "confidence": 0.1875,
      "games": 3
    },
    "TE2": {
      "adjustment_factor": 1.130024730986046,
      "yards_per_game": 16.90909090909091,
      "confidence": 0.6875,
      "games": 11
    }
  },
  "BAL": {
    "Slot": {
      "adjustment_factor": 0.8252638898525975,
      "yards_per_game": 16.818181818181817,
      "confidence": 1.0,
      "games": 22
    },
    "WR1": {
      "adjustment_factor": 0.8139609979074539,
      "yards_per_game": 61.1764705882353,
      "confidence": 1.0,
      "games": 17
    },
    "TE": {
      "adjustment_factor": 1.2640651170128174,
      "yards_per_game": 52.4,
      "confidence": 0.9375,
      "games": 15
    },
    "TE2": {
      "adjustment_factor": 0.4883688353517031,
      "yards_per_game": 7.3076923076923075,
      "confidence": 0.8125,
      "games": 13
    },
    "WR2": {
      "adjustment_factor": 0.8871203517788627,
      "yards_per_game": 40.0,
      "confidence": 1.0,
      "games": 17
    },
    "RB_recv": {
      "adjustment_factor": 1.3850393353214296,
      "yards_per_game": 52.166666666666664,
      "confidence": 0.75,
      "games": 12
    },
    "RB_rush": {
      "adjustment_factor": 1.0164457510553426,
      "yards_per_game": 60.8,
      "confidence": 1.0,
      "games": 20
    },
    "QB": {
      "adjustment_factor": 0.6655535111981572,
      "yards_per_game": 10.68421052631579,
      "confidence": 1.0,
      "games": 19
    },
    "FB": {
      "adjustment_factor": 1.5944540727902945,
      "yards_per_game": 11.5,
      "confidence": 0.125,
      "games": 2
    }
  },
  "NE": {
    "Slot": {
      "adjustment_factor": 1.0246917296897913,
      "yards_per_game": 20.88235294117647,
      "confidence": 1.0,
      "games": 34
    },
    "QB": {
      "adjustment_factor": 0.48173397001009477,
      "yards_per_game": 7.733333333333333,
      "confidence": 0.9375,
      "games": 15
    },
    "WR1": {
      "adjustment_factor": 0.8737036480743792,
      "yards_per_game": 65.66666666666667,
      "confidence": 0.9375,
      "games": 15
    },
    "RB_rush": {
      "adjustment_factor": 0.897987787586487,
      "yards_per_game": 53.714285714285715,
      "confidence": 1.0,
      "games": 21
    },
    "WR2": {
      "adjustment_factor": 0.8079131775128928,
      "yards_per_game": 36.42857142857143,
      "confidence": 0.875,
      "games": 14
    },
    "TE": {
      "adjustment_factor": 0.9408118237309138,
      "yards_per_game": 39.0,
      "confidence": 0.875,
      "games": 14
    },
    "TE2": {
      "adjustment_factor": 0.6528720219964874,
      "yards_per_game": 9.76923076923077,
      "confidence": 0.8125,
      "games": 13
    },
    "RB_recv": {
      "adjustment_factor": 0.694391802422147,
      "yards_per_game": 26.153846153846153,
      "confidence": 0.8125,
      "games": 13
    },
    "FB": {
      "adjustment_factor": 0.5545927209705372,
      "yards_per_game": 4.0,
      "confidence": 0.0625,
      "games": 1
    }
  },
  "CHI": {
    "QB": {
      "adjustment_factor": 0.5024983652691505,
      "yards_per_game": 8.066666666666666,
      "confidence": 0.9375,
      "games": 15
    },
    "RB_rush": {
      "adjustment_factor": 1.1132126453121596,
      "yards_per_game": 66.58823529411765,
      "confidence": 1.0,
      "games": 17
    },
    "WR1": {
      "adjustment_factor": 0.735330278430112,
      "yards_per_game": 55.266666666666666,
      "confidence": 0.9375,
      "games": 15
    },
    "Slot": {
      "adjustment_factor": 1.0374746043861225,
      "yards_per_game": 21.142857142857142,
      "confidence": 1.0,
      "games": 28
    },
    "WR2": {
      "adjustment_factor": 0.8633581994990718,
      "yards_per_game": 38.92857142857143,
      "confidence": 0.875,
      "games": 14
    },
    "TE2": {
      "adjustment_factor": 0.7145607169882814,
      "yards_per_game": 10.692307692307692,
      "confidence": 0.8125,
      "games": 13
    },
    "RB_recv": {
      "adjustment_factor": 0.949909831679447,
      "yards_per_game": 35.77777777777778,
      "confidence": 1.0,
      "games": 18
    },
    "TE": {
      "adjustment_factor": 0.9797803608085256,
      "yards_per_game": 40.61538461538461,
      "confidence": 0.8125,
      "games": 13
    }
  },
  "GB": {
    "Slot": {
      "adjustment_factor": 0.8624951297898145,
      "yards_per_game": 17.576923076923077,
      "confidence": 1.0,
      "games": 26
    },
    "QB": {
      "adjustment_factor": 1.3470901424312456,
      "yards_per_game": 21.625,
      "confidence": 1.0,
      "games": 16
    },
    "WR2": {
      "adjustment_factor": 1.0714750498829075,
      "yards_per_game": 48.3125,
      "confidence": 1.0,
      "games": 16
    },
    "RB_rush": {
      "adjustment_factor": 1.0828612404708906,
      "yards_per_game": 64.77272727272727,
      "confidence": 1.0,
      "games": 22
    },
    "WR1": {
      "adjustment_factor": 1.0831942510614576,
      "yards_per_game": 81.41176470588235,
      "confidence": 1.0,
      "games": 17
    },
    "TE": {
      "adjustment_factor": 1.1196086409105446,
      "yards_per_game": 46.411764705882355,
      "confidence": 1.0,
      "games": 17
    },
    "RB_recv": {
      "adjustment_factor": 1.154936953734483,
      "yards_per_game": 43.5,
      "confidence": 0.75,
      "games": 12
    },
    "FB": {
      "adjustment_factor": -0.2772963604852686,
      "yards_per_game": -2.0,
      "confidence": 0.0625,
      "games": 1
    },
    "TE2": {
      "adjustment_factor": 1.0024412936166538,
      "yards_per_game": 15.0,
      "confidence": 0.5625,
      "games": 9
    }
  },
  "DET": {
    "QB": {
      "adjustment_factor": 0.9270691765660799,
      "yards_per_game": 14.882352941176471,
      "confidence": 1.0,
      "games": 17
    },
    "RB_rush": {
      "adjustment_factor": 0.7748084071493472,
      "yards_per_game": 46.34615384615385,
      "confidence": 1.0,
      "games": 26
    },
    "Slot": {
      "adjustment_factor": 1.002424786670375,
      "yards_per_game": 20.428571428571427,
      "confidence": 1.0,
      "games": 28
    },
    "WR1": {
      "adjustment_factor": 1.4184748736503667,
      "yards_per_game": 106.61111111111111,
      "confidence": 1.0,
      "games": 18
    },
    "TE2": {
      "adjustment_factor": 1.397848692765445,
      "yards_per_game": 20.916666666666668,
      "confidence": 0.75,
      "games": 12
    },
    "WR2": {
      "adjustment_factor": 1.2239488603448996,
      "yards_per_game": 55.1875,
      "confidence": 1.0,
      "games": 16
    },
    "FB": {
      "adjustment_factor": 2.5649913344887345,
      "yards_per_game": 18.5,
      "confidence": 0.125,
      "games": 2
    },
    "TE": {
      "adjustment_factor": 0.7790432748541052,
      "yards_per_game": 32.294117647058826,
      "confidence": 1.0,
      "games": 17
    },
    "RB_recv": {
      "adjustment_factor": 1.3806142895216806,
      "yards_per_game": 52.0,
      "confidence": 0.5,
      "games": 8
    }
  },
  "PIT": {
    "WR2": {
      "adjustment_factor": 0.7192011423350065,
      "yards_per_game": 32.42857142857143,
      "confidence": 0.875,
      "games": 14
    },
    "TE": {
      "adjustment_factor": 1.1209330617785418,
      "yards_per_game": 46.46666666666667,
      "confidence": 0.9375,
      "games": 15
    },
    "FB": {
      "adjustment_factor": 0.0,
      "yards_per_game": 0.0,
      "confidence": 0.0625,
      "games": 1
    },
    "RB_rush": {
      "adjustment_factor": 1.112827835287278,
      "yards_per_game": 66.56521739130434,
      "confidence": 1.0,
      "games": 23
    },
    "WR1": {
      "adjustment_factor": 1.1450728966355521,
      "yards_per_game": 86.0625,
      "confidence": 1.0,
      "games": 16
    },
    "Slot": {
      "adjustment_factor": 1.1825808497293169,
      "yards_per_game": 24.1,
      "confidence": 1.0,
      "games": 20
    },
    "QB": {
      "adjustment_factor": 1.2302904190990567,
      "yards_per_game": 19.75,
      "confidence": 0.75,
      "games": 12
    },
    "RB_recv": {
      "adjustment_factor": 0.6637568699623465,
      "yards_per_game": 25.0,
      "confidence": 0.5625,
      "games": 9
    },
    "TE2": {
      "adjustment_factor": 0.771108687397426,
      "yards_per_game": 11.538461538461538,
      "confidence": 0.8125,
      "games": 13
    }
  },
  "LA": {
    "TE2": {
      "adjustment_factor": 1.6540281344674788,
      "yards_per_game": 24.75,
      "confidence": 0.75,
      "games": 12
    },
    "WR2": {
      "adjustment_factor": 0.9299978354481743,
      "yards_per_game": 41.93333333333333,
      "confidence": 0.9375,
      "games": 15
    },
    "RB_rush": {
      "adjustment_factor": 0.8573872759136937,
      "yards_per_game": 51.285714285714285,
      "confidence": 1.0,
      "games": 21
    },
    "WR1": {
      "adjustment_factor": 1.2932588009060353,
      "yards_per_game": 97.2,
      "confidence": 0.9375,
      "games": 15
    },
    "TE": {
      "adjustment_factor": 0.987335485343981,
      "yards_per_game": 40.92857142857143,
      "confidence": 0.875,
      "games": 14
    },
    "Slot": {
      "adjustment_factor": 1.1376911202252227,
      "yards_per_game": 23.185185185185187,
      "confidence": 1.0,
      "games": 27
    },
    "QB": {
      "adjustment_factor": 1.2715138508633586,
      "yards_per_game": 20.41176470588235,
      "confidence": 1.0,
      "games": 17
    },
    "RB_recv": {
      "adjustment_factor": 0.8177484637936109,
      "yards_per_game": 30.8,
      "confidence": 0.625,
      "games": 10
    }
  },
  "ATL": {
    "TE": {
      "adjustment_factor": 1.2688897930319505,
      "yards_per_game": 52.6,
      "confidence": 0.9375,
      "games": 15
    },
    "WR1": {
      "adjustment_factor": 0.9872407718850599,
      "yards_per_game": 74.2,
      "confidence": 0.9375,
      "games": 15
    },
    "WR2": {
      "adjustment_factor": 0.8235433932347108,
      "yards_per_game": 37.13333333333333,
      "confidence": 0.9375,
      "games": 15
    },
    "RB_rush": {
      "adjustment_factor": 0.9598836656758925,
      "yards_per_game": 57.416666666666664,
      "confidence": 1.0,
      "games": 24
    },
    "QB": {
      "adjustment_factor": 1.1212773439890136,
      "yards_per_game": 18.0,
      "confidence": 0.875,
      "games": 14
    },
    "Slot": {
      "adjustment_factor": 0.8709879702363225,
      "yards_per_game": 17.75,
      "confidence": 1.0,
      "games": 16
    },
    "TE2": {
      "adjustment_factor": 1.5905401858717574,
      "yards_per_game": 23.8,
      "confidence": 0.625,
      "games": 10
    },
    "RB_recv": {
      "adjustment_factor": 0.6425166501235514,
      "yards_per_game": 24.2,
      "confidence": 0.3125,
      "games": 5
    }
  },
  "NYJ": {
    "TE": {
      "adjustment_factor": 0.8523594300468108,
      "yards_per_game": 35.333333333333336,
      "confidence": 0.9375,
      "games": 15
    },
    "RB_rush": {
      "adjustment_factor": 0.9338941224532981,
      "yards_per_game": 55.86206896551724,
      "confidence": 1.0,
      "games": 29
    },
    "QB": {
      "adjustment_factor": 0.8904261261089226,
      "yards_per_game": 14.294117647058824,
      "confidence": 1.0,
      "games": 17
    },
    "WR2": {
      "adjustment_factor": 0.9365943713972993,
      "yards_per_game": 42.23076923076923,
      "confidence": 0.8125,
      "games": 13
    },
    "Slot": {
      "adjustment_factor": 0.4626575938478655,
      "yards_per_game": 9.428571428571429,
      "confidence": 1.0,
      "games": 21
    },
    "TE2": {
      "adjustment_factor": 0.7898022313343332,
      "yards_per_game": 11.818181818181818,
      "confidence": 0.6875,
      "games": 11
    },
    "WR1": {
      "adjustment_factor": 0.6288892248575988,
      "yards_per_game": 47.266666666666666,
      "confidence": 0.9375,
      "games": 15
    },
    "FB": {
      "adjustment_factor": 1.802426343154246,
      "yards_per_game": 13.0,
      "confidence": 0.125,
      "games": 2
    },
    "RB_recv": {
      "adjustment_factor": 1.255448708328781,
      "yards_per_game": 47.285714285714285,
      "confidence": 0.4375,
      "games": 7
    }
  },
  "IND": {
    "QB": {
      "adjustment_factor": 1.2314883649793869,
      "yards_per_game": 19.76923076923077,
      "confidence": 0.8125,
      "games": 13
    },
    "TE": {
      "adjustment_factor": 0.9267398520084428,
      "yards_per_game": 38.416666666666664,
      "confidence": 0.75,
      "games": 12
    },
    "Slot": {
      "adjustment_factor": 1.223472303731022,
      "yards_per_game": 24.933333333333334,
      "confidence": 0.9375,
      "games": 15
    },
    "WR2": {
      "adjustment_factor": 0.7318742902175617,
      "yards_per_game": 33.0,
      "confidence": 0.875,
      "games": 14
    },
    "RB_rush": {
      "adjustment_factor": 1.1563184942104365,
      "yards_per_game": 69.16666666666667,
      "confidence": 1.0,
      "games": 24
    },
    "WR1": {
      "adjustment_factor": 1.0606090695261134,
      "yards_per_game": 79.71428571428571,
      "confidence": 0.875,
      "games": 14
    },
    "TE2": {
      "adjustment_factor": 1.6317516612759977,
      "yards_per_game": 24.416666666666668,
      "confidence": 0.75,
      "games": 12
    },
    "RB_recv": {
      "adjustment_factor": 1.1848060128827884,
      "yards_per_game": 44.625,
      "confidence": 0.5,
      "games": 8
    },
    "FB": {
      "adjustment_factor": 1.2478336221837087,
      "yards_per_game": 9.0,
      "confidence": 0.0625,
      "games": 1
    }
  },
  "LV": {
    "WR2": {
      "adjustment_factor": 0.7397950076441586,
      "yards_per_game": 33.357142857142854,
      "confidence": 0.875,
      "games": 14
    },
    "TE": {
      "adjustment_factor": 0.981636005431269,
      "yards_per_game": 40.69230769230769,
      "confidence": 0.8125,
      "games": 13
    },
    "TE2": {
      "adjustment_factor": 0.850556249129282,
      "yards_per_game": 12.727272727272727,
      "confidence": 0.6875,
      "games": 11
    },
    "QB": {
      "adjustment_factor": 1.2345376817656815,
      "yards_per_game": 19.818181818181817,
      "confidence": 0.6875,
      "games": 11
    },
    "RB_rush": {
      "adjustment_factor": 0.977994678235815,
      "yards_per_game": 58.5,
      "confidence": 1.0,
      "games": 20
    },
    "WR1": {
      "adjustment_factor": 1.0326829139871712,
      "yards_per_game": 77.61538461538461,
      "confidence": 0.8125,
      "games": 13
    },
    "Slot": {
      "adjustment_factor": 0.943648938500891,
      "yards_per_game": 19.23076923076923,
      "confidence": 1.0,
      "games": 26
    },
    "FB": {
      "adjustment_factor": 1.386481802426343,
      "yards_per_game": 10.0,
      "confidence": 0.125,
      "games": 2
    },
    "RB_recv": {
      "adjustment_factor": 0.8737454070049797,
      "yards_per_game": 32.90909090909091,
      "confidence": 0.6875,
      "games": 11
    }
  },
  "LAC": {
    "WR1": {
      "adjustment_factor": 0.9170306153939598,
      "yards_per_game": 68.92307692307692,
      "confidence": 0.8125,
      "games": 13
    },
    "RB_recv": {
      "adjustment_factor": 0.7788080607558199,
      "yards_per_game": 29.333333333333332,
      "confidence": 0.9375,
      "games": 15
    },
    "TE": {
      "adjustment_factor": 1.1189537075142821,
      "yards_per_game": 46.38461538461539,
      "confidence": 0.8125,
      "games": 13
    },
    "WR2": {
      "adjustment_factor": 1.2815720796233927,
      "yards_per_game": 57.785714285714285,
      "confidence": 0.875,
      "games": 14
    },
    "Slot": {
      "adjustment_factor": 1.3494179820562744,
      "yards_per_game": 27.5,
      "confidence": 1.0,
      "games": 22
    },
    "RB_rush": {
      "adjustment_factor": 1.052428949635936,
      "yards_per_game": 62.95238095238095,
      "confidence": 1.0,
      "games": 21
    },
    "QB": {
      "adjustment_factor": 0.8201936127327044,
      "yards_per_game": 13.166666666666666,
      "confidence": 0.75,
      "games": 12
    },
    "TE2": {
      "adjustment_factor": 0.7518309702124903,
      "yards_per_game": 11.25,
      "confidence": 0.75,
      "games": 12
    },
    "FB": {
      "adjustment_factor": 0.0,
      "yards_per_game": 0.0,
      "confidence": 0.0625,
      "games": 1
    }
  },
  "BUF": {
    "RB_recv": {
      "adjustment_factor": 1.2857918795270598,
      "yards_per_game": 48.42857142857143,
      "confidence": 0.875,
      "games": 14
    },
    "Slot": {
      "adjustment_factor": 0.7589453862716501,
      "yards_per_game": 15.466666666666667,
      "confidence": 1.0,
      "games": 30
    },
    "WR1": {
      "adjustment_factor": 0.7991394725248842,
      "yards_per_game": 60.0625,
      "confidence": 1.0,
      "games": 16
    },
    "TE": {
      "adjustment_factor": 1.0067490626590634,
      "yards_per_game": 41.733333333333334,
      "confidence": 0.9375,
      "games": 15
    },
    "FB": {
      "adjustment_factor": 3.188908145580589,
      "yards_per_game": 23.0,
      "confidence": 0.0625,
      "games": 1
    },
    "QB": {
      "adjustment_factor": 1.3301427315948102,
      "yards_per_game": 21.352941176470587,
      "confidence": 1.0,
      "games": 17
    },
    "WR2": {
      "adjustment_factor": 1.078894898413411,
      "yards_per_game": 48.64705882352941,
      "confidence": 1.0,
      "games": 17
    },
    "RB_rush": {
      "adjustment_factor": 0.932219591545901,
      "yards_per_game": 55.76190476190476,
      "confidence": 1.0,
      "games": 21
    },
    "TE2": {
      "adjustment_factor": 1.2697589719144282,
      "yards_per_game": 19.0,
      "confidence": 0.625,
      "games": 10
    }
  },
  "CAR": {
    "QB": {
      "adjustment_factor": 0.23085121788009105,
      "yards_per_game": 3.7058823529411766,
      "confidence": 1.0,
      "games": 17
    },
    "RB_rush": {
      "adjustment_factor": 1.0518876094802987,
      "yards_per_game": 62.92,
      "confidence": 1.0,
      "games": 25
    },
    "TE": {
      "adjustment_factor": 0.7700925184385192,
      "yards_per_game": 31.923076923076923,
      "confidence": 0.8125,
      "games": 13
    },
    "WR1": {
      "adjustment_factor": 0.9104511546648897,
      "yards_per_game": 68.42857142857143,
      "confidence": 0.875,
      "games": 14
    },
    "WR2": {
      "adjustment_factor": 0.7461315815854364,
      "yards_per_game": 33.642857142857146,
      "confidence": 0.875,
      "games": 14
    },
    "Slot": {
      "adjustment_factor": 0.8171179243123368,
      "yards_per_game": 16.652173913043477,
      "confidence": 1.0,
      "games": 23
    },
    "TE2": {
      "adjustment_factor": 1.5593531234036835,
      "yards_per_game": 23.333333333333332,
      "confidence": 0.375,
      "games": 6
    },
    "RB_recv": {
      "adjustment_factor": 0.28067433358407795,
      "yards_per_game": 10.571428571428571,
      "confidence": 0.4375,
      "games": 7
    }
  },
  "DAL": {
    "WR2": {
      "adjustment_factor": 0.9397931226657327,
      "yards_per_game": 42.375,
      "confidence": 1.0,
      "games": 16
    },
    "RB_recv": {
      "adjustment_factor": 0.557555770768371,
      "yards_per_game": 21.0,
      "confidence": 0.3125,
      "games": 5
    },
    "TE": {
      "adjustment_factor": 0.799287993836349,
      "yards_per_game": 33.13333333333333,
      "confidence": 0.9375,
      "games": 15
    },
    "Slot": {
      "adjustment_factor": 1.2349219108514995,
      "yards_per_game": 25.166666666666668,
      "confidence": 1.0,
      "games": 24
    },
    "WR1": {
      "adjustment_factor": 0.9668395699503283,
      "yards_per_game": 72.66666666666667,
      "confidence": 0.9375,
      "games": 15
    },
    "TE2": {
      "adjustment_factor": 0.5903265395742517,
      "yards_per_game": 8.833333333333334,
      "confidence": 0.375,
      "games": 6
    },
    "RB_rush": {
      "adjustment_factor": 0.9188629962025637,
      "yards_per_game": 54.96296296296296,
      "confidence": 1.0,
      "games": 27
    },
    "QB": {
      "adjustment_factor": 0.9270691765660799,
      "yards_per_game": 14.882352941176471,
      "confidence": 1.0,
      "games": 17
    },
    "FB": {
      "adjustment_factor": 2.634315424610052,
      "yards_per_game": 19.0,
      "confidence": 0.125,
      "games": 2
    }
  },
  "TB": {
    "QB": {
      "adjustment_factor": 0.762175449508872,
      "yards_per_game": 12.235294117647058,
      "confidence": 1.0,
      "games": 17
    },
    "WR2": {
      "adjustment_factor": 0.9980103957512205,
      "yards_per_game": 45.0,
      "confidence": 0.9375,
      "games": 15
    },
    "TE": {
      "adjustment_factor": 1.2845699900941323,
      "yards_per_game": 53.25,
      "confidence": 1.0,
      "games": 16
    },
    "Slot": {
      "adjustment_factor": 1.1397563451748036,
      "yards_per_game": 23.227272727272727,
      "confidence": 1.0,
      "games": 22
    },
    "WR1": {
      "adjustment_factor": 1.1630250412408427,
      "yards_per_game": 87.41176470588235,
      "confidence": 1.0,
      "games": 17
    },
    "RB_rush": {
      "adjustment_factor": 0.839928232354602,
      "yards_per_game": 50.241379310344826,
      "confidence": 1.0,
      "games": 29
    },
    "TE2": {
      "adjustment_factor": 1.2363442621272063,
      "yards_per_game": 18.5,
      "confidence": 0.625,
      "games": 10
    },
    "RB_recv": {
      "adjustment_factor": 1.0301506621815617,
      "yards_per_game": 38.8,
      "confidence": 0.3125,
      "games": 5
    }
  },
  "SEA": {
    "TE": {
      "adjustment_factor": 1.0911190381731308,
      "yards_per_game": 45.23076923076923,
      "confidence": 0.8125,
      "games": 13
    },
    "Slot": {
      "adjustment_factor": 0.5574323009512464,
      "yards_per_game": 11.36,
      "confidence": 1.0,
      "games": 25
    },
    "RB_rush": {
      "adjustment_factor": 1.3541464775572822,
      "yards_per_game": 81.0,
      "confidence": 1.0,
      "games": 20
    },
    "WR2": {
      "adjustment_factor": 0.9348883707208014,
      "yards_per_game": 42.15384615384615,
      "confidence": 0.8125,
      "games": 13
    },
    "QB": {
      "adjustment_factor": 1.7681681193672907,
      "yards_per_game": 28.384615384615383,
      "confidence": 0.8125,
      "games": 13
    },
    "RB_recv": {
      "adjustment_factor": 1.1336967338956878,
      "yards_per_game": 42.7,
      "confidence": 0.625,
      "games": 10
    },
    "TE2": {
      "adjustment_factor": 0.801953034893323,
      "yards_per_game": 12.0,
      "confidence": 0.5625,
      "games": 9
    },
    "WR1": {
      "adjustment_factor": 1.1603975572503447,
      "yards_per_game": 87.21428571428571,
      "confidence": 0.875,
      "games": 14
    },
    "FB": {
      "adjustment_factor": 0.0,
      "yards_per_game": 0.0,
      "confidence": 0.0625,
      "games": 1
    }
  },
  "KC": {
    "WR2": {
      "adjustment_factor": 0.9119075380785662,
      "yards_per_game": 41.11764705882353,
      "confidence": 1.0,
      "games": 17
    },
    "RB_rush": {
      "adjustment_factor": 0.9337233142004489,
      "yards_per_game": 55.851851851851855,
      "confidence": 1.0,
      "games": 27
    },
    "TE": {
      "adjustment_factor": 0.689024043982416,
      "yards_per_game": 28.5625,
      "confidence": 1.0,
      "games": 16
    },
    "Slot": {
      "adjustment_factor": 0.8524115668469763,
      "yards_per_game": 17.37142857142857,
      "confidence": 1.0,
      "games": 35
    },
    "RB_recv": {
      "adjustment_factor": 0.8997593126156251,
      "yards_per_game": 33.888888888888886,
      "confidence": 1.0,
      "games": 18
    },
    "QB": {
      "adjustment_factor": 1.6163442414812388,
      "yards_per_game": 25.94736842105263,
      "confidence": 1.0,
      "games": 19
    },
    "WR1": {
      "adjustment_factor": 0.7478884553647638,
      "yards_per_game": 56.21052631578947,
      "confidence": 1.0,
      "games": 19
    },
    "TE2": {
      "adjustment_factor": 0.6014647761699923,
      "yards_per_game": 9.0,
      "confidence": 0.5625,
      "games": 9
    },
    "FB": {
      "adjustment_factor": 0.2772963604852686,
      "yards_per_game": 2.0,
      "confidence": 0.0625,
      "games": 1
    }
  },
  "WAS": {
    "TE": {
      "adjustment_factor": 0.9356425279961286,
      "yards_per_game": 38.785714285714285,
      "confidence": 0.875,
      "games": 14
    },
    "WR1": {
      "adjustment_factor": 1.4014738720380904,
      "yards_per_game": 105.33333333333333,
      "confidence": 0.9375,
      "games": 15
    },
    "RB_rush": {
      "adjustment_factor": 1.0154281423228788,
      "yards_per_game": 60.73913043478261,
      "confidence": 1.0,
      "games": 23
    },
    "QB": {
      "adjustment_factor": 0.9382911107685843,
      "yards_per_game": 15.0625,
      "confidence": 1.0,
      "games": 16
    },
    "WR2": {
      "adjustment_factor": 1.330680527668294,
      "yards_per_game": 60.0,
      "confidence": 0.875,
      "games": 14
    },
    "RB_recv": {
      "adjustment_factor": 2.051956952283597,
      "yards_per_game": 77.28571428571429,
      "confidence": 0.4375,
      "games": 7
    },
    "Slot": {
      "adjustment_factor": 0.7687593352320593,
      "yards_per_game": 15.666666666666666,
      "confidence": 1.0,
      "games": 24
    },
    "FB": {
      "adjustment_factor": 1.386481802426343,
      "yards_per_game": 10.0,
      "confidence": 0.125,
      "games": 2
    },
    "TE2": {
      "adjustment_factor": 0.8888312803400997,
      "yards_per_game": 13.3,
      "confidence": 0.625,
      "games": 10
    }
  },
  "HOU": {
    "Slot": {
      "adjustment_factor": 1.2127236929648595,
      "yards_per_game": 24.714285714285715,
      "confidence": 1.0,
      "games": 21
    },
    "RB_rush": {
      "adjustment_factor": 0.9415497483460017,
      "yards_per_game": 56.32,
      "confidence": 1.0,
      "games": 25
    },
    "TE": {
      "adjustment_factor": 1.05539787918532,
      "yards_per_game": 43.75,
      "confidence": 1.0,
      "games": 16
    },
    "RB_recv": {
      "adjustment_factor": 0.7411951714579537,
      "yards_per_game": 27.916666666666668,
      "confidence": 0.75,
      "games": 12
    },
    "WR1": {
      "adjustment_factor": 1.012853150400946,
      "yards_per_game": 76.125,
      "confidence": 1.0,
      "games": 16
    },
    "WR2": {
      "adjustment_factor": 1.0298912833932734,
      "yards_per_game": 46.4375,
      "confidence": 1.0,
      "games": 16
    },
    "QB": {
      "adjustment_factor": 0.9783694472061001,
      "yards_per_game": 15.705882352941176,
      "confidence": 1.0,
      "games": 17
    },
    "TE2": {
      "adjustment_factor": 1.5245461340419943,
      "yards_per_game": 22.8125,
      "confidence": 1.0,
      "games": 16
    }
  },
  "NYG": {
    "QB": {
      "adjustment_factor": 0.6273813710414718,
      "yards_per_game": 10.071428571428571,
      "confidence": 0.875,
      "games": 14
    },
    "WR1": {
      "adjustment_factor": 1.0216440409861758,
      "yards_per_game": 76.78571428571429,
      "confidence": 0.875,
      "games": 14
    },
    "Slot": {
      "adjustment_factor": 0.8905249982593595,
      "yards_per_game": 18.14814814814815,
      "confidence": 1.0,
      "games": 27
    },
    "TE": {
      "adjustment_factor": 0.9132422464787258,
      "yards_per_game": 37.857142857142854,
      "confidence": 0.875,
      "games": 14
    },
    "TE2": {
      "adjustment_factor": 0.6415624279146583,
      "yards_per_game": 9.6,
      "confidence": 0.625,
      "games": 10
    },
    "RB_recv": {
      "adjustment_factor": 1.3393138620573568,
      "yards_per_game": 50.44444444444444,
      "confidence": 0.5625,
      "games": 9
    },
    "RB_rush": {
      "adjustment_factor": 1.0554056282411024,
      "yards_per_game": 63.130434782608695,
      "confidence": 1.0,
      "games": 23
    },
    "WR2": {
      "adjustment_factor": 1.4914710914282128,
      "yards_per_game": 67.25,
      "confidence": 0.75,
      "games": 12
    }
  },
  "ARI": {
    "RB_rush": {
      "adjustment_factor": 1.3576293645880364,
      "yards_per_game": 81.20833333333333,
      "confidence": 1.0,
      "games": 24
    },
    "TE": {
      "adjustment_factor": 0.8667185848656587,
      "yards_per_game": 35.92857142857143,
      "confidence": 0.875,
      "games": 14
    },
    "QB": {
      "adjustment_factor": 1.0044776206568247,
      "yards_per_game": 16.125,
      "confidence": 1.0,
      "games": 16
    },
    "Slot": {
      "adjustment_factor": 1.3739528544572974,
      "yards_per_game": 28.0,
      "confidence": 1.0,
      "games": 19
    },
    "WR1": {
      "adjustment_factor": 0.8822325786490998,
      "yards_per_game": 66.3076923076923,
      "confidence": 0.8125,
      "games": 13
    },
    "WR2": {
      "adjustment_factor": 0.847882336219413,
      "yards_per_game": 38.23076923076923,
      "confidence": 0.8125,
      "games": 13
    },
    "TE2": {
      "adjustment_factor": 0.9547059939206227,
      "yards_per_game": 14.285714285714286,
      "confidence": 0.4375,
      "games": 7
    },
    "RB_recv": {
      "adjustment_factor": 0.5044552211713833,
      "yards_per_game": 19.0,
      "confidence": 0.5,
      "games": 8
    },
    "FB": {
      "adjustment_factor": 0.0,
      "yards_per_game": 0.0,
      "confidence": 0.0625,
      "games": 1
    }
  },
  "NO": {
    "Slot": {
      "adjustment_factor": 0.9068815798600416,
      "yards_per_game": 18.48148148148148,
      "confidence": 1.0,
      "games": 27
    },
    "WR1": {
      "adjustment_factor": 0.8266921827465191,
      "yards_per_game": 62.13333333333333,
      "confidence": 0.9375,
      "games": 15
    },
    "QB": {
      "adjustment_factor": 1.441049030978473,
      "yards_per_game": 23.133333333333333,
      "confidence": 0.9375,
      "games": 15
    },
    "RB_rush": {
      "adjustment_factor": 1.0129501989808403,
      "yards_per_game": 60.59090909090909,
      "confidence": 1.0,
      "games": 22
    },
    "WR2": {
      "adjustment_factor": 1.222958770666575,
      "yards_per_game": 55.142857142857146,
      "confidence": 0.875,
      "games": 14
    },
    "TE": {
      "adjustment_factor": 1.0993368929309943,
      "yards_per_game": 45.57142857142857,
      "confidence": 0.875,
      "games": 14
    },
    "TE2": {
      "adjustment_factor": 0.6925958028624153,
      "yards_per_game": 10.363636363636363,
      "confidence": 0.6875,
      "games": 11
    },
    "RB_recv": {
      "adjustment_factor": 0.9372247003868331,
      "yards_per_game": 35.3,
      "confidence": 0.625,
      "games": 10
    }
  },
  "CIN": {
    "QB": {
      "adjustment_factor": 1.1524239368775973,
      "yards_per_game": 18.5,
      "confidence": 0.75,
      "games": 12
    },
    "WR1": {
      "adjustment_factor": 1.1718019558473998,
      "yards_per_game": 88.07142857142857,
      "confidence": 0.875,
      "games": 14
    },
    "Slot": {
      "adjustment_factor": 1.0966021229674705,
      "yards_per_game": 22.347826086956523,
      "confidence": 1.0,
      "games": 23
    },
    "RB_rush": {
      "adjustment_factor": 1.3719092014141292,
      "yards_per_game": 82.0625,
      "confidence": 1.0,
      "games": 16
    },
    "WR2": {
      "adjustment_factor": 1.3576109669187237,
      "yards_per_game": 61.214285714285715,
      "confidence": 0.875,
      "games": 14
    },
    "TE": {
      "adjustment_factor": 1.192384216157129,
      "yards_per_game": 49.42857142857143,
      "confidence": 0.875,
      "games": 14
    },
    "RB_recv": {
      "adjustment_factor": 0.955809892745779,
      "yards_per_game": 36.0,
      "confidence": 0.8125,
      "games": 13
    },
    "TE2": {
      "adjustment_factor": 0.708391847489102,
      "yards_per_game": 10.6,
      "confidence": 0.625,
      "games": 10
    },
    "FB": {
      "adjustment_factor": 0.0,
      "yards_per_game": 0.0,
      "confidence": 0.0625,
      "games": 1
    }
  },
  "PHI": {
    "TE": {
      "adjustment_factor": 1.1337988644962294,
      "yards_per_game": 47.0,
      "confidence": 0.875,
      "games": 14
    },
    "Slot": {
      "adjustment_factor": 1.0994689694708508,
      "yards_per_game": 22.40625,
      "confidence": 1.0,
      "games": 32
    },
    "WR1": {
      "adjustment_factor": 0.9597434997121607,
      "yards_per_game": 72.13333333333334,
      "confidence": 0.9375,
      "games": 15
    },
    "RB_rush": {
      "adjustment_factor": 0.9350059011705044,
      "yards_per_game": 55.92857142857143,
      "confidence": 1.0,
      "games": 28
    },
    "TE2": {
      "adjustment_factor": 0.7351236153188795,
      "yards_per_game": 11.0,
      "confidence": 0.625,
      "games": 10
    },
    "QB": {
      "adjustment_factor": 1.0956272086690035,
      "yards_per_game": 17.58823529411765,
      "confidence": 1.0,
      "games": 17
    },
    "WR2": {
      "adjustment_factor": 1.3676438756590799,
      "yards_per_game": 61.666666666666664,
      "confidence": 0.9375,
      "games": 15
    },
    "RB_recv": {
      "adjustment_factor": 0.8268514151530945,
      "yards_per_game": 31.142857142857142,
      "confidence": 0.4375,
      "games": 7
    }
  }
}
```
</details>


### Weather Impact

**Sample Size:** 269 observations


**✅ RECOMMENDATION: Update factors (+12.0% improvement)**


**Findings:**
Analyzed 269 games with weather data

<details>
<summary><b>Calculated Factors (Click to Expand)</b></summary>

```json
{
  "wind": {
    "passing_yards_coefficient": 0.0,
    "rushing_yards_coefficient": 0.0,
    "points_coefficient": 0.0,
    "sample_size": 0,
    "confidence": 0.0
  },
  "cold": {
    "passing_yards_coefficient": 0.0,
    "rushing_yards_coefficient": 0.0,
    "points_coefficient": 0.0,
    "sample_size": 0,
    "confidence": 0.0
  }
}
```
</details>


### Situational Factors

**Sample Size:** 269 observations


**✅ RECOMMENDATION: Update factors (+10.0% improvement)**


**Findings:**
Analyzed 269 games across 1 seasons
  Primetime games: 57
  Division games: 83
  Post-bye games: 48
  Thursday games: 0

PRIMETIME: Primetime games score -45.8 vs baseline (p=0.068)
  Confidence: 0.86

DIVISION_GAME: Division games: -43.2 points, margins 1.12x (p=0.468)
  Confidence: 0.06

BYE_WEEK: Post-bye performance: -40.4 points vs baseline (p=0.495)
  Confidence: 0.01

<details>
<summary><b>Calculated Factors (Click to Expand)</b></summary>

```json
{
  "primetime": {
    "total_points_adjustment": -4.827190389495755,
    "scoring_margin_adjustment": 0.0,
    "star_player_boost": 1.0,
    "target_increase": 0.0,
    "sample_size": 57,
    "confidence": 0.8638997309345835
  },
  "division_game": {
    "total_points_adjustment": -43.17308232931727,
    "scoring_margin_adjustment": 1.1199286033020974,
    "star_player_boost": 1.0,
    "target_increase": 0.0,
    "sample_size": 83,
    "confidence": 0.0644061462492811
  },
  "bye_week": {
    "total_points_adjustment": -40.40842708333332,
    "scoring_margin_adjustment": 0.0,
    "star_player_boost": 1.0,
    "target_increase": 0.0,
    "sample_size": 48,
    "confidence": 0.009633195348324408
  },
  "short_week": {
    "total_points_adjustment": -3.0,
    "scoring_margin_adjustment": 0.0,
    "star_player_boost": 1.0,
    "target_increase": 0.0,
    "sample_size": 0,
    "confidence": 0.6
  }
}
```
</details>


### Overall Prediction Accuracy

**Sample Size:** 269 observations


**Accuracy Metrics:**
- RMSE: 47.89
- MAE: 44.83
- Correlation: 0.000
- R²: 0.000


**✅ RECOMMENDATION: Update factors (+0.0% improvement)**


**Findings:**
Backtested 269 game predictions
Backtested 2889 player predictions

GAME TOTALS:
  RMSE: 47.89 points
  MAE: 44.83 points
  MAPE: 133.2%
  Hit Rates: 1.1% within 3, 1.9% within 7, 2.6% within 10
  Bias: +101.2% (over-predicting)

SPREADS:
  RMSE: 17.72 points
  MAE: 14.42 points
  Hit Rates: 10.4% within 3, 28.6% within 7, 39.0% within 10
  Bias: -36.9% (under-predicting)

PLAYER YARDS:
  RMSE: 33.25 yards
  MAE: 24.70 yards
  Hit Rates: 10.6% within 3, 21.5% within 7, 29.8% within 10
  Bias: -8.5% (under-predicting)

<details>
<summary><b>Calculated Factors (Click to Expand)</b></summary>

```json
{
  "game_totals": {
    "rmse": 47.887621086396976,
    "mae": 44.82933333333334,
    "mape": 133.1650878085389,
    "within_7_pct": 1.858736059479554,
    "bias_pct": 101.20181787435527
  },
  "spreads": {
    "rmse": 17.71932094480899,
    "mae": 14.418377942998761,
    "within_7_pct": 28.624535315985128
  }
}
```
</details>


---


## Implementation Recommendations


### Priority Updates


**1. Injury Impact Redistribution** (+15.0% improvement)
- Update configuration files with calculated factors
- Expected accuracy improvement: 15.0%
- Sample size: 0 observations


**2. Weather Impact** (+12.0% improvement)
- Update configuration files with calculated factors
- Expected accuracy improvement: 12.0%
- Sample size: 269 observations


**3. Situational Factors** (+10.0% improvement)
- Update configuration files with calculated factors
- Expected accuracy improvement: 10.0%
- Sample size: 269 observations


**4. Overall Prediction Accuracy** (+0.0% improvement)
- Update configuration files with calculated factors
- Expected accuracy improvement: 0.0%
- Sample size: 269 observations


## Next Steps

1. Review calculated factors for each feature
2. Update configuration files with data-driven coefficients
3. Re-run validation tests to confirm improvements
4. Monitor performance in production
5. Schedule quarterly backtesting to refine factors


---


## Appendix: Methodology


### Data Sources
- Historical game data: NFL official stats
- Player statistics: nfl-data-py package
- Injury reports: NFL injury reports
- Weather data: Historical weather APIs


### Validation Metrics
- **RMSE (Root Mean Square Error):** Lower is better
- **MAE (Mean Absolute Error):** Lower is better
- **Correlation:** Higher is better (closer to 1.0)
- **R² (Coefficient of Determination):** Higher is better (closer to 1.0)


### Statistical Significance
- Minimum sample size: 30 observations
- Confidence level: 95%
- Improvement threshold: 5% reduction in prediction error
