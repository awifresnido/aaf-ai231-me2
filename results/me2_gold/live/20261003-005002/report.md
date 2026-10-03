# VCM benchmark - 202453069 - 20261003-005002

Wake word: **Hey Rhasspy** - trials: 202 with the wake word + 16 without - shuffle seed: 15206 - connection: ssh - holdout: huggingface

Pi log file: ~/vcm_benchmark/vcm_20261002-101622.log

Mic check (Pi input default): signal-to-noise 60.0 dB, laptop speech -120.0 dBFS, room noise -180.0 dBFS - not heard

## At a glance

|                                   | overall     | real voice  | synthetic voice |
|-----------------------------------|-------------|-------------|-----------------|
| intent accuracy (19)              | 80.7%       | 74.0%       | 86.8%           |
| command accuracy (93)             | 80.7%       | 74.0%       | 86.8%           |
| false accept (out of scope fired) | 6.2% (1/16) | 0.0% (0/10) | 16.7% (1/6)     |
| false reject (command ignored)    | 20.4%       | 29.1%       | 13.0%           |
| false wake (no wake word, fired)  | 0.0% (0/16) | 0.0% (0/5)  | 0.0% (0/11)     |
| slot exact                        | 100.0%      | 100.0%      | 100.0%          |
| latency p95                       | 3.18 s      | 3.05 s      | 3.22 s          |

**Pi:** real-time factor 0.008 (p95 0.010), inference 31 ms, CPU temp max 58.4 C, runtime CPU 22% mean, runtime RAM 467 MB peak

# Detailed metrics

## Classification

| metric                                            | 19 intents (+reject) | 93 commands (+reject) |
|---------------------------------------------------|----------------------|-----------------------|
| accuracy                                          | 80.7%                | 80.7%                 |
| balanced accuracy                                 | 77.5%                | 79.7%                 |
| precision (macro)                                 | 95.2%                | 95.7%                 |
| recall (macro)                                    | 77.5%                | 79.7%                 |
| F1 (macro)                                        | 82.8%                | 84.6%                 |
| F2 (macro)                                        | 78.9%                | 81.2%                 |
| false accept rate (OOS fired)                     | 6.2%                 | 6.2%                  |
| false reject rate (in-scope silent/rejected)      | 20.4%                | 20.4%                 |
| misfire rate (wrong command fired)                | 0.0%                 | 0.0%                  |
| accuracy 95% CI                                   | [75-86%]             | [75-86%]              |
| false accept 95% CI                               | [1-28%] (1/16)       | [1-28%]               |
| false wake rate (command without wake word fired) | 0.0% [0-19%] (0/16)  | 0.0% [0-19%] (0/16)   |

Responses: 91.1% of trials fired a command; no response: 18; extra fires: 0; wake detect rate: 92.1%

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

| metric                         | overall        | real voice     | synthetic voice |
|--------------------------------|----------------|----------------|-----------------|
| clips (with wake word)         | 202            | 96             | 106             |
| **19 intents** accuracy        | 80.7% [75-86%] | 74.0% [64-82%] | 86.8% [79-92%]  |
| balanced accuracy              | 77.5%          | 72.1%          | 81.0%           |
| F1 (macro)                     | 82.8%          | 78.2%          | 83.2%           |
| F2 (macro)                     | 78.9%          | 73.4%          | 81.3%           |
| false accept rate              | 6.2% (1/16)    | 0.0% (0/10)    | 16.7% (1/6)     |
| false reject rate              | 20.4%          | 29.1%          | 13.0%           |
| misfire rate                   | 0.0%           | 0.0%           | 0.0%            |
| **93 commands** accuracy       | 80.7%          | 74.0%          | 86.8%           |
| balanced accuracy              | 79.7%          | 71.3%          | 86.5%           |
| F1 (macro)                     | 84.6%          | 70.6%          | 85.9%           |
| F2 (macro)                     | 81.2%          | 70.9%          | 86.2%           |
| misfire rate                   | 0.0%           | 0.0%           | 0.0%            |
| slot exact (intent right)      | 100.0% (n=91)  | 100.0% (n=33)  | 100.0% (n=58)   |
| latency p50 / p95              | 2.38 / 3.18 s  | 2.28 / 3.05 s  | 2.48 / 3.22 s   |
| false wake rate (no wake word) | 0.0% (0/16)    | 0.0% (0/5)     | 0.0% (0/11)     |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

| intent          | n  | exact  | mean abs error | mean rel error | phonetic dist | char dist |
|-----------------|----|--------|----------------|----------------|---------------|-----------|
| ALARM           | 16 | 100.0% | 0.0 min        | 0.000          | 0.000         | 0.000     |
| BRIGHTNESS      | 15 | 100.0% | 0.0 %          | 0.000          | 0.000         | 0.000     |
| COLOR           | 14 | 100.0% | -              | -              | 0.000         | 0.000     |
| CREATE_REMINDER | 17 | 100.0% | -              | -              | 0.000         | 0.000     |
| TEMPERATURE     | 14 | 100.0% | 0.0 deg        | 0.000          | 0.000         | 0.000     |
| TIMER           | 15 | 100.0% | 0.0 s          | 0.000          | 0.000         | 0.000     |
| ALL             | 91 | 100.0% | -              | 0.000          | 0.000         | 0.000     |

## Raspberry Pi

- **Raspberry Pi 4 Model B Rev 1.5**, 4 cores  up to 1800.0 MHz, RAM 3794.7 MB, Debian GNU/Linux 13 (trixie), kernel 6.18.50+rpt-rpi-v8, Python 3.13.5
- packages: -

| metric                                      | mean / p95 / max         |
|---------------------------------------------|--------------------------|
| response latency (command end -> Pi output) | 2.226 / 3.176 / 3.480 s  |
| latency p50 / p99                           | 2.375 / 3.456 s          |
| inference time (Pi-reported)                | 31.4 / 38.9 / 44.2 ms    |
| real-time factor (infer / audio window)     | 0.008 / 0.010 / 0.011    |
| CPU temperature                             | 55.8 / 57.0 / 58.4 C     |
| CPU use, whole Pi                           | 6.1 / 11.5 / 21.3 %      |
| CPU use, your runtime process               | 22.2 / 41.7 / 81.3 %     |
| RAM (RSS), your runtime process             | 465.4 / 466.8 / 466.8 MB |
| RAM used, whole Pi                          | 544.6 / 546.3 / 548.0 MB |
| CPU clock                                   | 1569 / 1800 / 1800 MHz   |
| load average (1 min)                        | 0.43 / 0.66 / 0.88       |
| runtime CPU-seconds per second of speech    | 2.266                    |
| runtime CPU share of wall time              | 22.1%                    |
| throttling flags seen                       | none                     |
| test wall time                              | 61.6 min                 |

## Most frequent confusions

**intent level:** PAUSE -> REJECT (5); COLOR -> REJECT (4); TEMPERATURE -> REJECT (4); BRIGHTNESS -> REJECT (3); LIGHT_OFF -> REJECT (3); TIMER -> REJECT (3); CALL -> REJECT (2); ALARM -> REJECT (2); PLAY_MUSIC -> REJECT (2); VOLUME_UP -> REJECT (2)

**command level:** Pause audio -> REJECT (2); Kill the lights -> REJECT (2); Pause -> REJECT (2); Make a call -> REJECT (1); Skip song -> REJECT (1); Set an alarm for 9:00 PM -> REJECT (1); Change color to Red -> REJECT (1); Brightness 100 percent -> REJECT (1); Set the temperature to 18 degrees -> REJECT (1); Adjust brightness to 60 percent -> REJECT (1)

## Per-intent scores

| class           | n  | precision | recall | F1     | F2     |
|-----------------|----|-----------|--------|--------|--------|
| ALARM           | 18 | 100.0%    | 88.9%  | 94.1%  | 90.9%  |
| BRIGHTNESS      | 18 | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| CALL            | 6  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| COLOR           | 18 | 100.0%    | 77.8%  | 87.5%  | 81.4%  |
| CREATE_REMINDER | 18 | 100.0%    | 94.4%  | 97.1%  | 95.5%  |
| LIGHT_OFF       | 6  | 75.0%     | 50.0%  | 60.0%  | 53.6%  |
| LIGHT_ON        | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| LIST_REMINDERS  | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| MESSAGE         | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| NEXT            | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| PAUSE           | 6  | 100.0%    | 16.7%  | 28.6%  | 20.0%  |
| PLAY_MUSIC      | 6  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| REJECT          | 16 | 28.3%     | 93.8%  | 43.5%  | 64.1%  |
| STOP            | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| TEMPERATURE     | 18 | 100.0%    | 77.8%  | 87.5%  | 81.4%  |
| TIME            | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| TIMER           | 18 | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| VOLUME_DOWN     | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| VOLUME_UP       | 6  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| WEATHER         | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
