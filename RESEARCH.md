# Research notes and design rationale

This explains why Posture Coach works the way it does: the evidence behind the posture targets, the choice of vision models, and the measured limitations. Research and testing were done in October 2026.

---

## 1. What the science says about "ideal posture"

### There is no single correct posture
- **No strong evidence exists that one optimal posture exists**, or that avoiding "incorrect" postures prevents back pain. Spinal curves vary naturally between people, and no single curvature is strongly linked to pain. ([Slater, Korakakis, O'Sullivan et al., *JOSPT* 2019, "Sit up straight: time to re-evaluate"](https://pure.ul.ie/en/publications/sit-up-straight-time-to-re-evaluate/); [Korakakis et al., physiotherapist perceptions of optimal posture](https://pure.ul.ie/en/publications/physiotherapist-perceptions-of-optimal-sitting-and-standing-postu/))
- Researchers in the field now emphasize **posture variability**. Sitting rigidly upright all day may cause as much discomfort as slumping. ([NHS South Tees, "There is no correct posture"](https://www.southtees.nhs.uk/?p=20758); [UQ Health](https://health.uq.edu.au/node/171))
- A systematic review (Roffey et al., *Spine*) found strong, consistent evidence of **no association between occupational sitting and low back pain** ([IASP summary](https://www.iasp-pain.org/publications/relief-news/article/back-pain-myths-closing-sale-everything-must-go/)).

### But sustained, extreme head and neck postures do matter
- **Forward head posture (FHP) and neck pain:** a meta-analysis of 15 studies found that adults with neck pain have significantly more FHP than people without neck pain. The link was **not** seen in adolescents, and age is a confounder. The evidence is mixed rather than conclusive. ([Mahmoud et al., *Curr Rev Musculoskelet Med* 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6942109/))
- **Slumped sitting increases neck flexion, forward head translation and cervical muscle activity.** "Thoracic upright" (chest-out, military) sitting raises thoracic muscle activity instead. A relaxed **lumbo-pelvic upright** posture had the most favourable profile. ([Caneiro et al., *Manual Therapy* 2010](https://espace.curtin.edu.au/handle/20.500.11937/3456))
- **ISO 11226** (static working postures) treats a posture held longer than 4 s as "static". It classes head inclination of 0–25° and cervical flexion of 0–25° as neutral, with acceptability depending on holding time. ([standard overview](https://occup-med.biomedcentral.com/articles/10.1186/s12995-020-00286-9/tables/2)) **RULA** scores neck flexion of 10–20° as elevated, and >20° or any extension as higher risk.
- **OSHA workstation guidance:** put the top line of the screen at or slightly below eye level, with the centre of the screen about 15–20° below horizontal eye level, at a viewing distance of 50–100 cm. Leaning in to read small text is a common cause of forward lean. ([OSHA eTools: monitors](https://www.osha.gov/etools/computer-workstations/components/monitors))

### Breaks and movement
- **Cochrane review:** low-quality evidence that extra breaks reduce discomfort; mixed evidence on pain; no clear optimal frequency. ([Cochrane CD012886](https://cochranelibrary.com/cdsr/doi/10.1002/14651858.CD012886))
- **Microbreaks:** 30-second microbreaks every 20–40 minutes reduced perceived discomfort without hurting productivity. ([Cornell summary](https://evidencebasedliving.human.cornell.edu/blog/feeling-stressed-at-work-take-microbreaks); [PMC9603092](https://pmc.ncbi.nlm.nih.gov/articles/PMC9603092))

### Feedback that teaches rather than nags
- **Real-time postural biofeedback** improved sitting posture compared with feedback switched off, and showed immediate improvements in neck-pain populations. ([PostureChair, Carleton](https://curve.carleton.ca/297e4d41-ae13-4740-bccb-5d21ca40265d); [wearable feedback RCT NCT06455774](https://clinicaltrials.gov/study/NCT06455774))
- **Guidance hypothesis (motor learning):** feedback on every trial creates dependence. Reduced or faded feedback forces learners to use their own body sense, which tends to improve retention. In a postural control task, only the 67%-frequency feedback group improved at retention; the 100% group did not. ([PMC10933749](https://pmc.ncbi.nlm.nih.gov/articles/PMC10933749/))

### How this shaped the app
| Evidence | Design decision |
|---|---|
| No universal ideal; spines differ | Compare to **your own calibrated baseline**, not population norms |
| Upright "military" sitting increases muscle load | Calibration coaches a *relaxed, supported* posture; rules are **one-sided**, so sitting taller than baseline is never penalised; messages never say "pull your shoulders back" |
| Static postures matter, brief ones don't | An alert fires only after a deviation is **sustained** (default 30 s, with a grace period and hysteresis) |
| Variation and microbreaks help | **Movement reminder** after ~30 min in one position; **break reminder** after ~50 min at the screen |
| Neck flexion of 15–25° and above is non-neutral | Head-flexion alert at **+15°** beyond baseline (strict/gentle presets scale this) |
| Screen height and distance drive the head and trunk | Messages suggest **setup fixes** (raise the screen, enlarge text); calibration warns if you're looking well down or sitting too close |
| Constant feedback creates dependence | **Faded feedback**: as your 7-day good-posture share rises from 70% to 95%, the alert delay grows up to 2× |
| A baseline can drift toward bad posture | The baseline is **never auto-updated**; only an explicit recalibration changes it |
| Mixed evidence on FHP and pain | The app is framed as awareness and comfort feedback, with a not-medical-advice disclaimer |

---

## 2. Vision models: what was evaluated

### Candidates
| Model | Notes |
|---|---|
| **MediaPipe Pose Landmarker** (BlazePose GHUM; lite/full/heavy) | 33 landmarks including ears, eyes and shoulders, plus metric "world" landmarks; fastest; CPU-only works well; designed for close-range webcam use |
| **MediaPipe Face Landmarker** | 478 landmarks plus a **facial transformation matrix**, giving 3-D head pitch, yaw, roll and distance |
| **RTMPose** via [rtmlib](https://github.com/Tau-J/rtmlib) (ONNX, no mmcv) | Highest 2-D keypoint accuracy in recent benchmarks ([comparison](https://sponet.de/sponet/Record/4097725)); needs a person detector (YOLOX) first |
| YOLO11-pose / MoveNet | Good for multi-person or crowded scenes; higher joint-angle error in validation studies |

Prior art reviewed: [PostureFix](https://github.com/adnanakil/PostureFix) (Apple Vision, calibrate-from-frame), [BatesPosture](https://github.com/wtbates99/batesposture) (MediaPipe, 6-second baseline, cooldown-limited alerts), [PostureGuard](https://hunted.space/product/postureguard) (MediaPipe, local-only), [webcam-ergonomics](https://github.com/johanehr/webcam-ergonomics). All of these use a personal baseline plus thresholds, as this app does. None of them separates whole-body leaning from head-forward movement, gates on head yaw, widens thresholds by measured noise, fades feedback, or rewards movement.

### Measurements on this machine (aarch64 CPU, 640 px analysis width)
| Pipeline | Time per frame |
|---|---|
| MediaPipe Pose (full) + Face Landmarker | **~21 ms** (both models) |
| rtmlib "balanced" (YOLOX-m + RTMPose-m) | ~200–250 ms |

At the default 10 analysed frames per second, MediaPipe uses about 20% of one core.

### Decision
**MediaPipe Pose (full) + Face Landmarker.** Running both covers each model's blind spots with a head-on camera:
- The pose model supplies shoulders, which a face model can't.
- The face model's transformation matrix supplies head pitch and distance. A head-on camera can't measure these from 2-D body keypoints. Face-tracker head angles have been validated to about 2–3° median error ([Cambridge Face Tracker](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5054763/)).

RTMPose was tested specifically for shoulder localisation (§4). It showed the same failure mode at about 10× the cost, so it doesn't justify the extra cost here. The engine is a separate module (`landmarks.py`), so a different backend can be added later.

---

## 3. Measuring posture from a frontal webcam

The clinical gold standard for FHP is the **craniovertebral angle** (CVA): tragus to C7, measured from the side. It is valid and reliable from photographs ([PMC7559098](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7559098/); [Aafreen et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558335)), but **a head-on laptop camera can't see it**. The app uses frontal proxies, each compared with your own baseline:

| Measure | Frontal-view proxy | Why it works |
|---|---|---|
| **Slump / head drop** | Perpendicular eye-to-shoulder-line distance ÷ shoulder width | Falls as the head sinks or juts forward and down; unaffected by camera distance, roll and mirroring |
| **Head forward of shoulders** | Face magnification (from head distance) ÷ shoulder magnification | If the whole trunk leans in, both scale together; if only the head moves forward, the face grows more. This separates true FHP from leaning |
| **Leaning toward screen** | Head distance vs. baseline (from the face matrix, which accounts for rotation) | Unlike raw face size, it isn't fooled by turning your head |
| **Looking down / chin poke** | Head pitch from the face matrix | Pitch relative to the camera, minus the baseline, cancels camera height and tilt |
| **Head tilt / side lean** | Head roll; shoulder-line angle | Direct measurement |
| Head forward (depth) | BlazePose world-z of ears vs. shoulders | Learned from 2-D cues; **shown but not used for alerts** |

Robustness measures:
- Landmarks with visibility below 0.5, or outside the frame, are ignored.
- Face-based rules are skipped while your head is turned more than 35° from baseline.
- Each feature is smoothed with a One Euro filter (Casiez et al., CHI 2012).
- Each threshold is at least 3× the noise measured during calibration.
- Calibration rejects captures with too much movement or poor visibility, and warns about off-centre cameras, a tilted head, looking down, or sitting too close.

Head-pose sign conventions were checked empirically against 2-D landmark geometry on real video: the correlation between pitch and the nose-drop ratio was 0.63–0.95 across clips. The decomposition is also unit-tested.

---

## 4. Frontal-view limitations (measured)

**Shoulder priors under-report head drop.** In synthetic head-drop tests, both BlazePose and RTMPose re-estimated the shoulders *lower* when the head was moved down, even though the torso hadn't moved. This was strongest when the shoulders were low-contrast (a dark jacket) or partly hidden by a laptop. Geometric head drop of 25% was measured as about 12–20%, and in one clip as almost nothing, with shoulder confidence falling below the gate. Mitigations:
- The slump threshold is 10% rather than 15%.
- Head-forward and pitch measures come from the face model, which doesn't depend on shoulders.
- Tests mark the affected clip as a known expected failure.

Other limitations:
- **Shoulders out of view:** with very close laptop cameras the shoulders can be cropped. Slump and forward-head checks then switch off, and calibration tells you to tilt the screen or sit back.
- **Camera moves = invalid baseline:** if you move the laptop or camera, recalibrate. The app can't tell a moved camera from moved posture.
- **Back curvature isn't visible from the front.** Lumbar and thoracic slump are inferred from head and shoulder changes only.
- **Synthetic tests are approximations:** the warped videos (zoom, head magnification, neck compression, rotation) check that features move in the right direction and that alerts fire. They aren't a clinical validation. Use `record` and `evaluate` to measure accuracy on real recordings of yourself.
