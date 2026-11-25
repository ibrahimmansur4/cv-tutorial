# Sample asset provenance

`input/sample-face.png` is an AI-generated, entirely fictional portrait made
with OpenAI's built-in image-generation tool on 2026-10-04. It is included only
as a reproducible positive input for this face-detection tutorial; it does not
depict or identify a real person.

The corresponding files in `output/` were produced by this project's Haar and
DNN commands. Both detected one face. Their single-image FPS overlays are smoke
test observations, not benchmark results.

## Generation prompt

```text
Use case: photorealistic-natural
Asset type: license-safe sample input for an OpenCV face-detection tutorial and automated demo
Primary request: one entirely fictional adult person facing the camera, suitable for reliable frontal-face detection
Scene/backdrop: plain neutral light-gray studio background
Subject: a single adult with their full face clearly visible, natural symmetrical neutral expression, eyes open, no glasses, no hat, no hair covering the face
Style/medium: photorealistic natural portrait photograph with realistic skin texture
Composition/framing: centered head-and-shoulders portrait, eye-level, face occupies roughly one third of the square frame, generous margin around the head
Lighting/mood: soft even frontal daylight, moderate contrast, no harsh shadows
Constraints: fictional person only; no resemblance request; no text; no logos; no watermark; no extra people; no occlusion; natural color balance
Avoid: profile angle, dramatic lighting, face crop, beauty retouching, stylization, background clutter
```
