# Model Weights License Notice

The Python source code in this repository is licensed under the MIT License
(see `LICENSE`). This file documents the licensing situation of the model
**weights** that the code is designed to load. The weights are NOT
redistributed by this repository; they are pulled from the Hugging Face Hub
at runtime via `huggingface_hub.snapshot_download`.

If you intend to deploy OmniParser commercially or to embed it in a closed-
source product, **read this document carefully and consult counsel before
shipping**.

## Weight inventory

OmniParser v2.0 ships two distinct model artifacts hosted on Hugging Face:

| Artifact | Hugging Face path | Underlying model family | Effective license |
|---|---|---|---|
| Icon detection (YOLO) | `microsoft/OmniParser-v2.0`, subpath `icon_detect/` | Ultralytics YOLOv8 fine-tuned on UI icons | **AGPL-3.0** (inherited from YOLOv8) |
| Icon captioning (Florence-2) | `microsoft/OmniParser-v2.0`, subpath `icon_caption/` | Microsoft Florence-2 | MIT |
| Icon captioning (BLIP-2, legacy) | `microsoft/OmniParser`, subpath `icon_caption_blip2/` | Salesforce BLIP-2 | BSD-3-Clause |

## AGPL-3.0 implications for the icon detector

Ultralytics releases YOLOv8 under AGPL-3.0. AGPL is a strong copyleft license
with a network-use clause: any service that exposes the model over a network
is treated equivalently to distribution. In practice, this means:

- If your service performs icon detection by loading these weights and serves
  results to remote users (HTTP, RPC, or otherwise), Section 13 of AGPL-3.0
  obliges you to make the **complete corresponding source code** of your
  service available to those users under AGPL-3.0.
- This obligation extends to any code linked to or derived from the AGPL'd
  artifact, which in practice often includes your service's source.
- Internal use by employees of a single legal entity does not trigger the
  network-use clause, but redistribution to customers does.

If AGPL exposure is unacceptable for your project, you have three options:

1. **Purchase a commercial Ultralytics license.** Ultralytics offers
   commercial licensing that releases you from AGPL-3.0 obligations. See
   https://www.ultralytics.com/license.
2. **Replace the icon detector.** Train or substitute a permissively-licensed
   detector (for example, RT-DETR variants under Apache-2.0 or MIT, or a
   custom YOLO-format model trained from scratch and released under your
   chosen license). The OmniParser pipeline accepts any detector that
   implements the protocol declared in `omniparser_core.detection`.
3. **Stay AGPL-compliant.** Accept the obligation to publish your service's
   corresponding source code under AGPL-3.0 to your network users.

## Florence-2 (icon captioning, default)

The Florence-2 captioning weights are MIT-licensed and impose no additional
obligations beyond attribution. This is the default captioning backend in
v2.0 and what the pipeline uses unless you opt into the legacy BLIP-2 path.

## BLIP-2 (icon captioning, legacy)

BLIP-2 is BSD-3-Clause. It is permissively licensed but is not the default
in OmniParser v2.0; the legacy code path may be removed in a future release.

## Compliance checklist

For commercial deployment, confirm each of the following:

- [ ] Your legal team has reviewed AGPL-3.0 implications for the YOLO-derived
      icon detector, OR you have a commercial Ultralytics license, OR you
      have replaced the detector with a permissively-licensed alternative.
- [ ] Your build does not redistribute the weights file (`model.pt`) inside
      a container image, wheel, or installer **without** including the AGPL
      source-availability notice for end users.
- [ ] If you serve detection over a network, you have a public source-code
      offer in place per AGPL Section 13.
- [ ] Your `NOTICE` (or equivalent) preserves attribution to upstream
      OmniParser authors and lists the third-party model licenses in effect.

This document is informational and is not legal advice. Consult qualified
counsel for jurisdiction-specific guidance.
