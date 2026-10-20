# Experiment set v1: frozen inputs from the method repo

Read by `experiment_score_colab.ipynb`. Copied byte for byte with `git show` from
`mixture-motion-transfer` at commit `6714c64` (2026-10-20, "Grader reply: Q1 answered from the record").
Kept here so the scorer reads a fixed record without needing a second repo checkout.

| File | Source path in the method repo | sha256 |
|---|---|---|
| `experiment_video_hashes.yaml` | `configs/experiment_video_hashes.yaml` | `a177b43d552bd491c3029139040a622f91e52d0283609cb190f1ea8746a48769` |
| `m3_set_records.json` | `measurements/experiment/m3_set_records.json` | `a194f1c1191a6abd22d6adaa584121646c8328489a63d33486e658b89c4fdc68` |
| `contact_labels.csv` | `measurements/census/contact_labels.csv` | `35fbffad49706ec7ec9272573a146afb0bc657398c9367bb900dfbd4b32bea5c` |
| `source_geometry.json` | `measurements/experiment/source_geometry.json` | `167c2656094294f2d7dcd8c2e7eddc64eaac1a75448bb3f19cd0d642ef5a1b8e` |

What each one is used for:

- **Hashes**: every reference is asserted by content (24 frames, sha256), never by folder (brief §1).
- **Set records**: `footprint_fail`, `depth_fail` and `latent_mask_overlap_cells` per pair (D8 flag columns).
  The committed copy has no `flags` block. `kreg_nonpositive` and `heading_unchecked` are therefore
  read from the method's Drive collector output or caches (see notebook Cell 7). If neither is there,
  they are written as `not_provided`.
- **Contact labels**: John's blind labels (2026-10-19), the `contact_label` column.
- **Source geometry**: Q1 mapping rule (a) on all 15 pairs. The Wan frame is the whole 720x480 cut
  resized to 832x480: `u_cut = u_wan * 720/832`, `v_cut = v_wan`. Kept as the record of the rule the
  notebook applies.

Updating any of these files means a new set version (`experiment_set_v2/`), not an edit here.
