# Data, splits and training

## Data

`data.zip` (release v0.1.0) holds everything `evaluate.py` and the training commands below read,
unzipped as `data/`:

| path | content |
| --- | --- |
| `sam3_<collection>_models/final75.npz`, `final75_holdout.npz` | outlines of the outline split: 60 elliptic Fourier harmonics per leaflet in the canonical frame (base (0, 0), tip (0, 1)), with leaflet and photo ids |
| `sam3_<collection>_veins/raster256_split75.npz`, `raster256_split75_holdout.npz` | vein rasters of the vein split: 256 px, vein order coded by brightness, finer veins 1 px wide, with the outline mask |
| `drawings/veinflow_<collection>.npz` | the VeinFlow drawings of the 500 held-out leaflets of the vein table (seed 0) |

Collections (Chitwood et al. 2013, 2014): `chamber` (growth chamber), `field`, `BILs` (backcross
inbred lines), `tomato_entire`, `tomato_asymmetry`. The leaflets were segmented from the photos of
the [Zenodo archive](https://zenodo.org/records/19656245) with SAM 3, put in the canonical frame by
their base and tip, and their veins traced from the photo into a drainage tree rooted at the base;
that preparation chain is not part of this repository.

## Splits

Both splits are by photo, so leaflets of one photo never sit on both sides; `splits/` lists the
photos of each side per collection.

- **Outlines, `final75`**: 25 % of the photos held out (seed 20261019), drawn only among photos that
  had never been used to compare models. The two outline models were trained once on the other 75 %
  and evaluated once.
- **Veins, `split75`**: 25 % of the photos held out (seed 0). `splits/veins_dev_ids.txt` lists the
  training leaflets kept aside for model selection.

## Training

The commands of the released models (one GPU). Validation and checkpoint selection use part of
the training photos (15 % for the outline models, the leaflets of `splits/veins_dev_ids.txt` for
VeinFlow), never the held-out side.

```bash
BANKS="data/sam3_chamber_models/final75.npz data/sam3_field_models/final75.npz data/sam3_BILs_models/final75.npz data/sam3_tomato_entire_models/final75.npz data/sam3_tomato_asymmetry_models/final75.npz"

# outline: RAE-64 (40 min), then the flow on its codes (10 min)
uv run python -m leafgen.shape.fit --bank $BANKS --output weights/rae64 --model rae --minutes 40 --latent 64 --dataset SAM3_POOL_FINAL75
uv run python -m leafgen.shape.fit --bank $BANKS --output weights/rae64_flow --model rae_flow --rae weights/rae64 --minutes 10 --dataset SAM3_POOL_FINAL75

# outline, second model: flow matching on 128 px masks (4 h)
uv run python -m leafgen.shape.fit --bank $BANKS --output weights/mask_diffusion --model mask_diffusion --minutes 240 --schedule cosine --dataset SAM3_POOL_FINAL75

# veins: VeinFlow at 256 px (3 h)
uv run python -m leafgen.veins.model fit --output weights/veinflow --size 256 --fine --widths 32 64 128 256 256 --batch 8 --split photo --dev splits/veins_dev_ids.txt --schedule cosine --minutes 175
```

Training stops on a time budget, so a retrained model is close to the released one but not
identical. The bend prior (`weights/bend_prior.npz`) was fitted on the Pheno4D scans, which are not
redistributed here: the leaflets were flattened onto the normalised (s, t) domain, their height
fields compressed by PCA to 12 whitened components and modelled by a 2-component Gaussian
mixture chosen by BIC.
