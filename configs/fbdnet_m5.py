model = dict(
    name="FBDNet-M5",
    backbone=dict(
        type="mae_vit_backbone",
        img_size=512,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        out_indices=(2, 5, 8, 11),
    ),
    decode_head=dict(
        type="FBDNetHead",
        in_channels=[768, 768, 768, 768],
        channels=256,
        num_classes=2,
        edge_in_index=0,
        edge_mid_channels=128,
        dist_mid_channels=64,
        band_width=3,
        boundary_loss_weight=0.10,
        edge_loss_weight=0.30,
        distance_loss_weight=0.20,
        distance_alpha=3.0,
        smooth_l1_beta=0.1,
        skeleton_weight=5.0,
        buffer_radius=3,
    ),
)

pretraining = dict(
    method="FGE-MAE",
    patch_size=16,
    loss_freq_weight=0.5,
    gabor=dict(scales=(3, 5, 7), num_orientations=4, normalize=True),
)

decoding = dict(
    semantic_thr=0.5,
    boundary_thr=0.30,
    boundary_dilate=1,
    open_radius=5,
    min_island_area=2000,
    max_hole_area=20000,
    sigma=1.5,
    peak_thr=0.30,
    min_dist=15,
    max_markers=50,
    min_area_multi=50000,
    min_sep=0.20,
)
