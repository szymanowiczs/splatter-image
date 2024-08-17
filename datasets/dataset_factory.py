def get_dataset(cfg, name):
    if cfg.data.category == "cars" or cfg.data.category == "chairs":
        from .srn import SRNDataset
        return SRNDataset(cfg, name)
    elif cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
        from .co3d import CO3DDataset
        return CO3DDataset(cfg, name)
    elif cfg.data.category == "nmr":
        from .nmr import NMRDataset
        return NMRDataset(cfg, name)
    elif cfg.data.category == "objaverse":
        from .objaverse import ObjaverseDataset
        return ObjaverseDataset(cfg, name)
    elif cfg.data.category == "gso":
        from .gso import GSODataset
        return GSODataset(cfg, name)
