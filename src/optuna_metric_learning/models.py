from pytorch_metric_learning import losses, regularizers

def sample_regularizer(trial):
    reg_type = trial.suggest_categorical("reg_type", ["None", "Lp"])
    if reg_type == "None":
        return {"embedding_regularizer": None, "embedding_reg_weight": None}
    elif reg_type == "Lp":
        p = trial.suggest_categorical("reg_p", [1, 2])
        reg_weight = trial.suggest_loguniform("reg_weight", 1e-6, 1e2)
        return {"embedding_regularizer": regularizers.LpRegularizer(p=p), 
            "embedding_reg_weight": reg_weight}

def triplet_margin_loss(trial, margin_min=0.0, margin_max=2.0, **kwargs):
    margin = trial.suggest_uniform("margin", margin_min, margin_max)

    loss = losses.TripletMarginLoss(
        margin=margin, **sample_regularizer(trial)
    )

    return {"loss": loss}

def contrastive_loss(trial, pos_margin_range=(0.0, 2.0), neg_margin_range=(0.0, 2.0), **kwargs):
    pos_margin = trial.suggest_uniform("pos_margin", *pos_margin_range)
    neg_margin = trial.suggest_uniform("neg_margin", *neg_margin_range)

    loss = losses.ContrastiveLoss(
        pos_margin=pos_margin,
        neg_margin=neg_margin,
        **sample_regularizer(trial)
    )

    return {"loss": loss}

def arcface_loss(trial, num_classes, embedding_size, margin_range=(0.0, 90.0),
    scale_range=(0.01, 100), **kwargs):
    margin = trial.suggest_uniform("arc_margin", *margin_range)
    scale = trial.suggest_uniform("scale", *scale_range)
    
    loss = losses.ArcFaceLoss(
        num_classes=num_classes, embedding_size=embedding_size,
        margin=margin, scale=scale, **sample_regularizer(trial)
    )

    return {"loss": loss, "param": True}

def proxy_nca_loss(trial, num_classes, embedding_size, scale_range=(0.0, 100.0), **kwargs):
    scale = trial.suggest_uniform("scale", *scale_range)
    
    loss = losses.ProxyNCALoss(
        num_classes=num_classes, embedding_size=embedding_size,
        softmax_scale=scale, **sample_regularizer(trial)
    )

    return {"loss": loss, "param": True}

ALL_LOSSES = {
    "Triplet": triplet_margin_loss,
    "Contrastive": contrastive_loss,
    "ArcFace": arcface_loss,
    "ProxyNCA": proxy_nca_loss
}