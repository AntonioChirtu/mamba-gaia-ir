def run_mixer_layers(layers, x, dropout=None, **kwargs):
    """Runs a stack of pre-norm residual mixer blocks (as built by create_block
    or an equivalent Block), returning the combined residual+hidden stream
    before the caller's final norm. Shared by Mamba3 and Mamba2Stack."""
    residual = None
    for layer in layers:
        x, residual = layer(x, residual, **kwargs)
        if dropout is not None:
            x = dropout(x)
    return (x + residual) if residual is not None else x