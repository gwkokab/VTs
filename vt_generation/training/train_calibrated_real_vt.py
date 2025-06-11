import h5py
from gwkokab.parameters import (
    PRIMARY_MASS_SOURCE,
    SECONDARY_MASS_SOURCE,
)
from gwkokab.vts import (
    PopModelsCalibratedVolumeTimeSensitivity,
    train_regressor,
)
from jax import numpy as jnp, random as jrd


param_names = [
    PRIMARY_MASS_SOURCE.name,
    SECONDARY_MASS_SOURCE.name,
]
key = jrd.PRNGKey(0)
vt = PopModelsCalibratedVolumeTimeSensitivity(
    param_names,
    "./vt_aLIGO140MpcT1800545_BBH+BNS+NSBH_fine_m1_m2_a1z_a2z.hdf5",
    zero_spin=True,
    coeffs=[
        133.65628620462888,
        0.824063817123629,
        0.022642664895751174,
        1.635852123849066,
        -0.030873543735042964,
        1625.870793745771,
    ],
    basis="quadratic_Mc_eta",
    batch_size=200000,
)


key1, key2 = jrd.split(key, 2)
m1 = jrd.uniform(key1, shape=(2000,), minval=1.0, maxval=200)
m2 = jrd.uniform(key2, shape=(2000,), minval=1.0, maxval=200)

xx = jnp.stack(
    jnp.meshgrid(
        m1,
        m2,
        indexing="ij",
    ),
    axis=-1,
)

xx = xx.reshape(-1, xx.shape[-1])

logVT_fn = vt.get_mapped_logVT()

VT = jnp.expand_dims(jnp.exp(logVT_fn(xx)), axis=-1)
print(VT.shape)
yy = jnp.concatenate([xx, VT], axis=-1)


dataset_file = "dataset_calibrated_vt_aLIGO140MpcT1800545_BBH+BNS+NSBH_fine_m1_m2.hdf5"

with h5py.File(dataset_file, "w") as f:
    for i in range(len(param_names)):
        f.create_dataset(param_names[i], data=yy[..., i])
    f.create_dataset("VT", data=yy[..., -1])


checkpoint_path = "neural_calibrated_vt_aLIGO140MpcT1800545_BBH+BNS+NSBH_fine_m1_m2.hdf5"

train_regressor(
    input_keys=param_names,
    output_keys=["VT"],
    width_size=128,
    depth=4,
    batch_size=1024,
    data_path=dataset_file,
    checkpoint_path=checkpoint_path,
    epochs=1000,
    validation_split=0.1,
    learning_rate=0.001,
)
