from gwkokab import vts

filename = "vt_PSDaLIGO175MpcT1800545_m1m2_UniformRedshift.hdf5"
filepath = "/home/muhammad.zeeshan/projects/asset-store/classical_vts/" + filename


vts.train_regressor(
    data_path=filepath,
    input_keys=["mass_1_source", "mass_2_source", "redshift"],
    output_keys=["VT"],
    width_size=256,
    depth=6,
    batch_size=1024,
    checkpoint_path="neural_" + filename,
    epochs=200,
    validation_split=0.1,
    learning_rate=1e-3,
)
