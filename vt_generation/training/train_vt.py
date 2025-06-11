from gwkokab import vts

filename = "vt_0p5_200_IMRPhenomPv2_SimNoisePSDaLIGO175MpcT1800545_spin.hdf5"
filepath = "/home/muhammad.zeeshan/projects/asset-store/classical_vts/" + filename


vts.train_regressor(
    data_path=filepath,
    input_keys=["mass_1_source", "mass_2_source","a_1", "a_2", "redshift"],
    output_keys=["VT"],
    width_size=128,
    depth=4,
    batch_size=1024,
    checkpoint_path="neural_" + filename,
    epochs=200,
    validation_split=0.1,
    learning_rate=1e-3,
)
